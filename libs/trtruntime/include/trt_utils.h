#ifndef __TRT_UTILS_H__
#define __TRT_UTILS_H__

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include "NvInferRuntimeCommon.h"
#include <spdlog/spdlog.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

static constexpr int CHECK_COUNT = 3;
static constexpr float IGNORE_THRESH = 0.1f;
struct YoloKernel
{
    int width;
    int height;
    float anchors[CHECK_COUNT * 2];
};

static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
static constexpr int CLASS_NUM = 2;  //PVD model 8 MaskFace model 2
static constexpr int INPUT_H = 640;  // yolov5's input height and width must be divisible by 32.
static constexpr int INPUT_W = 640;
static constexpr int LOCATIONS = 4;

struct alignas(float) Detection {
    //center_x center_y w h
    float bbox[LOCATIONS];
    float conf;  // bbox_conf * cls_conf
    float class_id;
};

struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

std::vector<std::string> getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

inline float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

inline bool cmp(const Detection& a, const Detection& b) {
    return a.conf > b.conf;
}

inline void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    
    for (auto it = m.begin(); it != m.end(); it++) {
        // std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            // std::cout<<*item.bbox<<std::endl;
            // for (int que = 0; que < 4; que++)
                // *item.bbox[que] = (int)*item.bbox[que];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

static inline void preprocess_img(float* data, cv::Mat& frame, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (frame.cols*1.0);
    float r_h = input_h / (frame.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * frame.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * frame.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    // std::cout << r_h << "\t" << r_w << "\t" << img.cols << "\t" << img.rows << "\t" << input_w << "\t" << input_h << "\t" << w << "\t" << h << std::endl;

    cv::Mat re(h, w, CV_8UC3);
    cv::resize(frame, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    int i = 0;
    for (int row = 0; row < input_h; ++row) {
        uchar* uc_pixel = out.data + row * out.step;
        for (int col = 0; col < input_w; ++col) {
            data[0 * 3 * input_h * input_w + i] = (float)uc_pixel[2] / 255.0;
            data[0 * 3 * input_h * input_w + i + input_h * input_w] = (float)uc_pixel[1] / 255.0;
            data[0 * 3 * input_h * input_w + i + 2 * input_h * input_w] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
    // cv::imshow("tmp", out);
    // std::cout << out.size() << std::endl;
    // cv::waitKey(1);
    // return out;
    
    // if (frame.empty())
    // {
    //     SPDLOG_CRITICAL("Mat frame is nullptr");
    // }
    // cv::cuda::GpuMat gpu_frame;
    // // upload image to GPU
    // gpu_frame.upload(frame);
    // SPDLOG_ERROR("B");

    // auto input_size = cv::Size(input_w, input_h);
    // SPDLOG_ERROR("B");
    // // resize
    // cv::cuda::GpuMat resized;
    // SPDLOG_ERROR("B");
    // cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    // SPDLOG_ERROR("B");
    // // normalize
    // cv::cuda::GpuMat flt_image;
    // SPDLOG_ERROR("B");
    // resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    // SPDLOG_ERROR("B");
    // cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    // SPDLOG_ERROR("B");
    // cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    // // to tensor
    // std::vector<cv::cuda::GpuMat> chw;
    // for (size_t i = 0; i < 3; ++i)
    // {
    //     chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, data + i * input_w * input_h));
    // }
}

inline std::vector<int> float_int_get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    std::vector<int> _bbox = {l, t, r, b};
    return _bbox;
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}
#endif // __TRT_UTILS_H__

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
{\
    cudaError_t error_code = callstr;\
    if (error_code != cudaSuccess) {\
        std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
        assert(0);\
    }\
}
#define CHECK(status)									\
{														\
    if (status != 0)									\
    {													\
        std::cout << "Cuda failure: " << status;		\
        abort();										\
    }													\
}
#endif

#ifndef __MACROS_H
#define __MACROS_H

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#else

#if defined(_MSC_VER)
#define API __declspec(dllimport)
#else
#define API
#endif
#endif  // API_EXPORTS

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

#endif  // __MACROS_H
