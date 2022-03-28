#include <iostream>
#include <spdlog/spdlog.h>
#include "configparser.h"
#include <opencv2/opencv.hpp>
#ifdef TENSORRT
#include <frvf_trt.h>
#endif
#ifdef ONNX
#include <frvf_onnx.h>
#endif

#define USECAM false

struct Arg
{
	int _B;
	int _W;
	int _H;
	int _C;
	int _iter;
	int _acc;
	int _opti;
	std::string _model;
	std::string _engine;
}args, *args_ptr;

int main(int argc, char* argv[]){
	spdlog::set_level(spdlog::level::info); 

    Arg parser = args;
    CConfigParser config("config.ini");
	if (config.IsSuccess()) {
		parser._B = config.GetInt("B");
		parser._W = config.GetInt("W");
		parser._H = config.GetInt("H");
		parser._C = config.GetInt("C");
		parser._iter = config.GetInt("ITERATION");
		parser._acc = config.GetInt("ACCELERATOR");
		parser._opti = config.GetInt("OPTIMIZER");
        parser._model = config.GetString("MODEL");
		parser._engine = config.GetString("ENGINE");
	}
	SPDLOG_INFO(parser._model);
	SPDLOG_INFO(parser._engine);
    SPDLOG_INFO(parser._acc);
    SPDLOG_INFO(parser._opti);
    SPDLOG_INFO(parser._B);
    SPDLOG_INFO(parser._C);
    SPDLOG_INFO(parser._W);
    SPDLOG_INFO(parser._H);
#ifdef TENSORRT
	trt_frvf::frvf_trt *trt;
	trt = new trt_frvf::frvf_trt(parser._model.c_str(), parser._acc, parser._opti, parser._B, parser._C, parser._W, parser._H);

#endif
#ifdef ONNX
	onnx_frvf::frvf_onnx *onnx;
	onnx = new onnx_frvf::frvf_onnx(parser._model.c_str(), parser._acc, parser._opti, parser._B, parser._C, parser._W, parser._H);
#endif

#if USECAM
	cv::VideoCapture cap(0);
	cv::Mat frame;
	if(!cap.isOpened()) return 0;
	else{
		while(true){
			cap >> frame;

			if(frame.rows == 0){
				SPDLOG_INFO("rows : {}\tcols : {}",frame.rows, frame.cols);
				continue;
			}
			else{
				float p_time = 0.0;
#ifdef TENSORRT
				p_time = trt->do_inference(frame);
#endif
#ifdef ONNX
				p_time = onnx->do_inference(frame);
#endif
    			SPDLOG_INFO("processing time : {}", p_time);
				// cv::imshow("", frame);
				// if(cv::waitKey(1) == 27) break;
			}
		}
	}
#else
	SPDLOG_INFO("Zero matrix type");
	cv::Mat frame = cv::imread("img.png", 1);
	float p_time = 0.0;
	SPDLOG_INFO("Go");

	for(int i=0; i < parser._iter; i++){
	#ifdef TENSORRT
		p_time = trt->do_inference(frame);
	#elif ONNX
		p_time = onnx->do_inference(frame);
	#endif
		SPDLOG_INFO("Call {} - processing time : {}", i, p_time);
	}
#endif
	return 0;
}