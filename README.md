# ONNX runtime test using C++


## Install & Dependence
> Dependence
* CMake 3.20.1
* ONNX Runtime 1.8.2
* OpenCV V4
> Install
* Package dependecies
```bash
sudo apt -y install g++
sudo apt -y install build-essential cmake
sudo apt -y install pkg-config
sudo apt -y install libjpeg-dev libpng-dev
sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev
sudo apt -y install lib41-dev v4l-utils
sudo apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt -y install libgtk2.0-dev
sudo apt -y install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev  
sudo apt -y install libatlas-base-dev gfortran libeigen3-dev
sudo apt -y install python2.7-dev python3-dev python-numpy python3-numpy
sudo apt -y install libgtk-3-dev
sudo apt -y install libqt5-dev

sudo apt install -y libgstreamer1.0-0 libgstreamer1.0-dev \
                    gstreamer1.0-tools gstreamer1.0-doc gstreamer1.0-x gstreamer1.0-plugins-base \
                    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly  \
                    gstreamer1.0-alsa gstreamer1.0-libav gstreamer1.0-gl gstreamer1.0-gtk3 \
                    gstreamer1.0-qt5 gstreamer1.0-pulseaudio libgstreamer-plugins-base1.0-dev
```
* CMake
```bash
CMALE_VERSION = 3.21.1

mkdir SDK_Tools
wget https://github.com/Kitware/CMake/releases/download/v${CMALE_VERSION}/cmake-${CMALE_VERSION}-linux-x86_64.sh
bash cmake-${CMALE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
```

* OpenCV
```bash
NUM_JOBS=8

mkdir -p ~/SDK_Tools/OpenCV_440
cd ~/SDK_Tools/OpenCV_440
wget -O opencv.zip https://github.com/Itseez/opencv/archive/4.4.0.zip
unzip opencv.zip
wget –O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/4.4.0.zip
mv 4.4.0.zip opencv_contrib.zip
unzip opencv_contrib.zip
cd opencv-4.4.0
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_C_COMPILER=/usr/bin/gcc-7 \ <- Your gcc version
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_DOCS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PACKAGE=OFF \
-D BUILD_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1 \ <- Your CUDA version
-D WITH_CUDA=ON \
-D WITH_CUBLAS=ON \
-D WITH_CUFFT=ON \
-D WITH_NVCUVID=ON \
-D WITH_IPP=OFF \
-D WITH_V4L=ON \
-D WITH_1394=OFF \
-D WITH_GTK=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_OPENCL=OFF \
-D WITH_EIGEN=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D BUILD_JAVA=OFF \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D OPENCV_SKIP_PYTHON_LOADER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.4.0/modules \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN='6.0 6.2 7.0 7.5' \
-D CUDA_ARCH_PTX="" \
-D CUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so.8.1.1 \
-D CUDNN_INCLUDE_DIR=/usr/local/cuda/include  ..

make -j${NUM_JOBS}
sudo make install

pkg-config --modversion opencv4
pkg-config --libs --cflags opencv4
```

* ONNX runtime
```bash
NUM_JOBS=8
ONNXRUNTIME_VERSION=1.8.2

cd SDK_Tools
git clone --recursive --branch v${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime && \
cd onnxruntime && \
./build.sh \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/x86_64-linux-gnu/ \
    --use_cuda \
    --config RelWithDebInfo \
    --build_shared_lib \
    --build_wheel \
    --skip_tests \
    --parallel ${NUM_JOBS} && \
cd build/Linux/RelWithDebInfo && \
make install && \
pip install dist/*
```

* Bash script
```bash
#!/bin/sh

# commend
# bash build_run.sh run 0 img
# bash build_run.sh run 0 img.png
# bash build_run.sh run 1000 img.png

# bash build_run.sh build 0 img
# bash build_run.sh build 0 img.png
# bash build_run.sh build 1000 img.png


# exit on first error
set -e

BUILD=$1
BATCH=$2
C=$3
W=$4
H=$5
ITER=$6
ACC=$7
OPTI=$8
MODEL=$9
ENGINE=${10}
IMG=${11}

echo "BUILD or RUN : $BUILD"
echo "INPUT BATCH SIZE : $BATCH"
echo "INPUT CHANNEL : $C"
echo "INPUT WIDTH : $W"
echo "INPUT HEIGHT : $H"
echo "ITERATE : $ITER"
echo "ACCELERATOR : $ACC"
echo "OPTIMIZER : $OPTI"
echo "MODEL PATH : $MODEL"
echo "USING ENGINE : $ENGINE"
echo "IMAGE PATH : $IMG"

function main {
    ./build/main $BATCH $C $W $H $ITER $ACC $OPTI $MODEL $ENGINE $IMG
}

if [ $BUILD = "build" ]
then
    echo "Build and Run"

    cd build

    cmake ..

    make all
    cd ..

    sleep 2s
    main
elif [ $BUILD = "run" ]
then
    echo "Run"

    main
fi
```

## Use
* for cmake
```bash
mkdir build; cd build
cmake ..; make all

cd ..

# Params
## 1. [Build and Run] or [Run]
## 2. [Batch size]
## 3. [Input Channel]
## 4. [Input Width]
## 5. [Input Height]
## 6. [Iterate]
## 7. [Accelerator]
## 8. [Optimizer]
## 9. [Model path]
## 10. [Inference engine]
## 11. [Input image] if use zero Matrix -> do not insert '.'
bash build_run.sh build 1 3 112 112 10000 1 0 model.onnx onnx img.png # -> image running
bash build_run.sh build 1 3 112 112 10000 1 0 model.onnx onnx img # -> zero matrix running 
```

* Result : output.csv

| BATCH | CHANNEL | WIDTH | HEIGHT | ITERATE | ACCELERATOR | OPTIMIZER | MODEL | ENGINE | FILE | MICRO | MILLI |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 1 | 3 | 112 | 112 | 100 | 1 | 0 | model.onnx | onnx | img.png | 761.524 | 0.761524 |

