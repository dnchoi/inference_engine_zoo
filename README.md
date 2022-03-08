# ONNX runtime test using C++

## Docker build
> x86_64
```bash
docker build -f docker/Dockerfile  --no-cache --tag=onnxruntime-cuda:1.8.2 .
```
> amd64 [jetson nano / nx]
```bash
docker build --platform linux/amd64 -f docker/Dockerfile  --no-cache --tag=onnxruntime-cuda:1.8.2 .
```
## Run Docker Container
```bash
docker run -it --gpus device=0 -v $(pwd):/mnt --net host --privileged -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix onnxruntime-cuda:1.8.2 
```



## Install & Dependence
> Dependence
* CMake 3.20.1
* ONNX Runtime 1.8.2
* OpenCV V4
> Install
* Package dependecies
```bash
sudo apt update
sudo apt -y install g++
sudo apt -y install build-essential cmake
sudo apt -y install pkg-config
sudo apt -y install libjpeg-dev libpng-dev
sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev
sudo apt -y install libv4l-dev v4l-utils
sudo apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt -y install libgtk2.0-dev
sudo apt -y install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev  
sudo apt -y install libatlas-base-dev gfortran libeigen3-dev
sudo apt -y install python2.7-dev python3-dev python-numpy python3-numpy
sudo apt -y install libgtk-3-dev

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
OPENCV_VERSION=4.4.0

mkdir -p ~/SDK_Tools/OpenCV_440
cd ~/SDK_Tools/OpenCV_440
wget -O opencv.zip https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.zip
unzip opencv.zip
wget –O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/${OPENCV_VERSION}.zip
mv ${OPENCV_VERSION}.zip opencv_contrib.zip
unzip opencv_contrib.zip
cd opencv-${OPENCV_VERSION}
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_C_COMPILER=/usr/bin/gcc-7 \ <- Your gcc version
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DINSTALL_PYTHON_EXAMPLES=ON \
      -DINSTALL_C_EXAMPLES=ON \
      -DBUILD_DOCS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PACKAGE=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DWITH_TBB=ON \
      -DENABLE_FAST_MATH=1 \
      -DCUDA_FAST_MATH=1 \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1 \ <- Your CUDA version
      -DWITH_CUDA=ON \
      -DWITH_CUBLAS=ON \
      -DWITH_CUFFT=ON \
      -DWITH_NVCUVID=ON \
      -DWITH_IPP=OFF \
      -DWITH_V4L=ON \
      -DWITH_1394=OFF \
      -DWITH_GTK=ON \
      -DWITH_QT=OFF \
      -DWITH_OPENGL=ON \
      -DWITH_OPENCL=OFF \
      -DWITH_EIGEN=ON \
      -DWITH_FFMPEG=ON \
      -DWITH_GSTREAMER=ON \
      -DBUILD_JAVA=OFF \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_NEW_PYTHON_SUPPORT=ON \
      -DOPENCV_SKIP_PYTHON_LOADER=ON \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
      -DOPENCV_ENABLE_NONFREE=ON \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
      -DWITH_CUDNN=ON \
      -DOPENCV_DNN_CUDA=ON \
      -DCUDA_ARCH_BIN='6.0 6.2 7.0 7.5' \
      -DCUDA_ARCH_PTX="" \
      -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so.8.1.1 \
      -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include  \
      -DOPENCV_TEST_DATA_PATH=../opencv_extra-${OPENCV_VERSION}/testdata \
              ../opencv-${OPENCV_VERSION} && \
cmake --build . --parallel ${NUM_JOBS} && \
make install

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

