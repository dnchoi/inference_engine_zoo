FROM nvcr.io/nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ARG OPENCV_VERSION=4.4.0
ARG ONNXRUNTIME_VERSION=1.8.2
ARG CMALE_VERSION=3.21.1
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        curl \
        libjpeg-dev \
        libpng-dev \
        language-pack-en \
        locales \
        locales-all \
        python3 \
        python3-py \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-pytest \
        python3-setuptools \
        libprotobuf-dev \
        protobuf-compiler \
        zlib1g-dev \
        swig \
        vim \
        gdb \
        valgrind \
        libsm6 \
        libxext6 \
        libxrender-dev \
        cmake \
        unzip
RUN apt-get clean

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools wheel

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMALE_VERSION}/cmake-${CMALE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMALE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
RUN rm -rf /tmp/*

# Install OpenCV
# OpenCV-Python dependencies
RUN apt -y install g++
RUN apt -y install build-essential cmake
RUN apt -y install pkg-config
RUN apt -y install libjpeg-dev libpng-dev
RUN apt -y install libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev
RUN apt -y install v4l-utils
RUN apt -y install libv4l-dev
RUN apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN apt -y install libgtk2.0-dev
RUN apt -y install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev  
RUN apt -y install libatlas-base-dev gfortran libeigen3-dev
RUN apt -y install python2.7-dev python3-dev python-numpy python3-numpy
RUN apt -y install libgtk-3-dev

RUN apt install -y libgstreamer1.0-0 libgstreamer1.0-dev \
                    gstreamer1.0-tools gstreamer1.0-doc gstreamer1.0-x gstreamer1.0-plugins-base \
                    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly  \
                    gstreamer1.0-alsa gstreamer1.0-libav gstreamer1.0-gl gstreamer1.0-gtk3 \
                    gstreamer1.0-qt5 gstreamer1.0-pulseaudio libgstreamer-plugins-base1.0-dev

RUN cd /tmp && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    wget -O opencv_extra.zip https://github.com/opencv/opencv_extra/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    unzip opencv_extra.zip && \
    cd opencv-4.4.0 \
    mkdir build \
    cd build \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_C_COMPILER=/usr/bin/gcc-7 \
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
		-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
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
		-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
        -D OPENCV_TEST_DATA_PATH=../opencv_extra-${OPENCV_VERSION}/testdata \
		-D WITH_CUDNN=ON \
		-D OPENCV_DNN_CUDA=ON \
		-D CUDA_ARCH_BIN='6.0 6.2 7.0 7.5' \
		-D CUDA_ARCH_PTX="" \
		-D CUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so.8.1.1 \
		-D CUDNN_INCLUDE_DIR=/usr/local/cuda/include  .. \
    make -j${NUM_JOBS}} \
	make install
RUN rm -rf /tmp/*

# Install ONNX Runtime
RUN pip install pytest==6.2.1 onnx==1.10.1
RUN cd /tmp && \
    git clone --recursive --branch v${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime && \
    cd onnxruntime && \
    ./build.sh \
        --cuda_home /usr/local/cuda-11.1 \
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
RUN rm -rf /tmp/*
