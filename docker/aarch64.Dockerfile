FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG OPENCV_VERSION=4.4.0
ARG ONNXRUNTIME_VERSION=1.8.2
ARG CMALE_VERSION=3.21.1
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt update && apt install -y --no-install-recommends \
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
    wget https://github.com/Kitware/CMake/releases/download/v${CMALE_VERSION}/cmake-${CMALE_VERSION}-linux-aarch64.sh && \
    bash cmake-${CMALE_VERSION}-linux-aarch64.sh --prefix=/usr/local --exclude-subdir --skip-license
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
    mkdir -p build && cd build && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENGL=ON \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=ON \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCUDA_ARCH_BIN='6.0 6.2 7.0 7.5' \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=ON \
        -DINSTALL_TESTS=OFF \
        -DOPENCV_TEST_DATA_PATH=../opencv_extra-${OPENCV_VERSION}/testdata \
        ../opencv-${OPENCV_VERSION} && \
    cmake --build . --parallel 20 && \
    make install
RUN rm -rf /tmp/*

# Install ONNX Runtime
RUN pip install pytest==6.2.1 onnx==1.10.1
RUN cd /tmp && \
    git clone --recursive --branch v${ONNXRUNTIME_VERSION} https://github.com/Microsoft/onnxruntime && \
    cd onnxruntime && \
    ./build.sh \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/aarch64-linux-gnu/ \
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
