ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.4.4
# ARG PYTORCH_IMAGE
# ARG TENSORFLOW_IMAGE

ARG OPENCV_VERSION=4.5.3
ARG ONNXRUNTIME_VERSION=1.8.2
ARG CMALE_VERSION=3.21.1

# FROM ${PYTORCH_IMAGE} as pytorch
# FROM ${TENSORFLOW_IMAGE} as tensorflow
FROM ${BASE_IMAGE}


#
# setup environment
#
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"
ARG MAKEFLAGS=-j$(nproc) 

RUN printenv


#
# apt packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-py \
        python3-pip \
        python3-numpy \
        python3-pytest \
        python3-setuptools \
		python3-dev \
		python3-matplotlib \
		build-essential \
        software-properties-common \
		gfortran \
		git \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        libjpeg-dev \
        libpng-dev \
        language-pack-en \
        locales \
        locales-all \
        libprotobuf-dev \
        zlib1g-dev \
        swig \
        vim \
        gdb \
        valgrind \
        libsm6 \
        libxext6 \
        libxrender-dev \
        unzip \
		cmake \
		curl \
		libopenblas-dev \
		liblapack-dev \
		libblas-dev \
		libhdf5-serial-dev \
		hdf5-tools \
		libhdf5-dev \
		zlib1g-dev \
		zip \
		libjpeg8-dev \
		libopenmpi2 \
		openmpi-bin \
		openmpi-common \
		protobuf-compiler \
		libprotoc-dev \
		llvm-9 \
		llvm-9-dev \
		libffi-dev \
		libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

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

#
# pull protobuf-cpp from TF container
#
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp

# COPY --from=tensorflow /usr/local/bin/protoc /usr/local/bin
# COPY --from=tensorflow /usr/local/lib/libproto* /usr/local/lib/
# COPY --from=tensorflow /usr/local/include/google /usr/local/include/google


# #
# # python packages from TF/PyTorch containers
# #
# COPY --from=tensorflow /usr/local/lib/python2.7/dist-packages/ /usr/local/lib/python2.7/dist-packages/
# COPY --from=tensorflow /usr/local/lib/python3.6/dist-packages/ /usr/local/lib/python3.6/dist-packages/

# COPY --from=pytorch /usr/local/lib/python2.7/dist-packages/ /usr/local/lib/python2.7/dist-packages/
# COPY --from=pytorch /usr/local/lib/python3.6/dist-packages/ /usr/local/lib/python3.6/dist-packages/


#
# python pip packages
#
RUN pip3 install --no-cache-dir --ignore-installed pybind11 
RUN pip3 install --no-cache-dir --verbose onnx
RUN pip3 install --no-cache-dir --verbose scipy
RUN pip3 install --no-cache-dir --verbose scikit-learn
RUN pip3 install --no-cache-dir --verbose pandas
RUN pip3 install --no-cache-dir --verbose pycuda
RUN pip3 install --no-cache-dir --verbose numba

#
# CuPy
#
ARG CUPY_VERSION=v9.2.0
ARG CUPY_NVCC_GENERATE_CODE="arch=compute_53,code=sm_53;arch=compute_62,code=sm_62;arch=compute_72,code=sm_72"

RUN git clone -b ${CUPY_VERSION} --recursive https://github.com/cupy/cupy cupy && \
    cd cupy && \
    pip3 install --no-cache-dir fastrlock && \
    python3 setup.py install --verbose && \
    cd ../ && \
    rm -rf cupy


#
# install OpenCV (with CUDA)
# note:  do this after numba, because this installs TBB and numba complains about old TBB
#
ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

RUN mkdir opencv && \
    cd opencv && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${OPENCV_URL} -O ${OPENCV_DEB} && \
    tar -xzvf ${OPENCV_DEB} && \
    dpkg -i --force-depends *.deb && \
    apt-get update && \
    apt-get install -y -f --no-install-recommends && \
    dpkg -i *.deb && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    cd ../ && \
    rm -rf opencv && \
    cp -r /usr/include/opencv4 /usr/local/include/opencv4 && \
    cp -r /usr/lib/python3.6/dist-packages/cv2 /usr/local/lib/python3.6/dist-packages/cv2


# #
# # JupyterLab
# #
# RUN curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
#     apt-get update && \
#     apt-get install -y nodejs && \
#     rm -rf /var/lib/apt/lists/* && \
#     apt-get clean && \
#     pip3 install --no-cache-dir --verbose jupyter jupyterlab==2.2.9 && \
#     jupyter labextension install @jupyter-widgets/jupyterlab-manager
    
# RUN jupyter lab --generate-config
# RUN python3 -c "from notebook.auth.security import set_password; set_password('nvidia', '/root/.jupyter/jupyter_notebook_config.json')"

# CMD /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &> /var/log/jupyter.log" & \
# 	echo "allow 10 sec for JupyterLab to start @ http://$(hostname -I | cut -d' ' -f1):8888 (password nvidia)" && \
# 	echo "JupterLab logging location:  /var/log/jupyter.log  (inside the container)" && \
# 	/bin/bash

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
        --parallel ${MAKEFLAGS} && \
    cd build/Linux/RelWithDebInfo && \
    make install && \
    pip install dist/*
RUN rm -rf /tmp/*
