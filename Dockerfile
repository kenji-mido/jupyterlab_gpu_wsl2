# Stage 1: Build OpenCV with CUDA
FROM docker.io/nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

# Set environment variables to minimize interactive prompts and set locale
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Install build dependencies and CUDA libraries
RUN apt-get update && apt-get install --allow-change-held-packages -y --no-install-recommends \
    wget \
    build-essential \
    gcc-10 g++-10 \
    cmake \
    git \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libgtk-3-0 \
    libcanberra-gtk3-module \
    libgl1-mesa-glx \
    libatlas-base-dev \
    gfortran \
    libgl1 \
    python3-dev \
    python3-pip \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies needed for building OpenCV
RUN python3 -m pip install --upgrade pip numpy

# Define OpenCV version
ENV OPENCV_VERSION=4.10.0

# Create a directory for OpenCV
WORKDIR /opt/opencv_build

# Download OpenCV and OpenCV_contrib
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && unzip opencv_contrib.zip && \
    rm opencv.zip opencv_contrib.zip && \
    mv opencv-${OPENCV_VERSION} opencv && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib && \
    rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /opt/opencv_build/opencv/build

# Configure OpenCV with CUDA and set GCC/G++ 10
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_C_COMPILER=gcc-10 \
    -D CMAKE_CXX_COMPILER=g++-10 \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_build/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_GSTREAMER=ON \
    -D WITH_LIBV4L=ON \
    # Disable QT to save space
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_TESTS=OFF \
    ..

# Build and install OpenCV
RUN make -j4 && make install && ldconfig && make clean && \
    rm -rf /opt/opencv_build/opencv && rm -rf /opt/opencv_build/opencv_contrib && \
    apt-get purge -y --auto-remove \
        wget \
        build-essential \
        gcc-10 \
        g++-10 \
        cmake \
        git \
        unzip \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libatlas-base-dev \
        gfortran \
        libgl1 \
        python3-dev \
        python3-pip \
        espeak-ng \
        libespeak-ng1 \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# =============================================================================

FROM docker.io/nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Install common dependencies and Python
RUN apt-get update && \
    apt-get install --allow-change-held-packages -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    emacs \
    wget \
    tzdata \
    libgomp1 \
    software-properties-common \
    x11-apps \
#    libnvidia-ml-dev \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
#    cuda-toolkit-12-4 \
#    libcublas-12-4 \
#    libcublas-dev-12-4 \
    espeak-ng \
    libespeak-ng1 \
    libgtk-3-dev \
    libgtk-3-0 \
    libcanberra-gtk3-module \
    libgl1-mesa-glx \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy OpenCV from the builder stage
COPY --from=builder /usr/local /usr/local

# Verify OpenCV installation
RUN python3 -c "import cv2; print('OpenCV Version:', cv2.__version__)" && \
    python3 -c "import cv2; print('CUDA Enabled Devices:', cv2.cuda.getCudaEnabledDeviceCount())"

# Install TensorFlow
RUN pip3 install tensorflow[and-cuda]==2.16.1

# Install PyTorch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
#     mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
#     add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     libcudnn8 libcudnn8-dev \
#     && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TF_ENABLE_ONEDNN_OPTS=0

# Install ONNX and ONNX Runtime
RUN pip install onnx onnxruntime-gpu

# Install TensorRT
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libnvinfer8=8.* \
    libnvinfer-plugin8=8.* \
    libnvparsers8=8.* \
    libnvonnxparsers8=8.* \
    tensorrt \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# PyCUDA
RUN pip install pycuda

# Verify installations
RUN python -c "import torch, tensorflow, onnx, onnxruntime; print('All frameworks imported successfully')"

COPY gpu_check.py /app/gpu_check.py

# JupyterLab
COPY python/requirements.txt .

RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app

CMD ["/bin/bash"]
