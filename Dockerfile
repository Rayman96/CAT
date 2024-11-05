# 使用Ubuntu 18.04作为基础镜像
FROM ubuntu:18.04

# 设置环境变量以避免在安装过程中出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装基本工具和依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    wget \
    curl \
    ca-certificates \
    lsb-release \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 安装CMake 3.18.3
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.3/cmake-3.18.3-Linux-x86_64.sh \
    && chmod +x cmake-3.18.3-Linux-x86_64.sh \
    && ./cmake-3.18.3-Linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.18.3-Linux-x86_64.sh

# 添加NVIDIA GPG key
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add -

# 添加NVIDIA CUDA repository
RUN echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" > /etc/apt/sources.list.d/cuda.list

# 安装CUDA 11.7
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-11-7 \
    && rm -rf /var/lib/apt/lists/*

# 设置CUDA环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 复制当前git库的代码到容器中
COPY . /my_project

# 设置工作目录
WORKDIR /my_project

# 编译代码
# 假设你的项目有一个名为build.sh的脚本来编译项目
RUN chmod +x build.sh && ./build.sh

# 设置默认命令
CMD ["bash"]
