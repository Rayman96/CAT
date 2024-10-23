# SEAL库的GPU加速版本

## 准备操作
1. 安装CUDA
    下载安装11.7版本CUDA 
    https://developer.nvidia.com/cuda-11-7-0-download-archive
2. 安装CMake
    安装3.18.3版本cmake


## 编译方法
根目录下执行
```
cmake --GPU_SUPPORT ON -S . -B build
cd build
make 
```

测试：
```
cd build/native/examples
make
build/bin/sealexamples
```

