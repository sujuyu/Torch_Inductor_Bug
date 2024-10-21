cmake_minimum_required(VERSION 3.10)
project(threadpool_torchaot)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 ABI 版本为 0
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# 定义一个用于传递 PyTorch 路径的 CMake 变量
set(TORCH_DIR "/home/admin/zy429782/miniforge3/envs/torch231_cuda121/lib/python3.8/site-packages/torch" CACHE PATH "Path to PyTorch")

# 包含头文件目录
include_directories(${TORCH_DIR}/include/torch/csrc/api/include/)
include_directories(${TORCH_DIR}/include)
include_directories(/home/zy429782/my_study/myThreadPool)

# 创建可执行文件
add_executable(threadpool_torchaot threadpool_torchaot.cpp myThreadPool/ThreadPool.cpp)

# 手动添加库文件路径
set(TORCH_LIB_DIR ${TORCH_DIR}/lib)

# 链接额外的PyTorch库，使用完整路径
list(APPEND TORCH_LIBRARIES
    ${TORCH_LIB_DIR}/libc10.so
    ${TORCH_LIB_DIR}/libc10_cuda.so
    ${TORCH_LIB_DIR}/libtorch.so
    ${TORCH_LIB_DIR}/libtorch_cpu.so
    ${TORCH_LIB_DIR}/libtorch_cuda.so
    ${TORCH_LIB_DIR}/libtorch_cuda_linalg.so
    ${TORCH_LIB_DIR}/libtorch_global_deps.so
    # ${TORCH_LIB_DIR}/libtorch_python.so
)

# 链接库
target_link_libraries(threadpool_torchaot ${TORCH_LIBRARIES})

# CUDA 支持
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(threadpool_torchaot ${CUDA_LIBRARIES})
endif()

# 设置默认构建类型为 Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()