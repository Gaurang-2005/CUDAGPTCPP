cmake_minimum_required(VERSION 3.20)

project(CUDAGPTCPP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(CUDA-GPT
    src/main.cpp
    src/tensor/tensor.cu
    src/autograd/node.cu
    src/nn/module.cu
)

target_include_directories(CUDA-GPT PRIVATE
    include
)