#include "tensor/tensor.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda/cmath>
#include <curand_kernel.h>

template class tensor<float>;
template class tensor<double>;

template <typename t>
void tensor<t>::toGPU() {
    if (dev == device::GPU) {
        return; 
    }
    t* tempgpuData = nullptr;

    cudaError_t err = cudaMalloc(&tempgpuData, storageLength * sizeof(t));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    if (tens) {
        cudaMemcpy(tempgpuData, tens, storageLength * sizeof(t), cudaMemcpyHostToDevice);
        delete[] tens;
    }
    dev = device::GPU;
    tens = tempgpuData;
}

template <typename t>
void tensor<t>::toCPU() {
    if (dev == device::CPU) {
        return; 
    }
    t* tempcpuData = new t[storageLength];
    cudaMemcpy(tempcpuData, tens, storageLength * sizeof(t), cudaMemcpyDeviceToHost);
    cudaFree(tens);
    tens = tempcpuData;
    dev = device::CPU;
}

template <typename t>
tensor<t>::~tensor() {
    if (dev == device::GPU) {
        cudaFree(tens);
    }
    else if (dev == device::CPU) {
        delete[] tens;
    }
}

template <typename t>
tensor<t>::tensor(const tensor& other) : shape(other.shape), storageLength(other.storageLength), dev(other.dev) {
    if (dev == device::CPU) {
        tens = new t[storageLength];
        std::copy(other.tens, other.tens + storageLength, tens);
    }
    else if (dev == device::GPU) {
        cudaError_t err = cudaMalloc(&tens, storageLength * sizeof(t));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: "
                    << cudaGetErrorString(err)
                    << '\n';
        }
        cudaMemcpy(tens, other.tens, storageLength * sizeof(t), cudaMemcpyDefault);
    }
}

template <typename t>
tensor<t>::tensor(tensor&& other) noexcept : shape(std::move(other.shape)), storageLength(other.storageLength), tens(other.tens), dev(other.dev) {
    other.tens = nullptr;
    other.storageLength = 0;
    other.dev = device::CPU;
}

template <typename t>
tensor<t>& tensor<t>::operator=(const tensor& other) {
    if (this != &other) {
        if (dev == device::GPU) {
            cudaFree(tens);
        }
        else if (dev == device::CPU) {
            delete[] tens;
        }
        shape = other.shape;
        storageLength = other.storageLength;
        dev = other.dev;
        if (dev == device::CPU) {
            tens = new t[storageLength];
            std::copy(other.tens, other.tens + storageLength, tens);
        }
        else if (dev == device::GPU) {
            cudaError_t err = cudaMalloc(&tens, storageLength * sizeof(t));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: "
                        << cudaGetErrorString(err)
                        << '\n';
            }
            cudaMemcpy(tens, other.tens, storageLength * sizeof(t), cudaMemcpyDeviceToDevice);
        }
    }
    return *this;
}

template <typename t>
tensor<t>& tensor<t>::operator=(tensor&& other) noexcept {
    if (this != &other) {
        if (dev == device::GPU) {
            cudaFree(tens);
        }
        else if (dev == device::CPU) {
            delete[] tens;
        }
        shape = std::move(other.shape);
        storageLength = other.storageLength;
        tens = other.tens;
        dev = other.dev;
        other.tens = nullptr;
        other.storageLength = 0;
        other.dev = device::CPU;
    }
    return *this;
}

template <typename t>
__global__ void fillKernel(t val, size_t storageLength, t* tens) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < storageLength) {
        tens[idx] = val;
    }
}

template<typename t>
void tensor<t>::fill(t val) {
    if (dev == device::CPU) {
        toGPU();
    }
    fillKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(val, storageLength, tens);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
}

template <typename t>
__global__ void randomKernel(size_t storageLength, t* tens) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < storageLength) {
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        tens[idx] = curand_uniform(&state);
    }
}

template <typename t>
void tensor<t>::random() {
    if (dev == device::CPU) {
        toGPU();
    }
    randomKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, tens);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
}

template <typename t>
__global__ void addKernel(size_t storageLength, t* tensA, t* tensB) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < storageLength) {
        tensA[idx] += tensB[idx];
    }
}

template <typename t>
__global__ void subtractKernel(size_t storageLength, t* tensA, t* tensB) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < storageLength) {
        tensA[idx] -= tensB[idx];
    }
}

template <typename t>
tensor<t> tensor<t>::operator+(tensor& other) {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for addition.");
    }
    if (dev == device::CPU || other.dev == device::CPU) {
        toGPU();
        other.toGPU();
    }
    tensor<t> temp(*this);
    addKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, temp.data(), other.tens);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator+=(tensor& other) {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for addition.");
    }
    if (dev == device::CPU || other.dev == device::CPU) {
        toGPU();
        other.toGPU();
    }
    addKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, tens, other.tens);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    return *this;
}
template <typename t>
tensor<t> tensor<t>::operator-(tensor& other) {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for subtraction.");
    }
    if (dev == device::CPU || other.dev == device::CPU) {
        toGPU();
        other.toGPU();
    }
    tensor<t> temp(*this);
    subtractKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, temp.data(), other.tens);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator-=(tensor& other) {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for subtraction.");
    }
    if (dev == device::CPU || other.dev == device::CPU) {
        toGPU();
        other.toGPU();
    }
    subtractKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, tens, other.tens);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    return *this;
}


