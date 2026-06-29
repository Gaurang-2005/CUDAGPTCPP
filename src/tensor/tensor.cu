#include "tensor/tensor.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda/cmath>
#include <curand_kernel.h>

template class tensor<float>;
template class tensor<double>;

template <typename t>
void tensor<t>::constructorAllocate() {
    if (tens) return;

    cudaError_t err = cudaMalloc(&tens, storageLength * sizeof(t));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
}

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

template <typename t>
__global__ void multiplyKernel(size_t storageLength, t* tensA, t* tensB) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < storageLength) {
        tensA[idx] *= tensB[idx];
    }
}

template <typename t>
tensor<t> tensor<t>::operator*(tensor& other) {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for multiplication.");
    }
    if (dev == device::CPU || other.dev == device::CPU) {
        toGPU();
        other.toGPU();
    }
    tensor<t> temp(*this);
    multiplyKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, temp.data(), other.tens);
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
tensor<t>& tensor<t>::operator*=(tensor& other) {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for multiplication.");
    }
    if (dev == device::CPU || other.dev == device::CPU) {
        toGPU();
        other.toGPU();
    }
    multiplyKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, tens, other.tens);
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
__global__ void divideKernel(size_t storageLength, t* tensA, t* tensB) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < storageLength) {
        tensA[idx] /= tensB[idx];
    }
}

template <typename t>
tensor<t> tensor<t>::operator/(tensor& other) {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for division.");
    }
    if (dev == device::CPU || other.dev == device::CPU) {
        toGPU();
        other.toGPU();
    }
    tensor<t> temp(*this);
    divideKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, temp.data(), other.tens);
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
tensor<t>& tensor<t>::operator/=(tensor& other) {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for division.");
    }
    if (dev == device::CPU || other.dev == device::CPU) {
        toGPU();
        other.toGPU();
    }
    divideKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, tens, other.tens);
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
void tensor<t>::print() const {
    if (storageLength == 0) {
        std::cout << "Empty Tensor\n";
        return;
    }       
    t* tempData = nullptr;
    if (dev == device::GPU) {
        tempData = new t[storageLength];
        cudaError_t err = cudaMemcpy(tempData, tens, storageLength * sizeof(t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Memory copy failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            return;
        }
    }
    else {
        tempData = tens;
    }
    std::cout<<"Tensor shape: (";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i != shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "), device: " << (dev == device::CPU ? "CPU" : "GPU") << std::endl << std::endl;
    for (size_t i = 0; i < storageLength; ++i) {
        std::cout << tempData[i] << " ";
    }
    std::cout << std::endl;
    if (dev == device::GPU) {
        delete[] tempData;
    }
}

template <typename t>
__global__ void transposeKernel(t* temp, t* tens, size_t storageLength, size_t x, size_t y) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= storageLength) return;
    size_t xOld = idx / y;
    size_t yOld = idx % y;

    temp[yOld * x + xOld] = tens[idx];
}

template <typename t>
tensor<t> tensor<t>::transposed() {
    if (shape.size() != 2) {
        throw std::invalid_argument("transposed() currently supports only rank-2 tensors.");
    }
    if (dev == device::CPU) {
        toGPU();
    }
    tensor<t> temp(device::GPU, shape[1], shape[0]);
    transposeKernel<<<cuda::ceil_div(storageLength, 256), 256>>> (temp.tens, tens, storageLength, shape[0], shape[1]);
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
tensor<t>& tensor<t>::transpose() {
    *this = transposed();
    return *this;
}

template <typename t>
__global__ void matMulKernel(t* output, t* A, t* B, size_t com, size_t outY, size_t storageLength) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= storageLength) return;
    size_t Ax = idx / outY;
    size_t By = idx % outY;
    output[idx] = 0;
    for (size_t i = 0; i < com; i++) {
        output[idx] += A[Ax * com + i] * B[i * outY + By];
    }

} 

template <typename t>
tensor<t> tensor<t>::matMul(tensor<t>& other) {
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Matrix multiplication requires A.cols == B.rows.");
    }
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("Matrix multiplication currently only supports rank-2 tensors");
    }

    toGPU();
    other.toGPU();

    tensor<t> out(device::GPU, shape[0], other.shape[1]);

    matMulKernel<<<cuda::ceil_div(shape[0]*other.shape[1], 256), 256>>>(out.tens, tens, other.tens, shape[1], out.shape[1], out.storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    return out;
}