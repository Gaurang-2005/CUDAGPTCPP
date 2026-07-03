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
    if (isGradEnabled) delete gradFunction;
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
    isGradEnabled = other.isGradEnabled;
    grad = other.grad;
    gradFunction = other.gradFunction;
}

template <typename t>
tensor<t>::tensor(tensor&& other) noexcept : shape(std::move(other.shape)), storageLength(other.storageLength), tens(other.tens), dev(other.dev) {
    other.tens = nullptr;
    other.storageLength = 0;
    isGradEnabled = other.isGradEnabled;
    grad = other.grad;
    gradFunction = other.gradFunction;
    other.isGradEnabled = false;
    other.grad = nullptr;
    other.gradFunction = nullptr;
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
        isGradEnabled = other.isGradEnabled;
        grad = other.grad;
        gradFunction = other.gradFunction;
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
        isGradEnabled = other.isGradEnabled;
        grad = other.grad;
        gradFunction = other.gradFunction;
        other.tens = nullptr;
        other.storageLength = 0;
        other.isGradEnabled = false;
        other.grad = nullptr;
        other.gradFunction = nullptr;
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
    if (isGradEnabled) {
        temp.gradFunction = new addNode<t>(this, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator+=(tensor& other) {
    if (isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
    *this = *this + other;
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
    if (isGradEnabled) {
        temp.gradFunction = new subtractNode<t>(this, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator-=(tensor& other) {
    if (isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
    *this = *this - other;
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
    if (isGradEnabled) {
        temp.gradFunction = new multiplyNode<t>(this, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator*=(tensor& other) {
    if (isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
    *this = *this * other;
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
    if (isGradEnabled) {
        temp.gradFunction = new divideNode<t>(this, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator/=(tensor& other) {
    if (isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
    *this = *this / other;
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
    if (isGradEnabled) {
        temp.gradFunction = new transposeNode<t>(this);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::transpose() {
    if (isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
    *this = transposed();
    return *this;
}

template <typename t>
__global__ void matMulKernel(t* output, t* A, t* B, size_t com, size_t outY, size_t outX, size_t storageLength) {
    constexpr int tileSize = 16;
    __shared__ t At[tileSize][tileSize];
    __shared__ t Bt[tileSize][tileSize];
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("kernel running! %d %d", row, col);
    t sum = 0;
    for (int i = 0; i < cuda::ceil_div(com, tileSize); i++) {
        size_t common = i * tileSize;
        size_t Ay = row;
        if (common + threadIdx.x < com && Ay < outY) {
            At[threadIdx.y][threadIdx.x] = A[common + threadIdx.x + Ay * com];
        }
        else At[threadIdx.y][threadIdx.x] = 0;
        size_t Bx = col;
        if (common + threadIdx.y < com && Bx < outX) {
            Bt[threadIdx.y][threadIdx.x] = B[(common + threadIdx.y) * outX + Bx];
        }
        else Bt[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();
        for (int j = 0; j < tileSize; j++) {
            sum += At[threadIdx.y][j] * Bt[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < outY && col < outX)
    output[col + row * outX] = sum;
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
    constexpr int tileSize = 16;
    dim3 blockSize = dim3(tileSize, tileSize, 1);
    dim3 gridSize = dim3(cuda::ceil_div(out.shape[1], tileSize), cuda::ceil_div(out.shape[0], tileSize), 1);
    // std::cout << gridSize.x * gridSize.y << std::endl << blockSize.x * blockSize.y << std::endl;
    matMulKernel<<<gridSize, blockSize>>>(out.tens, tens, other.tens, shape[1], out.shape[0], out.shape[1], out.storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    if (isGradEnabled) {
        out.gradFunction = new matMulNode<t>(this, &other);
        out.isGradEnabled = true;
    }
    return out;
}

template<typename t>
__global__ void sumKernel(t* out, t* tens, size_t storageLength) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < storageLength) {

        __shared__ t tempStore[256];

        tempStore[threadIdx.x] = tens[idx];
        __syncthreads();
        for (int i = 1; i < 256; i*=2) {
            if (!(threadIdx.x % (2 * i) == i || threadIdx.x + i >= 256)) 
            tempStore[threadIdx.x] += tempStore[threadIdx.x + i];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[blockIdx.x] = tempStore[0];
    }
}

template<typename t>
tensor<t> tensor<t>::sum() {
    if (dev == device::CPU) toGPU();

    tensor<t> out(device::CPU, 1, 1);
    size_t blocks = cuda::ceil_div(storageLength, 256);
    t* tempOut;
    cudaMallocManaged(&tempOut, sizeof(t)*blocks);
    sumKernel <<<blocks, 256>>> (tempOut, tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }    
    out.tens[0] = 0;
    for (int i = 0; i < blocks; i++) {
        out.tens[0] += tempOut[i]; 
    }
    if (isGradEnabled) {
        out.gradFunction = new sumNode<t>(this);
        out.isGradEnabled = true;
    }
    return out;
}

