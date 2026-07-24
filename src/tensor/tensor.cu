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
void tensor<t>::toGPU() const {
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
void tensor<t>::toCPU() const {
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
    if (refCount) (*refCount)--;
    if (grad && !refCount || grad && *refCount == 0) delete grad;
    if (refCount && *refCount == 0) delete refCount;
}

template <typename t>
tensor<t>::tensor(device dev, std::initializer_list<std::initializer_list<t>> list) : dev(dev) {
    shape.push_back(list.size());
    shape.push_back(list.begin()->size());
    for (auto& i : shape) {
        storageLength*=i;
    }
    tens = new t[storageLength]{};

    size_t idx = 0;
    for (const auto& row : list) {
        if (row.size() != shape[1])
            throw std::runtime_error("Initializer list rows have different lengths");
        for (const auto& val : row) {
            tens[idx++] = val;
        }
    }

    if (dev == device::GPU) {
        t* temp;
        cudaError_t err = cudaMalloc(&temp, storageLength * sizeof(t));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: "
                    << cudaGetErrorString(err)
                    << '\n';
        }
        err = cudaMemcpy(temp, tens, storageLength * sizeof(t), cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: "
                    << cudaGetErrorString(err)
                    << '\n';
        }
        delete[] tens;
        tens = temp;
    }
    refCount = new int(1);
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
        err = cudaMemcpy(tens, other.tens, storageLength * sizeof(t), cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: "
                    << cudaGetErrorString(err)
                    << '\n';
        }
    }
    isGradEnabled = other.isGradEnabled;
    grad = nullptr;
    if (other.grad) grad = other.grad;
    gradFunction = other.gradFunction;
    refCount = other.refCount;
    (*refCount)++;
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
    refCount = other.refCount;
    other.refCount = nullptr;
}

template <typename t>
tensor<t>& tensor<t>::operator=(const tensor& other) {
    if (this != &other) {
        if (dev == device::GPU) {
            cudaFree(tens);
            tens = nullptr;
        }
        else if (dev == device::CPU) {
            delete[] tens;
            tens = nullptr;
        }
        shape = other.shape;
        storageLength = other.storageLength;
        dev = other.dev;
        isGradEnabled = other.isGradEnabled;
        (*refCount)--;
        if (grad && !(*refCount)) delete grad;
        grad = nullptr;
        if (other.grad) grad = other.grad;
        if (!(*refCount)) delete refCount;
        refCount = nullptr;
        refCount = other.refCount;
        (*refCount)++;
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
    if (gradFunction || other.gradFunction) throw std::logic_error("Move Assignment of tensors participating in an autograd graph is not supported.");
    if (this != &other) {
        if (dev == device::GPU) {
            cudaFree(tens);
            tens = nullptr;
        }
        else if (dev == device::CPU) {
            delete[] tens;
            tens = nullptr;
        }
        (*refCount)--;
        if (grad && !(*refCount)) delete grad;
        grad = nullptr;
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
        if (refCount && !(*refCount)) delete refCount;
        refCount = nullptr;
        refCount = other.refCount;
        other.refCount = nullptr;
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
        tens[idx] = 2 * curand_uniform(&state) - 1;
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
tensor<t> tensor<t>::operator+(const tensor& other) const & {
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
    if (isGradEnabled || other.isGradEnabled) {
        temp.gradFunction = std::make_shared<addNode<t>>(this, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator+(const tensor<t>& other) && {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for addition.");
    }
    toGPU();
    other.toGPU();
    tensor<t> temp(*this);
    addKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, temp.data(), other.tens);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        temp.gradFunction = std::make_shared<addNode<t>>(first, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator+(tensor<t>&& other) const & {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for addition.");
    }
    toGPU();
    other.toGPU();
    tensor<t> temp(*this);
    addKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, temp.data(), other.tens);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        temp.gradFunction = std::make_shared<addNode<t>>(this, second);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator+(tensor<t>&& other) && {
    if (shape != other.shape){
        throw std::invalid_argument("Tensors must have the same shape for addition.");
    }
    toGPU();
    other.toGPU();
    tensor<t> temp(*this);
    addKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(storageLength, temp.data(), other.tens);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        temp.gradFunction = std::make_shared<addNode<t>>(first, second);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator+=(const tensor& other) {
    if (isGradEnabled || other.isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
    *this = *this + other;
    return *this;
}
template <typename t>
tensor<t> tensor<t>::operator-(const tensor& other) const & {
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
    if (isGradEnabled || other.isGradEnabled) {
        temp.gradFunction = std::make_shared<subtractNode<t>>(this, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator-(const tensor& other) && {
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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        temp.gradFunction = std::make_shared<subtractNode<t>>(first, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator-(tensor&& other) const & {
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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        temp.gradFunction = std::make_shared<subtractNode<t>>(this, second);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator-(tensor&& other) && {
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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        temp.gradFunction = std::make_shared<subtractNode<t>>(first, second);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator-=(const tensor& other) {
    if (isGradEnabled || other.isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
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
tensor<t> tensor<t>::operator*(const tensor& other) const & {
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
    if (isGradEnabled || other.isGradEnabled) {
        temp.gradFunction = std::make_shared<multiplyNode<t>>(this, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator*(const tensor& other) && {
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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        temp.gradFunction = std::make_shared<multiplyNode<t>>(first, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator*(tensor&& other) const & {
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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        temp.gradFunction = std::make_shared<multiplyNode<t>>(this, second);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator*(tensor&& other) && {
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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        temp.gradFunction = std::make_shared<multiplyNode<t>>(first, second);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator*=(const tensor& other) {
    if (isGradEnabled || other.isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
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
tensor<t> tensor<t>::operator/(const tensor& other) const & {
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
    if (isGradEnabled || other.isGradEnabled) {
        temp.gradFunction = std::make_shared<divideNode<t>>(this, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator/(const tensor& other) && {
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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        temp.gradFunction = std::make_shared<divideNode<t>>(first, &other);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator/(tensor&& other) const & {
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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        temp.gradFunction = std::make_shared<divideNode<t>>(this, second);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::operator/(tensor&& other) && {
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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        temp.gradFunction = std::make_shared<divideNode<t>>(first, second);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t>& tensor<t>::operator/=(const tensor& other) {
    if (isGradEnabled || other.isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
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
    std::cout << "), device: " << (dev == device::CPU ? "CPU" : "GPU") << std::endl;
    for (size_t i = 0; i < storageLength; ++i) {
        if (!(i % shape[1])) std::cout << std::endl;
        std::cout << tempData[i] << " ";
    }
    std::cout << std::endl << std::endl;
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
tensor<t> tensor<t>::transposed() const & {
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
        temp.gradFunction = std::make_shared<transposeNode<t>>(this);
        temp.isGradEnabled = true;
    }
    return temp;
}

template <typename t>
tensor<t> tensor<t>::transposed() && {
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
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        temp.gradFunction = std::make_shared<transposeNode<t>>(first);
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
tensor<t> tensor<t>::matMul(const tensor<t>& other) const & {
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Matrix multiplication requires A.cols == B.rows.");
    }
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("Matrix multiplication currently only supports rank-2 tensors");
    }
    if (isIdentity) return other;
    if (other.isIdentity) return *this;

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
    if (isGradEnabled || other.isGradEnabled) {
        out.gradFunction = std::make_shared<matMulNode<t>>(this, &other);
        out.isGradEnabled = true;
    }
    return out;
}

template <typename t>
tensor<t> tensor<t>::matMul(const tensor<t>& other) && {
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Matrix multiplication requires A.cols == B.rows.");
    }
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("Matrix multiplication currently only supports rank-2 tensors");
    }
    if (isIdentity) return other;
    if (other.isIdentity) return *this;

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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.gradFunction = std::make_shared<matMulNode<t>>(first, &other);
        out.isGradEnabled = true;
    }
    return out;
}

template <typename t>
tensor<t> tensor<t>::matMul(tensor<t>&& other) const & {
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Matrix multiplication requires A.cols == B.rows.");
    }
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("Matrix multiplication currently only supports rank-2 tensors");
    }
    if (isIdentity) return other;
    if (other.isIdentity) return *this;

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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        out.gradFunction = std::make_shared<matMulNode<t>>(this, second);
        out.isGradEnabled = true;
    }
    return out;
}

template <typename t>
tensor<t> tensor<t>::matMul(tensor<t>&& other) && {
    if (shape[1] != other.shape[0]) {
        throw std::invalid_argument("Matrix multiplication requires A.cols == B.rows.");
    }
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("Matrix multiplication currently only supports rank-2 tensors");
    }
    if (isIdentity) return other;
    if (other.isIdentity) return *this;

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
    if (isGradEnabled || other.isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        std::shared_ptr<tensor<t>> second = std::make_shared<tensor<t>>(std::move(other));
        out.gradFunction = std::make_shared<matMulNode<t>>(first, second);
        out.isGradEnabled = true;
    }
    return out;
}

template<typename t>
__global__ void sumKernel(t* out, t* tens, size_t storageLength) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ t tempStore[256];
    
    if (idx < storageLength) tempStore[threadIdx.x] = tens[idx];
    else tempStore[threadIdx.x] = 0;
    
    __syncthreads();
    for (int i = 1; i < 256; i*=2) {
        if (!(threadIdx.x % (2 * i) == i || threadIdx.x + i >= 256)) 
        tempStore[threadIdx.x] += tempStore[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = tempStore[0];
    
}

template<typename t>
tensor<t> tensor<t>::sum() const & {
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
    cudaFree(tempOut);
    if (isGradEnabled) {
        out.gradFunction = std::make_shared<sumNode<t>>(this);
        out.isGradEnabled = true;
    }
    return out;
}

template<typename t>
tensor<t> tensor<t>::sum() && {
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
    cudaFree(tempOut);
    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.gradFunction = std::make_shared<sumNode<t>>(first);
        out.isGradEnabled = true;
    }
    return out;
}

template <typename t>
__global__ void identityKernel(t* tens, size_t x, size_t y) {
    size_t idX = threadIdx.x + blockIdx.x * blockDim.x;
    size_t idY = threadIdx.y + blockIdx.y * blockDim.y;

    if (idX >= x || idY >= y) return;

    if (idX == idY) tens[idX + idY * x] = 1;
    else tens[idX + idY * x] = 0;
}

template <typename t>
void tensor<t>::identity() {
    toGPU();
    dim3 gridSize = dim3(cuda::ceil_div(shape[0], 16), cuda::ceil_div(shape[1], 16));
    dim3 blockSize = dim3(16, 16);
    isIdentity = true;
    identityKernel <<<gridSize, blockSize>>> (tens, shape[0], shape[1]);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
}

template <typename t>
__global__ void negateKernel(t* tens, size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    tens[idx] *= -1;
}
template <typename t>
tensor<t> tensor<t>::operator-() const {
    tensor<t> temp(*this);

    temp.toGPU();

    negateKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(temp.tens, storageLength);
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
__global__ void expKernel(t* out, t* in, size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = exp(in[idx]);
}

template <typename t>
tensor<t> tensor<t>::exp() const & {
    tensor<t> out(device::GPU, shape[0], shape[1]);
    expKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(out.tens, tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<expNode<t>>(this);
    }
    
    return out;
}

template <typename t>
tensor<t> tensor<t>::exp() && {
    tensor<t> out(device::GPU, shape[0], shape[1]);
    expKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(out.tens, tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<expNode<t>>(first);
    }
    
    return out;
}

template <typename t>
__global__ void powKernel(t* out, t* in, size_t storageLength, t power) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = pow(in[idx], power);
}

template <typename t>
tensor<t> tensor<t>::pow(t power) const & {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);
    powKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(out.tens, tens, storageLength, power);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<powNode<t>>(this, power);
    }
    
    return out;
}

template <typename t>
tensor<t> tensor<t>::pow(t power) && {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);
    powKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(out.tens, tens, storageLength, power);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<powNode<t>>(first, power);
    }
    
    return out;
}

template <typename t>
__global__ void logKernel(t* out, t* in, size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = log(in[idx]);
}

template <typename t>
tensor<t> tensor<t>::log() const & {
    tensor<t> out(device::GPU, shape[0], shape[1]);
    logKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(out.tens, tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<logNode<t>>(this);
    }
    
    return out;
}

template <typename t>
tensor<t> tensor<t>::log() && {
    tensor<t> out(device::GPU, shape[0], shape[1]);
    logKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(out.tens, tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<logNode<t>>(first);
    }
    
    return out;
}

template <typename t>
__global__ void digitMultiplyKernel(t* out, t* in, size_t storageLength, t val) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = in[idx] * val;
}

template <typename t>
tensor<t> tensor<t>::operator*(t val) const {
    tensor<t> out(device::GPU, shape[0], shape[1]);
    digitMultiplyKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(out.tens, tens, storageLength, val);   
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    return out;
}

template <typename t>
__global__ void digitDivideKernel(t* out, t* in, size_t storageLength, t val) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = in[idx] / val;
}

template <typename t>
tensor<t> tensor<t>::operator/(t val) const {
    tensor<t> out(device::GPU, shape[0], shape[1]);
    digitDivideKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(out.tens, tens, storageLength, val);   
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    return out;
}

template <typename t>
__global__ void ReLUKernel(t*tens, t* out, size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= storageLength) return;

    if (tens[idx] >= 0) out[idx] = tens[idx];
    else out[idx] = 0;
}

template <typename t>
tensor<t> tensor<t>::ReLU() const & {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);

    ReLUKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(tens, out.tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<reluNode<t>>(this);
    }

    return out;
}

template <typename t>
tensor<t> tensor<t>::ReLU() && {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);

    ReLUKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(tens, out.tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<reluNode<t>>(first);
    }

    return out;
}

template <typename t>
__global__ void sigmoidKernel(t*tens, t* out, size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = 1 / (1 + exp(-tens[idx]));
}

template <typename t>
tensor<t> tensor<t>::sigmoid() const & {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);

    sigmoidKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(tens, out.tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<sigmoidNode<t>>(this);
    }

    return out;
}

template <typename t>
tensor<t> tensor<t>::sigmoid() && {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);

    sigmoidKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(tens, out.tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<sigmoidNode<t>>(first);
    }

    return out;
}

template <typename t>
__global__ void tanhKernel(t*tens, t* out, size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = (exp(tens[idx]) - exp(-tens[idx])) / (exp(tens[idx]) + exp(-tens[idx]));
}

template <typename t>
tensor<t> tensor<t>::tanh() const & {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);

    tanhKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(tens, out.tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<tanhNode<t>>(this);
    }

    return out;
}

template <typename t>
tensor<t> tensor<t>::tanh() && {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);

    tanhKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(tens, out.tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<tanhNode<t>>(first);
    }

    return out;
}

template <typename t>
__global__ void geluKernel(t* tens, t* out, size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;
    constexpr t root2OnRootPi = t(0.79788456080286535587989211986876L);
    constexpr t geluConst = t(0.044715);
    t temp = geluConst * tens[idx] * tens[idx] * tens[idx] + tens[idx];
    temp *= root2OnRootPi;
    out[idx] = 0.5 * tens[idx] * (1 + (tanh(temp)));
}

template <typename t>
tensor<t> tensor<t>::gelu() const & {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);

    geluKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(tens, out.tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<geluNode<t>>(this);
    }

    return out;
}

template <typename t>
tensor<t> tensor<t>::gelu() && {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);

    geluKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(tens, out.tens, storageLength);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<geluNode<t>>(first);
    }

    return out;
}

template <typename t>
__global__ void rowSumKernel(t* tens, t* out, size_t rows, size_t cols) {
    size_t row = blockIdx.x;
    size_t pos = row * cols;
    __shared__ t temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    for (int i = 0; i < cols; i++) {
        if (threadIdx.x + blockDim.x * i >= cols) break;
        temp[threadIdx.x] += tens[pos + threadIdx.x + blockDim.x * i];
    }
    __syncthreads();
    for (int i = 1; i < 256; i*=2) {
        if (!(threadIdx.x % (2 * i) == i || threadIdx.x + i >= 256)) 
        temp[threadIdx.x] += temp[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = temp[0];
}

template <typename t>
tensor<t> tensor<t>::rowSum() const {
    toGPU();

    tensor<t> out(device::GPU, shape[0], 1);
    rowSumKernel<<<shape[0], 256>>>(tens, out.tens, shape[0], shape[1]);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    return out;
}
template <typename t>
__global__ void colSumKernel(t* tens, t* out, size_t rows, size_t cols) {
    size_t col = blockIdx.x;
    size_t pos = col;
    __shared__ t temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    for (int i = 0; i < rows; i++) {
        if (threadIdx.x + blockDim.x * i >= rows) break;
        temp[threadIdx.x] += tens[pos + (threadIdx.x + blockDim.x * i)*cols];
    }
    __syncthreads();
    for (int i = 1; i < 256; i*=2) {
        if (!(threadIdx.x % (2 * i) == i || threadIdx.x + i >= 256)) 
        temp[threadIdx.x] += temp[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[col] = temp[0];
}
template <typename t>
tensor<t> tensor<t>::colSum() const {
    toGPU();

    tensor<t> out(device::GPU, 1, shape[1]);
    colSumKernel<<<shape[1], 256>>>(tens, out.tens, shape[0], shape[1]);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    return out;
}

template <typename t>
__global__ void softmaxKernel(t* tens, t* sum, t* out, size_t row, size_t col) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= row * col) return;
    out[idx] = tens[idx] / sum[idx / col];
}

template <typename t>
__global__ void rowMaxKernel(t* tens, t* out, size_t rows, size_t cols) {
    size_t row = blockIdx.x;
    size_t pos = row * cols;
    __shared__ t temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    for (int i = 0; i < cols; i++) {
        if (threadIdx.x + blockDim.x * i >= cols) break;
        if (temp[threadIdx.x] < tens[pos + threadIdx.x + blockDim.x * i]) temp[threadIdx.x] = tens[pos + threadIdx.x + blockDim.x * i];
    }
    __syncthreads();
    for (int i = 1; i < 256; i*=2) {
        if (!(threadIdx.x % (2 * i) == i || threadIdx.x + i >= 256)) 
        if (temp[threadIdx.x] < temp[threadIdx.x + i]) temp[threadIdx.x] = temp[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = temp[0];
}

template <typename t>
tensor<t> tensor<t>::rowMax() const {
    toGPU();

    tensor<t> out(device::GPU, shape[0], 1);
    rowMaxKernel<<<shape[0], 256>>>(tens, out.tens, shape[0], shape[1]);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }

    return out;
}

template <typename t>
__global__ void broadcastSubtractKernel(t* A, t* B, t* out, size_t row, size_t col) {
    size_t idxX = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idxY = threadIdx.y + blockDim.y * blockIdx.y;

    if (idxX >= col ||  idxY >= row) return;

    out[idxY*col + idxX] = A[idxY*col + idxX] - B[idxY];
}

template <typename t>
tensor<t> tensor<t>::softmax() const & {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);
    
    if (isGradEnabled) {
        requiresGrad(false);
        tensor<t> temp (device::GPU, shape[0], shape[1]);
        dim3 blocks = dim3(cuda::ceil_div(shape[1], 16), cuda::ceil_div(shape[0], 16));
        dim3 threads = dim3(16, 16);
        broadcastSubtractKernel<<<blocks, threads>>>(tens, rowMax().tens, temp.tens, temp.shape[0], temp.shape[1]);
        tensor<t> numerator = temp.exp();
        tensor<t> summed = numerator.rowSum();
        softmaxKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(numerator.tens, summed.tens, out.tens, shape[0], shape[1]);
        requiresGrad(true);
    }
    else {
        tensor<t> temp (device::GPU, shape[0], shape[1]);
        dim3 blocks = dim3(cuda::ceil_div(shape[1], 16), cuda::ceil_div(shape[0], 16));
        dim3 threads = dim3(16, 16);
        broadcastSubtractKernel<<<blocks, threads>>>(tens, rowMax().tens, temp.tens, temp.shape[0], temp.shape[1]);
        tensor<t> numerator = temp.exp();
        tensor<t> summed = numerator.rowSum();
        softmaxKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(numerator.tens, summed.tens, out.tens, shape[0], shape[1]);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    if (isGradEnabled) {
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<softmaxNode<t>>(this);
    }
    return out;
}

template <typename t>
tensor<t> tensor<t>::softmax() && {
    toGPU();
    tensor<t> out(device::GPU, shape[0], shape[1]);
    
    if (isGradEnabled) {
        requiresGrad(false);
        tensor<t> temp (device::GPU, shape[0], shape[1]);
        dim3 blocks = dim3(cuda::ceil_div(shape[1], 16), cuda::ceil_div(shape[0], 16));
        dim3 threads = dim3(16, 16);
        broadcastSubtractKernel<<<blocks, threads>>>(tens, rowMax().tens, temp.tens, temp.shape[0], temp.shape[1]);
        tensor<t> numerator = temp.exp();
        tensor<t> summed = numerator.rowSum();
        softmaxKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(numerator.tens, summed.tens, out.tens, shape[0], shape[1]);
        requiresGrad(true);
    }
    else {
        tensor<t> temp (device::GPU, shape[0], shape[1]);
        dim3 blocks = dim3(cuda::ceil_div(shape[1], 16), cuda::ceil_div(shape[0], 16));
        dim3 threads = dim3(16, 16);
        broadcastSubtractKernel<<<blocks, threads>>>(tens, rowMax().tens, temp.tens, temp.shape[0], temp.shape[1]);
        tensor<t> numerator = temp.exp();
        tensor<t> summed = numerator.rowSum();
        softmaxKernel<<<cuda::ceil_div(storageLength, 256), 256>>>(numerator.tens, summed.tens, out.tens, shape[0], shape[1]);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<softmaxNode<t>>(first);
    }
    return out;
}

template <typename t>
__global__ void batch1Kernel(t* tens, t* out, size_t storageLength, size_t dataLen) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = tens[idx % dataLen];
} 

template <typename t>
__global__ void batch2Kernel(t* tens, t* out, size_t storageLength, size_t dataLen) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = tens[idx / dataLen];
} 

template <typename t>
tensor<t> tensor<t>::batch(size_t batchSize, int axis) const & {
    tensor<t> out;
    if (axis == 0) {
        out = tensor<t>(dev, batchSize, shape[1]);
        batch1Kernel<<<cuda::ceil_div(out.storageLength, 256), 256>>>(tens, out.tens, out.storageLength, shape[1]);
    }
    if (axis == 1) {
        out = tensor<t>(dev, shape[0], batchSize);
        batch2Kernel<<<cuda::ceil_div(out.storageLength, 256), 256>>>(tens, out.tens, out.storageLength, batchSize);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    if (isGradEnabled) {
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<batchNode<t>>(this, axis);
    }
    return out;
}

template <typename t>
tensor<t> tensor<t>::batch(size_t batchSize, int axis) && {
    tensor<t> out;
    if (axis == 0) {
        out = tensor<t>(dev, batchSize, shape[1]);
        batch1Kernel<<<cuda::ceil_div(out.storageLength, 256), 256>>>(tens, out.tens, out.storageLength, shape[1]);
    }
    if (axis == 1) {
        out = tensor<t>(dev, shape[0], batchSize);
        batch2Kernel<<<cuda::ceil_div(out.storageLength, 256), 256>>>(tens, out.tens, out.storageLength, batchSize);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    if (isGradEnabled) {
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
        out.isGradEnabled = true;
        out.gradFunction = std::make_shared<batchNode<t>>(first, axis);
    }
    return out;
}