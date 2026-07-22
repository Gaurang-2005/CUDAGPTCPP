#include "nn/module.hpp"
#include <cuda/cmath>
#include <iostream>

template class tokenEmbedding<float>;
template class tokenEmbedding<double>;

template class positionEmbedding<float>;
template class positionEmbedding<double>;

template <typename t>
__global__ void tokenEmbeddingKernel(t* output, const size_t* input, const t* weight, const size_t dim, const size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    size_t in = idx / dim;
    size_t yout = idx % dim;

    output[idx] = weight[input[in] * dim + yout];
}

template <typename t>
tensor<t> tokenEmbedding<t>::forward(const size_t* input, size_t len) {
    tensor<t> out(device::GPU, len, weight.getShape()[1]);
    size_t* temp;
    cudaError_t err = cudaMalloc(&temp, len * sizeof(size_t));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    err = cudaMemcpy(temp, input, len * sizeof(size_t), cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    tokenEmbeddingKernel<<<cuda::ceil_div(out.numElements(), 256), 256>>>(out.data(), temp, weight.data(), weight.getShape()[1], out.numElements());
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    cudaFree(temp);
    out.requiresGrad(true);
    out.setGradientFunction(std::make_shared<tokenEmbeddingNode<t>>(&weight, input, len));
    return out;
}

template <typename t>
__global__ void positionEmbeddingKernel(t* output, const t* weight, const size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    output[idx] = weight[idx];
}

template <typename t>
tensor<t> positionEmbedding<t>::forward(size_t len) {
    if (len >= weight.getShape()[0]) throw std::invalid_argument("Input sequence is longer than the maximum supported sequence length.");
    tensor<t> out(device::GPU, len, weight.getShape()[1]);
    positionEmbeddingKernel<<<cuda::ceil_div(out.numElements(), 256), 256>>>(out.data(), weight.data(), out.numElements());
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    out.requiresGrad(true);
    out.setGradientFunction(std::make_shared<positionEmbeddingNode<t>>(&weight, len));
    return out;
}