#include "nn/module.hpp"
#include <cuda/cmath>
#include <iostream>

template class tokenEmbedding<float>;
template class tokenEmbedding<double>;

template class positionEmbedding<float>;
template class positionEmbedding<double>;

template class singleHeadAttention<float>;
template class singleHeadAttention<double>;

template <typename t>
__global__ void tokenEmbeddingKernel(t* output, const TokenID* input, const t* weight, const size_t dim, const size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    size_t in = idx / dim;
    size_t yout = idx % dim;

    output[idx] = weight[input[in] * dim + yout];
}

template <typename t>
tensor<t> tokenEmbedding<t>::forward(const std::vector<TokenID>& input) {
    size_t len = input.size();
    TokenID* inputCpy = new TokenID[len];
    for (int i = 0; i < len; i++) inputCpy[i] = input[i];
    tensor<t> out(device::GPU, len, weight.getShape()[1]);
    TokenID* temp;
    cudaError_t err = cudaMalloc(&temp, len * sizeof(TokenID));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    err = cudaMemcpy(temp, inputCpy, len * sizeof(TokenID), cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    delete[] inputCpy;
    tokenEmbeddingKernel<<<cuda::ceil_div(out.numElements(), 256), 256>>>(out.data(), temp, weight.data(), weight.getShape()[1], out.numElements());
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    cudaFree(temp);
    out.requiresGrad(true);
    out.setGradientFunction(std::make_shared<tokenEmbeddingNode<t>>(&weight, input));
    return out;
}

template <typename t>
tensor<t> tokenEmbedding<t>::forward(const std::vector<std::vector<TokenID>>& input) {
    if (!input.size()) throw std::invalid_argument("Input is empty!");
    size_t batchSize = input.size();
    size_t len = input[0].size();
    tensor<t> out(device::GPU, batchSize, len, weight.getShape()[1]);
    TokenID* temp;
    cudaError_t err = cudaMalloc(&temp, len * batchSize * sizeof(TokenID));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    for (int i = 0; i < batchSize; i++) {
        err = cudaMemcpy(&temp[i * len], input[i].data(), len * sizeof(TokenID), cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
    }
    tokenEmbeddingKernel<<<cuda::ceil_div(out.numElements(), 256), 256>>>(out.data(), temp, weight.data(), weight.getShape()[1], out.numElements());
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    cudaFree(temp);
    out.requiresGrad(true);
    out.setGradientFunction(std::make_shared<tokenEmbeddingNode<t>>(&weight, input));
    return out;
}

template <typename t>
__global__ void positionEmbeddingKernel(t* output, const t* weight, const size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    output[idx] = weight[idx];
}

template <typename t>
tensor<t> positionEmbedding<t>::forward(size_t len, size_t batchSize) {
    if (len > weight.getShape()[0]) throw std::invalid_argument("Input sequence is longer than the maximum supported sequence length.");
    tensor<t> out(device::GPU, len, weight.getShape()[1]);
    positionEmbeddingKernel<<<cuda::ceil_div(out.numElements(), 256), 256>>>(out.data(), weight.data(), out.numElements());
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    out.requiresGrad(true);
    out.setGradientFunction(std::make_shared<positionEmbeddingNode<t>>(&weight, len));
    if (batchSize > 1) return out.batch(batchSize, 2);
    else out;
}

template <typename t>
__global__ void softmaxMaskKernel(t* scores, size_t rows, size_t cols) {
    size_t idxR = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idxC = threadIdx.y + blockDim.y * blockIdx.y;

    if (idxR >= rows || idxC >= cols) return;

    if (idxR < idxC) scores[blockIdx.z * cols * rows + idxR * cols + idxC] = -INFINITY;
}

template <typename t>
tensor<t> singleHeadAttention<t>::scaledDotProductAttention(const tensor<t>& Q, const tensor<t>& K, const tensor<t>& V, std::shared_ptr<tensor<t>>& score) const {
    auto scores = Q.matMul(K.transposed());
    if (Q.getShape().size() == 2) {
        scores = scores / std::sqrt(Q.getShape()[1]);
        softmaxMaskKernel<<<dim3(cuda::ceil_div(scores.getShape()[0], 16), cuda::ceil_div(scores.getShape()[1], 16)), dim3(16, 16)>>>(scores.data(), scores.getShape()[0], scores.getShape()[1]);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }    
        score = std::make_shared<tensor<t>>(scores.softmax());
    }
    if (Q.getShape().size() == 3) {
        scores = scores / std::sqrt(Q.getShape()[2]);
        softmaxMaskKernel<<<dim3(cuda::ceil_div(scores.getShape()[1], 16), cuda::ceil_div(scores.getShape()[2], 16), scores.getShape()[0]), dim3(16, 16, 1)>>>(scores.data(), scores.getShape()[1], scores.getShape()[2]);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }    
        score = std::make_shared<tensor<t>>(scores.softmax());
    }
    return score->matMul(V);
}
