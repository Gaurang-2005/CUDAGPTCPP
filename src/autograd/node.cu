#include "autograd/node.hpp"
#include "tensor/tensor.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda/cmath>

template class addNode<float>;
template class addNode<double>;

template class subtractNode<float>;
template class subtractNode<double>;

template class multiplyNode<float>;
template class multiplyNode<double>;

template class divideNode<float>;
template class divideNode<double>;

template class matMulNode<float>;
template class matMulNode<double>;

template class transposeNode<float>;
template class transposeNode<double>;

template class sumNode<float>;
template class sumNode<double>;

template class meanNode<float>;
template class meanNode<double>;

template class reshapeNode<float>;
template class reshapeNode<double>;

template class expNode<float>;
template class expNode<double>;

template class logNode<float>;
template class logNode<double>;

template class powNode<float>;
template class powNode<double>;

template class reluNode<float>;
template class reluNode<double>;

template class sigmoidNode<float>;
template class sigmoidNode<double>;

template class tanhNode<float>;
template class tanhNode<double>;

template class geluNode<float>;
template class geluNode<double>;

template class softmaxNode<float>;
template class softmaxNode<double>;

template class crossEntropyLossNode<float>;
template class crossEntropyLossNode<double>;

template class batchNode<float>;
template class batchNode<double>;

template class layerNormNode<float>;
template class layerNormNode<double>;

template class tokenEmbeddingNode<float>;
template class tokenEmbeddingNode<double>;

template class positionEmbeddingNode<float>;
template class positionEmbeddingNode<double>;

template class singleHeadAttentionNode<float>;
template class singleHeadAttentionNode<double>;

template class gatherNode<float>;
template class gatherNode<double>;

template class rowSumNode<float>;
template class rowSumNode<double>;

template class colSumNode<float>;
template class colSumNode<double>;

template <typename t>
void addNode<t>::backward(const tensor<t>& owner) {
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient();
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
    if (B -> gradient()) *B -> gradient() += *owner.gradient();
    else B -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    A -> clearGradientFunction();
    B -> clearGradientFunction();
}

template <typename t>
void subtractNode<t>::backward(const tensor<t>& owner) {
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient();
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
    if (B -> gradient()) *B -> gradient() -= *owner.gradient();
    else B -> setGradient(std::make_shared<tensor<t>>(-*owner.gradient()));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    A -> clearGradientFunction();
    B -> clearGradientFunction();
}

template <typename t>
void multiplyNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
    if (B -> gradient()) *B -> gradient() += *owner.gradient() * (*A.get());
    else B -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*A.get())));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    A -> clearGradientFunction();
    B -> clearGradientFunction();
}

template <typename t>
void divideNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B.get());
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() / (*B.get())));
    if (B -> gradient()) *B -> gradient() -= (*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()));
    else B -> setGradient(std::make_shared<tensor<t>>(-(*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    A -> clearGradientFunction();
    B -> clearGradientFunction();
}

template <typename t>
void matMulNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    // std::cout<<A->getShape()[0]<<' '<<A->getShape()[1]<<'\n';
    if (A -> gradient()) *A -> gradient() += (*owner.gradient()).matMul(B -> transposed());
    else A -> setGradient(std::make_shared<tensor<t>>((*owner.gradient()).matMul(B -> transposed())));
    if (B -> gradient()) *B -> gradient() += (A -> transposed()).matMul(*owner.gradient());
    else B -> setGradient(std::make_shared<tensor<t>>((A -> transposed()).matMul(*owner.gradient())));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    A -> clearGradientFunction();
    B -> clearGradientFunction();
}

template <typename t>
void transposeNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += (*owner.gradient()).transposed();
    else A -> setGradient(std::make_shared<tensor<t>>((*owner.gradient()).transposed()));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void sumNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape()[0], A->getShape()[1]);
    owner.gradient()->toCPU();
    temp.fill(owner.gradient()->data()[0]);
    if (A -> gradient()) *A -> gradient() += temp;
    else A -> setGradient(std::make_shared<tensor<t>>(temp));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void meanNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape()[0], A->getShape()[1]);
    owner.gradient()->toCPU();
    temp.fill(owner.gradient()->data()[0]/temp.numElements());
    if (A -> gradient()) *A -> gradient() += temp;
    else A -> setGradient(std::make_shared<tensor<t>>(temp));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}
    
template <typename t>
void reshapeNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += owner.gradient()->reshaped(oldShape[0], oldShape[1]);
    else A -> setGradient(std::make_shared<tensor<t>>(owner.gradient()->reshaped(oldShape[0], oldShape[1])));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void expNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * A -> exp();
    else A -> setGradient(std::make_shared<tensor<t>>(*(owner.gradient()) * A -> exp()));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void logNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(*A.get());
    temp.ones();
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * temp / *A.get();
    else A -> setGradient(std::make_shared<tensor<t>>(*(owner.gradient()) * temp / *A.get()));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void powNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * A -> pow(power - 1) * power;
    else A -> setGradient(std::make_shared<tensor<t>>(*(owner.gradient()) * A -> pow(power - 1) * power));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
__global__ void reluGradKernel(const t* tens, t* out, size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    if (tens[idx] > 0) out[idx] = t(1);
    else out[idx] = t(0);
}

template <typename t>
void reluNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape()[0], A->getShape()[1]);
    reluGradKernel<<<cuda::ceil_div(temp.numElements(), 256), 256>>>(A->data(),temp.data(), temp.numElements());
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }

    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * temp));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void sigmoidNode<t>::backward(const tensor<t>& owner) {
    owner.requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> one(device::GPU, owner.getShape()[0], owner.getShape()[1]);
    one.ones();
    tensor<t> temp = owner * (one - owner);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * temp));
    A -> requiresGrad(true);
    owner.requiresGrad(true);
    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void tanhNode<t>::backward(const tensor<t>& owner) {
    A->requiresGrad(false);
    tensor<t> temp = A->tanh();
    temp *= temp;
    tensor<t> one(device::GPU, A->getShape()[0], A->getShape()[1]);
    one.ones();
    temp = one - temp;
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * temp));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
__global__ void geluGradKernel(const t* tens, t* out, size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    constexpr t root2OnRootPi = t(0.79788456080286535587989211986876L);
    constexpr t geluConst = t(0.044715);
    constexpr t geluGradConst = t(0.134145);

    t u = root2OnRootPi * (tens[idx] + tens[idx] * tens[idx] * tens[idx] * geluConst);
    u = tanh(u);
    out[idx] = 0.5 * ((1 + u) + tens[idx] * (1 - u * u) * root2OnRootPi * (1 + geluGradConst * tens[idx] * tens[idx])); 

}

template <typename t>
void geluNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape()[0], A->getShape()[1]);
    geluGradKernel<<<cuda::ceil_div(temp.numElements(), 256), 256>>>(A->data(),temp.data(), temp.numElements());
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * temp));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
__global__ void broadcastSubtractKernel(t* A, t* B, t* out, size_t row, size_t col) {
    size_t idxX = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idxY = threadIdx.y + blockDim.y * blockIdx.y;

    if (idxX >= col ||  idxY >= row) return;

    out[idxY*col + idxX] = A[idxY*col + idxX] - B[idxY];
}

template <typename t>
void softmaxNode<t>::backward(const tensor<t>& owner) {
    // std::cout << "Backward, A = " << A.get() << '\n';
    // std::cout<<A->numElements()<<"run\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    owner.requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape()[0], A->getShape()[1]);
    tensor<t> dotProd = (*owner.gradient() * owner).rowSum();
    dim3 blocks = dim3(cuda::ceil_div(A->getShape()[1], 16), cuda::ceil_div(A->getShape()[0], 16));
    dim3 threads = dim3(16, 16);
    broadcastSubtractKernel<<<blocks, threads>>>(owner.gradient()->data(), dotProd.data(), temp.data(), A->getShape()[0], A->getShape()[1]);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    if (A -> gradient()) *A -> gradient() += owner * temp;
    else A -> setGradient(std::make_shared<tensor<t>>(owner * temp));
    A -> requiresGrad(true);
    owner.requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
__global__ void crossEntropyGradKernel(const t* pred, const t* targ, t* out, size_t storageLength, size_t rows) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = - (targ[idx] / (rows * pred[idx]));
}

template <typename t>
void crossEntropyLossNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape()[0], A->getShape()[1]);
    crossEntropyGradKernel<<<cuda::ceil_div(temp.numElements(), 256), 256>>>(A->data(), B->data(), temp.data(), temp.numElements(), temp.getShape()[0]);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    
    if (A-> gradient()) *A-> gradient() += *owner.gradient() * temp;
    else A-> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * temp));
    A-> requiresGrad(true);

    if (A-> gradientFunction()) A-> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void batchNode<t>::backward(const tensor<t>& owner) {
    A->requiresGrad(false);
    if (!axis) {    
        if (A -> gradient()) *A -> gradient() += owner.gradient()->colSum();
        else A -> setGradient(std::make_shared<tensor<t>>(owner.gradient()->colSum()));
    }
    else {
        if (A -> gradient()) *A -> gradient() += owner.gradient()->rowSum();
        else A -> setGradient(std::make_shared<tensor<t>>(owner.gradient()->rowSum()));
    }
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
__global__ void tokenEmbeddingNodeKernel(t* grad, const t* outGrad, const TokenID* token, const size_t len, const size_t dim) {
    size_t tokenIdx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dimIdx = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (tokenIdx >= len || dimIdx >= dim) return;

    atomicAdd(&grad[token[tokenIdx] * dim + dimIdx], outGrad[tokenIdx * dim + dimIdx]);
}

template <typename t>
void tokenEmbeddingNode<t>::backward(const tensor<t>& owner) {
    size_t len = tokenIds.size();
    TokenID* tokenIdsCpy = new TokenID[len];
    for (int i = 0; i < len; i++) tokenIdsCpy[i] = tokenIds[i];
    weight->requiresGrad(false);
    if (!weight->gradient()) {
        weight->setGradient(std::make_shared<tensor<t>>(device::GPU, weight->getShape()[0], weight->getShape()[1]));
        weight->gradient()->zeros();
    }
    TokenID* temp;
    cudaError_t err = cudaMalloc(&temp, len * sizeof(TokenID));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    err = cudaMemcpy(temp, tokenIdsCpy, len * sizeof(TokenID), cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    } 
    delete[] tokenIdsCpy;  
    tokenEmbeddingNodeKernel<<<dim3(cuda::ceil_div(len, 16),cuda::ceil_div(weight->getShape()[1], 16)), dim3(16, 16)>>>(weight->gradient()->data(), owner.gradient()->data(), temp, len, weight->getShape()[1]);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    cudaFree(temp);
    weight -> requiresGrad(true);

    if (weight -> gradientFunction()) weight -> gradientFunction() -> backward(*weight.get());
    weight -> clearGradientFunction();
}

template <typename t>
__global__ void positionEmbeddingNodeKernel(t* grad, const t* outGrad, const size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    grad[idx] += outGrad[idx];
}

template <typename t>
void positionEmbeddingNode<t>::backward(const tensor<t>& owner) {
    weight->requiresGrad(false);
    if (!weight->gradient()) {
        weight->setGradient(std::make_shared<tensor<t>>(device::GPU, weight->getShape()[0], weight->getShape()[1]));
        weight->gradient()->zeros();
    }
    positionEmbeddingNodeKernel<<<cuda::ceil_div(owner.gradient()->numElements(), 256), 256>>>(weight->gradient()->data(), owner.gradient()->data(), owner.gradient()->numElements());
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    weight -> requiresGrad(true);

    if (weight -> gradientFunction()) weight -> gradientFunction() -> backward(*weight.get());
    weight -> clearGradientFunction();
}

template <typename t>
void singleHeadAttentionNode<t>::backward(const tensor<t>& owner) {
    input->requiresGrad(false);
    owner.gradient()->requiresGrad(false);
    wQuery->requiresGrad(false);
    wKey->requiresGrad(false);
    wVal->requiresGrad(false);
    auto dV = score->transposed().matMul(*owner.gradient());
    if (score-> gradient()) *score-> gradient() += owner.gradient()->matMul(V->transposed());
    else score-> setGradient(std::make_shared<tensor<t>>(owner.gradient()->matMul(V->transposed())));
    if (wVal-> gradient()) *wVal-> gradient() += input->transposed().matMul(dV);
    else wVal-> setGradient(std::make_shared<tensor<t>>(input->transposed().matMul(dV)));
    tensor<t> tempSoftGrad(device::GPU, score->getShape()[0], score->getShape()[1]);
    softmaxNode<t> temp(&tempSoftGrad);
    temp.backward(*score.get());
    score->requiresGrad(false);
    auto& softmaxGrad = *tempSoftGrad.gradient();
    softmaxGrad = softmaxGrad / sqrt(wQuery->getShape()[1]);
    auto dQ = softmaxGrad.matMul(*K.get());
    auto dK = softmaxGrad.transposed().matMul(*Q.get());
    if (wKey-> gradient()) *wKey-> gradient() += input->transposed().matMul(dK);
    else wKey-> setGradient(std::make_shared<tensor<t>>(input->transposed().matMul(dK)));
    if (wQuery-> gradient()) *wQuery-> gradient() += input->transposed().matMul(dQ);
    else wQuery-> setGradient(std::make_shared<tensor<t>>(input->transposed().matMul(dQ)));
    if (input-> gradient()) *input-> gradient() += dV.matMul(wVal->transposed()) + dK.matMul(wKey->transposed()) + dQ.matMul(wQuery->transposed());
    else input-> setGradient(std::make_shared<tensor<t>>(dV.matMul(wVal->transposed()) + dK.matMul(wKey->transposed()) + dQ.matMul(wQuery->transposed())));
    wQuery->requiresGrad(true);
    wKey->requiresGrad(true);
    wVal->requiresGrad(true);
    input->requiresGrad(true);
    score->clearGrad();
    if (input -> gradientFunction()) input -> gradientFunction() -> backward(*input.get());
    input -> clearGradientFunction();
}

template <typename t>
__global__ void scatterKernel(t* grad, const t* ownerGrad, const TokenID* targ, const size_t targLength, const size_t vocabLen) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= targLength) return;

    grad[idx * vocabLen + targ[idx]] = ownerGrad[idx];
}

template <typename t>
void gatherNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> grad(device::GPU, A->getShape()[0], A->getShape()[1]);
    grad.zeros();
    TokenID* temp;
    cudaError_t err = cudaMalloc(&temp, B->size() * sizeof(TokenID));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    err = cudaMemcpy(temp, B->data(), B->size() * sizeof(TokenID), cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    owner.gradient()->toGPU();
    scatterKernel<<<cuda::ceil_div(B->size(), 256), 256>>>(grad.data(), owner.gradient()->data(), temp, B->size(), grad.getShape()[1]);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    cudaFree(temp);
    if (A-> gradient()) *A-> gradient() += grad;
    else A-> setGradient(std::make_shared<tensor<t>>(grad));
    A-> requiresGrad(true);

    if (A-> gradientFunction()) A-> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void rowSumNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);   
    size_t colSize; 
    if (A->getShape().size() == 2) colSize = A -> getShape()[1];
    else if (A->getShape().size() == 3) colSize = A -> getShape()[2];
    else throw std::invalid_argument("rowSumNode only supports rank-2 and rank-3 tensors");
    if (A-> gradient()) *A-> gradient() += owner.gradient()->batch(colSize, 1);
    else A-> setGradient(std::make_shared<tensor<t>>(owner.gradient()->batch(colSize, 1)));
    A-> requiresGrad(true);

    if (A-> gradientFunction()) A-> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}

template <typename t>
void colSumNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);   
    size_t rowSize; 
    if (A->getShape().size() == 2) rowSize = A -> getShape()[0];
    else if (A->getShape().size() == 3) rowSize = A -> getShape()[1];
    else throw std::invalid_argument("colSumNode only supports rank-2 and rank-3 tensors");
    if (A-> gradient()) *A-> gradient() += owner.gradient()->batch(rowSize, 0);
    else A-> setGradient(std::make_shared<tensor<t>>(owner.gradient()->batch(rowSize, 0)));
    A-> requiresGrad(true);

    if (A-> gradientFunction()) A-> gradientFunction() -> backward(*A.get());
    A -> clearGradientFunction();
}