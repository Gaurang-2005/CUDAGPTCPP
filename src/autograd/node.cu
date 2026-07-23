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

template <typename t>
void addNode<t>::backward(const tensor<t>& owner) {
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient();
    else A -> setGradient(new tensor(*owner.gradient()));
    if (B -> gradient()) *B -> gradient() += *owner.gradient();
    else B -> setGradient(new tensor(*owner.gradient()));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
}

template <typename t>
void subtractNode<t>::backward(const tensor<t>& owner) {
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient();
    else A -> setGradient(new tensor(*owner.gradient()));
    if (B -> gradient()) *B -> gradient() -= *owner.gradient();
    else B -> setGradient(new tensor(-*owner.gradient()));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
}

template <typename t>
void multiplyNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
    else A -> setGradient(new tensor(*owner.gradient() * (*B.get())));
    if (B -> gradient()) *B -> gradient() += *owner.gradient() * (*A.get());
    else B -> setGradient(new tensor(*owner.gradient() * (*A.get())));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
}

template <typename t>
void divideNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B.get());
    else A -> setGradient(new tensor(*owner.gradient() / (*B.get())));
    if (B -> gradient()) *B -> gradient() -= (*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()));
    else B -> setGradient(new tensor(-(*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
}

template <typename t>
void matMulNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += (*owner.gradient()).matMul(B -> transposed());
    else A -> setGradient(new tensor((*owner.gradient()).matMul(B -> transposed())));
    if (B -> gradient()) *B -> gradient() += (A -> transposed()).matMul(*owner.gradient());
    else B -> setGradient(new tensor((A -> transposed()).matMul(*owner.gradient())));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
}

template <typename t>
void transposeNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += (*owner.gradient()).transposed();
    else A -> setGradient(new tensor((*owner.gradient()).transposed()));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
}

template <typename t>
void sumNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape()[0], A->getShape()[1]);
    owner.gradient()->toCPU();
    temp.fill(owner.gradient()->data()[0]);
    if (A -> gradient()) *A -> gradient() += temp;
    else A -> setGradient(new tensor(temp));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
}

template <typename t>
void meanNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape()[0], A->getShape()[1]);
    owner.gradient()->toCPU();
    temp.fill(owner.gradient()->data()[0]/temp.numElements());
    if (A -> gradient()) *A -> gradient() += temp;
    else A -> setGradient(new tensor(temp));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
}
    
template <typename t>
void reshapeNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += owner.gradient()->reshaped(oldShape[0], oldShape[1]);
    else A -> setGradient(new tensor(owner.gradient()->reshaped(oldShape[0], oldShape[1])));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
}

template <typename t>
void expNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * A -> exp();
    else A -> setGradient(new tensor(*(owner.gradient()) * A -> exp()));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
}

template <typename t>
void logNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(*A.get());
    temp.ones();
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * temp / *A.get();
    else A -> setGradient(new tensor(*(owner.gradient()) * temp / *A.get()));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
}

template <typename t>
void powNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * A -> pow(power - 1) * power;
    else A -> setGradient(new tensor(*(owner.gradient()) * A -> pow(power - 1) * power));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
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
    }

    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(new tensor(*owner.gradient() * temp));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
}

template <typename t>
void sigmoidNode<t>::backward(const tensor<t>& owner) {
    owner.requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> one(device::GPU, owner.getShape()[0], owner.getShape()[1]);
    one.ones();
    tensor<t> temp = owner * (one - owner);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(new tensor(*owner.gradient() * temp));
    A -> requiresGrad(true);
    owner.requiresGrad(true);
    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
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
    else A -> setGradient(new tensor(*owner.gradient() * temp));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
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
    }
    
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(new tensor(*owner.gradient() * temp));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
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
    }
    if (A -> gradient()) *A -> gradient() += owner * temp;
    else A -> setGradient(new tensor(owner * temp));
    A -> requiresGrad(true);
    owner.requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
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
    }
    
    if (A-> gradient()) *A-> gradient() += *owner.gradient() * temp;
    else A-> setGradient(new tensor(*owner.gradient() * temp));
    A-> requiresGrad(true);

    if (A-> gradientFunction()) A-> gradientFunction() -> backward(*A.get());
}

template <typename t>
void batchNode<t>::backward(const tensor<t>& owner) {
    A->requiresGrad(false);
    if (!axis) {    
        if (A -> gradient()) *A -> gradient() += owner.gradient()->colSum();
        else A -> setGradient(new tensor(owner.gradient()->colSum()));
    }
    else {
        if (A -> gradient()) *A -> gradient() += owner.gradient()->rowSum();
        else A -> setGradient(new tensor(owner.gradient()->rowSum()));
    }
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
}

template <typename t>
void layerNormNode<t>::backward(const tensor<t>& owner) {
    std::cout << "Incoming gradient:\n";
    input->requiresGrad(false);
    owner.gradient()->requiresGrad(false);
    gamma->requiresGrad(false);
    beta->requiresGrad(false);
    if (beta-> gradient()) *beta-> gradient() += owner.gradient()->colSum();
    else beta-> setGradient(new tensor(owner.gradient()->colSum()));
    if (gamma-> gradient()) *gamma-> gradient() += (*owner.gradient() * (*norm.get())).colSum();
    else gamma-> setGradient(new tensor((*owner.gradient() * (*norm.get())).colSum()));
    if (input-> gradient()) *input-> gradient() += (gamma->batch(inv->getShape()[0]) * (*inv.get()))*(*owner.gradient() - (owner.gradient()->rowSum()/owner.gradient()->getShape()[1]).batch(owner.gradient()->getShape()[1], 1) - (*norm.get()) * ((*owner.gradient() * (*norm.get())).rowSum()/owner.gradient()->getShape()[1]).batch(norm->getShape()[1], 1));
    else input-> setGradient(new tensor(((gamma->batch(inv->getShape()[0]) * (*inv.get()))*(*owner.gradient() - (owner.gradient()->rowSum()/owner.gradient()->getShape()[1]).batch(owner.gradient()->getShape()[1], 1) - (*norm.get()) * ((*owner.gradient() * (*norm.get())).rowSum()/owner.gradient()->getShape()[1]).batch(norm->getShape()[1], 1)))));
    input -> requiresGrad(true);
    gamma -> requiresGrad(true);
    beta -> requiresGrad(true);

    if (input-> gradientFunction()) input-> gradientFunction() -> backward(*input.get());
}

template <typename t>
__global__ void tokenEmbeddingNodeKernel(t* grad, const t* outGrad, const size_t* token, const size_t len, const size_t dim) {
    size_t tokenIdx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dimIdx = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (tokenIdx >= len || dimIdx >= dim) return;

    atomicAdd(&grad[token[tokenIdx] * dim + dimIdx], outGrad[tokenIdx * dim + dimIdx]);
}

template <typename t>
void tokenEmbeddingNode<t>::backward(const tensor<t>& owner) {
    weight->requiresGrad(false);
    if (!weight->gradient()) {
        weight->setGradient(new tensor<t>(device::GPU, weight->getShape()[0], weight->getShape()[1]));
        weight->gradient()->zeros();
    }
    size_t* temp;
    cudaError_t err = cudaMalloc(&temp, len * sizeof(size_t));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    err = cudaMemcpy(temp, tokenIds, len * sizeof(size_t), cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                << cudaGetErrorString(err)
                << '\n';
    }    
    tokenEmbeddingNodeKernel<<<dim3(cuda::ceil_div(len, 16),cuda::ceil_div(weight->getShape()[1], 16)), dim3(16, 16)>>>(weight->gradient()->data(), owner.gradient()->data(), temp, len, weight->getShape()[1]);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    cudaFree(temp);
    weight -> requiresGrad(true);

    if (weight -> gradientFunction()) weight -> gradientFunction() -> backward(*weight.get());
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
        weight->setGradient(new tensor<t>(device::GPU, weight->getShape()[0], weight->getShape()[1]));
        weight->gradient()->zeros();
    }
    positionEmbeddingNodeKernel<<<cuda::ceil_div(owner.gradient()->numElements(), 256), 256>>>(weight->gradient()->data(), owner.gradient()->data(), owner.gradient()->numElements());
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    weight -> requiresGrad(true);

    if (weight -> gradientFunction()) weight -> gradientFunction() -> backward(*weight.get());
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
    else score-> setGradient(new tensor(owner.gradient()->matMul(V->transposed())));
    if (wVal-> gradient()) *wVal-> gradient() += input->transposed().matMul(dV);
    else wVal-> setGradient(new tensor(input->transposed().matMul(dV)));
    tensor<t> tempSoftGrad(device::GPU, score->getShape()[0], score->getShape()[1]);
    softmaxNode<t> temp(&tempSoftGrad);
    temp.backward(*score.get());
    auto& softmaxGrad = *tempSoftGrad.gradient();
    softmaxGrad = softmaxGrad / sqrt(wQuery->getShape()[1]);
    auto dQ = softmaxGrad.matMul(*K.get());
    auto dK = softmaxGrad.transposed().matMul(*Q.get());
    if (wKey-> gradient()) *wKey-> gradient() += input->transposed().matMul(dK);
    else wKey-> setGradient(new tensor(input->transposed().matMul(dK)));
    if (wQuery-> gradient()) *wQuery-> gradient() += input->transposed().matMul(dQ);
    else wQuery-> setGradient(new tensor(input->transposed().matMul(dQ)));
    if (input-> gradient()) *input-> gradient() += dV.matMul(wVal->transposed()) + dK.matMul(wKey->transposed()) + dQ.matMul(wQuery->transposed());
    else input-> setGradient(new tensor(dV.matMul(wVal->transposed()) + dK.matMul(wKey->transposed()) + dQ.matMul(wQuery->transposed())));
    score->clearGrad();
}
