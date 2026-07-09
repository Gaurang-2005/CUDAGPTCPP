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

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B);
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

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B);
}

template <typename t>
void multiplyNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B);
    else A -> setGradient(new tensor(*owner.gradient() * (*B)));
    if (B -> gradient()) *B -> gradient() += *owner.gradient() * (*A);
    else B -> setGradient(new tensor(*owner.gradient() * (*A)));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B);
}

template <typename t>
void divideNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B);
    else A -> setGradient(new tensor(*owner.gradient() / (*B)));
    if (B -> gradient()) *B -> gradient() -= (*owner.gradient() * (*A))/((*B)*(*B));
    else B -> setGradient(new tensor(-(*owner.gradient() * (*A))/((*B)*(*B))));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B);
}

template <typename t>
void matMulNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += (*owner.gradient()).matMul((*B).transposed());
    else A -> setGradient(new tensor((*owner.gradient()).matMul((*B).transposed())));
    if (B -> gradient()) *B -> gradient() += ((*A).transposed()).matMul(*owner.gradient());
    else B -> setGradient(new tensor(((*A).transposed()).matMul(*owner.gradient())));
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B);
}

template <typename t>
void transposeNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += (*owner.gradient()).transposed();
    else A -> setGradient(new tensor((*owner.gradient()).transposed()));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
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

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
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

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
}
    
template <typename t>
void reshapeNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += owner.gradient()->reshaped(oldShape[0], oldShape[1]);
    else A -> setGradient(new tensor(owner.gradient()->reshaped(oldShape[0], oldShape[1])));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
}

template <typename t>
void expNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * A -> exp();
    else A -> setGradient(new tensor(*(owner.gradient()) * A -> exp()));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
}

template <typename t>
void logNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(*A);
    temp.ones();
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * temp / *A;
    else A -> setGradient(new tensor(*(owner.gradient()) * temp / *A));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
}

template <typename t>
void powNode<t>::backward(const tensor<t>& owner) {
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * A -> pow(power - 1) * power;
    else A -> setGradient(new tensor(*(owner.gradient()) * A -> pow(power - 1) * power));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
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

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
}

template <typename t>
void sigmoidNode<t>::backward(const tensor<t>& owner) {
    A->requiresGrad(false);
    tensor<t> temp = A->sigmoid();
    tensor<t> one(device::GPU, A->getShape()[0], A->getShape()[1]);
    one.ones();
    temp = temp * (one - temp);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(new tensor(*owner.gradient() * temp));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
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

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
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

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A);
}