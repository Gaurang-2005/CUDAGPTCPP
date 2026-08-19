#include "autograd/node.hpp"
#include "tensor/tensor.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda/cmath>

inline bool debugGraph = false;

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

template class rowMaxNode<float>;
template class rowMaxNode<double>;

template class scalarDivideNode<float>;
template class scalarDivideNode<double>;

template class scalarMultiplyNode<float>;
template class scalarMultiplyNode<double>;

template class scalarAddNode<float>;
template class scalarAddNode<double>;

template class scalarSubtractNode<float>;
template class scalarSubtractNode<double>;

template <typename t>
void addNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "AddNode init!\n";
    owner.gradient()->requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> getShape() == B -> getShape()) {
        if (A -> gradient()) *A -> gradient() += *owner.gradient();
        else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
        if (B -> gradient()) *B -> gradient() += *owner.gradient();
        else B -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
    }
    else if (A -> getShape().size() == 2) {
        if (B -> getShape()[1] == 1 && A -> getShape()[0] == B -> getShape()[0]) {
            if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
            else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
            if (B -> gradient()) *B -> gradient() += owner.gradient() -> rowSum();
            else B -> setGradient(std::make_shared<tensor<t>>(owner.gradient() -> rowSum()));
        }
        else if (B -> getShape()[0] == 1 && A -> getShape()[1] == B -> getShape()[1]) {
            if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
            else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
            if (B -> gradient()) *B -> gradient() += owner.gradient() -> colSum();
            else B -> setGradient(std::make_shared<tensor<t>>(owner.gradient() -> colSum()));
        }
    }
    else if (A -> getShape().size() == 3) {
        if (B -> getShape().size() == 2) {
            if (B -> getShape()[1] == 1 && A -> getShape()[1] == B -> getShape()[0]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient();
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
                if (B -> gradient()) *B -> gradient() += owner.gradient() -> rowSum().batchSum();
                else B -> setGradient(std::make_shared<tensor<t>>(owner.gradient() -> rowSum().batchSum()));
            }
            else if (B -> getShape()[0] == 1 && A -> getShape()[2] == B -> getShape()[1]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient();
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
                if (B -> gradient()) *B -> gradient() += owner.gradient() -> colSum().batchSum();
                else B -> setGradient(std::make_shared<tensor<t>>(owner.gradient() -> colSum().batchSum()));
            }
        } 
        else if (B -> getShape().size() == 3) {
            if (B -> getShape()[2] == 1 && A -> getShape()[1] == B -> getShape()[1]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient();
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
                if (B -> gradient()) *B -> gradient() += owner.gradient() -> rowSum();
                else B -> setGradient(std::make_shared<tensor<t>>(owner.gradient() -> rowSum()));
            }
            else if (B -> getShape()[1] == 1 && A -> getShape()[2] == B -> getShape()[2]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient();
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
                if (B -> gradient()) *B -> gradient() += owner.gradient() -> colSum();
                else B -> setGradient(std::make_shared<tensor<t>>(owner.gradient() -> colSum()));
            }
        }
    }
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    // A -> clearGradientFunction();
    // B -> clearGradientFunction();
}

template <typename t>
void subtractNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "subtractNode init!\n";
    // if (A->DebugID() == 1) std::abort();
    owner.gradient()->requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> getShape() == B -> getShape()) {
        if (A -> gradient()) *A -> gradient() += *owner.gradient();
        else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
        if (B -> gradient()) *B -> gradient() -= *owner.gradient();
        else B -> setGradient(std::make_shared<tensor<t>>(-*owner.gradient()));
    }
    else if (A -> getShape().size() == 2) {
        if (B -> getShape()[1] == 1 && A -> getShape()[0] == B -> getShape()[0]) {
            if (A -> gradient()) *A -> gradient() += *owner.gradient();
            else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
            if (B -> gradient()) *B -> gradient() -= owner.gradient() -> rowSum();
            else B -> setGradient(std::make_shared<tensor<t>>(-owner.gradient() -> rowSum()));
        }
        else if (B -> getShape()[0] == 1 && A -> getShape()[1] == B -> getShape()[1]) {
            if (A -> gradient()) *A -> gradient() += *owner.gradient();
            else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
            if (B -> gradient()) *B -> gradient() -= owner.gradient() -> colSum();
            else B -> setGradient(std::make_shared<tensor<t>>(-owner.gradient() -> colSum()));
        }
    }
    else if (A -> getShape().size() == 3) {
        if (B -> getShape().size() == 2) {
            if (B -> getShape()[1] == 1 && A -> getShape()[1] == B -> getShape()[0]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient();
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
                if (B -> gradient()) *B -> gradient() -= owner.gradient() -> rowSum().batchSum();
                else B -> setGradient(std::make_shared<tensor<t>>(-owner.gradient() -> rowSum().batchSum()));
            }
            else if (B -> getShape()[0] == 1 && A -> getShape()[2] == B -> getShape()[1]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient();
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
                if (B -> gradient()) *B -> gradient() -= owner.gradient() -> colSum().batchSum();
                else B -> setGradient(std::make_shared<tensor<t>>(-owner.gradient() -> colSum().batchSum()));
            }
        } 
        else if (B -> getShape().size() == 3) {
            if (B -> getShape()[2] == 1 && A -> getShape()[1] == B -> getShape()[1]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient();
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
                if (B -> gradient()) *B -> gradient() -= owner.gradient() -> rowSum();
                else B -> setGradient(std::make_shared<tensor<t>>(-owner.gradient() -> rowSum()));
            }
            else if (B -> getShape()[1] == 1 && A -> getShape()[2] == B -> getShape()[2]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient();
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
                if (B -> gradient()) *B -> gradient() -= owner.gradient() -> colSum();
                else B -> setGradient(std::make_shared<tensor<t>>(-owner.gradient() -> colSum()));
            }
        }
    }
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    // A -> clearGradientFunction();
    // B -> clearGradientFunction();
}

template <typename t>
void multiplyNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "multiplyNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> getShape() == B -> getShape()) {
        if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
        else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
        if (B -> gradient()) *B -> gradient() += *owner.gradient() * (*A.get());
        else B -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*A.get())));
    }
    else if (A -> getShape().size() == 2) {
        if (B -> getShape()[1] == 1 && A -> getShape()[0] == B -> getShape()[0]) {
            if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
            else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
            if (B -> gradient()) *B -> gradient() += (*owner.gradient() * *A.get()).rowSum();
            else B -> setGradient(std::make_shared<tensor<t>>((*owner.gradient() * *A.get()).rowSum()));
        }
        else if (B -> getShape()[0] == 1 && A -> getShape()[1] == B -> getShape()[1]) {
            if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
            else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
            if (B -> gradient()) *B -> gradient() += (*owner.gradient() * *A.get()).colSum();
            else B -> setGradient(std::make_shared<tensor<t>>((*owner.gradient() * *A.get()).colSum()));
        }
    }
    else if (A -> getShape().size() == 3) {
        if (B -> getShape().size() == 2) {
            if (B -> getShape()[1] == 1 && A -> getShape()[1] == B -> getShape()[0]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
                if (B -> gradient()) *B -> gradient() += (*owner.gradient() * *A.get()).rowSum().batchSum();
                else B -> setGradient(std::make_shared<tensor<t>>((*owner.gradient() * *A.get()).rowSum().batchSum()));
            }
            else if (B -> getShape()[0] == 1 && A -> getShape()[2] == B -> getShape()[1]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
                if (B -> gradient()) *B -> gradient() += (*owner.gradient() * *A.get()).colSum().batchSum();
                else B -> setGradient(std::make_shared<tensor<t>>((*owner.gradient() * *A.get()).colSum().batchSum()));
            }
        } 
        else if (B -> getShape().size() == 3) {
            if (B -> getShape()[2] == 1 && A -> getShape()[1] == B -> getShape()[1]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
                if (B -> gradient()) *B -> gradient() += (*owner.gradient() * *A.get()).rowSum();
                else B -> setGradient(std::make_shared<tensor<t>>((*owner.gradient() * *A.get()).rowSum()));
            }
            else if (B -> getShape()[1] == 1 && A -> getShape()[2] == B -> getShape()[2]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient() * (*B.get());
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (*B.get())));
                if (B -> gradient()) *B -> gradient() += (*owner.gradient() * *A.get()).colSum();
                else B -> setGradient(std::make_shared<tensor<t>>((*owner.gradient() * *A.get()).colSum()));
            }
        }
    }
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    // A -> clearGradientFunction();
    // B -> clearGradientFunction();
}

template <typename t>
void divideNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "divideNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> getShape() == B -> getShape()) {
        if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B.get());
        else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() / (*B.get())));
        if (B -> gradient()) *B -> gradient() -= ((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get())));
        else B -> setGradient(std::make_shared<tensor<t>>(-((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get())))));
    }
    else if (A -> getShape().size() == 2) {
        if (B -> getShape()[1] == 1 && A -> getShape()[0] == B -> getShape()[0]) {
            if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B.get());
            else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() / (*B.get())));
            if (B -> gradient()) *B -> gradient() -= ((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).rowSum();
            else B -> setGradient(std::make_shared<tensor<t>>(-((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).rowSum()));
        }
        else if (B -> getShape()[0] == 1 && A -> getShape()[1] == B -> getShape()[1]) {
            if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B.get());
            else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() / (*B.get())));
            if (B -> gradient()) *B -> gradient() -= ((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).colSum();
            else B -> setGradient(std::make_shared<tensor<t>>(-((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).colSum()));
        }
    }
    else if (A -> getShape().size() == 3) {
        if (B -> getShape().size() == 2) {
            if (B -> getShape()[1] == 1 && A -> getShape()[1] == B -> getShape()[0]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B.get());
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() / (*B.get())));
                if (B -> gradient()) *B -> gradient() -= ((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).rowSum().batchSum();
                else B -> setGradient(std::make_shared<tensor<t>>(-((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).rowSum().batchSum()));
            }
            else if (B -> getShape()[0] == 1 && A -> getShape()[2] == B -> getShape()[1]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B.get());
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() / (*B.get())));
                if (B -> gradient()) *B -> gradient() -= ((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).colSum().batchSum();
                else B -> setGradient(std::make_shared<tensor<t>>(-((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).colSum().batchSum()));
            }
        } 
        else if (B -> getShape().size() == 3) {
            if (B -> getShape()[2] == 1 && A -> getShape()[1] == B -> getShape()[1]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B.get());
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() / (*B.get())));
                if (B -> gradient()) *B -> gradient() -= ((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).rowSum();
                else B -> setGradient(std::make_shared<tensor<t>>(-((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).rowSum()));
            }
            else if (B -> getShape()[1] == 1 && A -> getShape()[2] == B -> getShape()[2]) {
                if (A -> gradient()) *A -> gradient() += *owner.gradient() / (*B.get());
                else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() / (*B.get())));
                if (B -> gradient()) *B -> gradient() -= ((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).colSum();
                else B -> setGradient(std::make_shared<tensor<t>>(-((*owner.gradient() * (*A.get()))/((*B.get())*(*B.get()))).colSum()));
            }
        }
    }
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    // A -> clearGradientFunction();
    // B -> clearGradientFunction();
}

template <typename t>
void matMulNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "matMulNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    B->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += (*owner.gradient()).matMul(B -> transposed());
    else A -> setGradient(std::make_shared<tensor<t>>((*owner.gradient()).matMul(B -> transposed())));
    if (B -> getShape().size() == A -> getShape().size()) {
        if (B -> gradient()) *B -> gradient() += (A -> transposed()).matMul(*owner.gradient());
        else B -> setGradient(std::make_shared<tensor<t>>((A -> transposed()).matMul(*owner.gradient())));
    }
    else {
        if (B -> gradient()) *B -> gradient() += (A -> transposed()).matMul(*owner.gradient()).batchSum();
        else B -> setGradient(std::make_shared<tensor<t>>((A -> transposed()).matMul(*owner.gradient()).batchSum()));
    }
    A->requiresGrad(true);
    B->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    if (B -> gradientFunction()) B -> gradientFunction() -> backward(*B.get());
    // A -> clearGradientFunction();
    // B -> clearGradientFunction();
}

template <typename t>
void transposeNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "transposeNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += (*owner.gradient()).transposed();
    else A -> setGradient(std::make_shared<tensor<t>>((*owner.gradient()).transposed()));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void sumNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "sumNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape());
    owner.gradient()->toCPU();
    temp.fill(owner.gradient()->data()[0]);
    if (A -> gradient()) *A -> gradient() += temp;
    else A -> setGradient(std::make_shared<tensor<t>>(temp));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void meanNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "meanNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape());
    owner.gradient()->toCPU();
    temp.fill(owner.gradient()->data()[0] / temp.numElements());
    if (A -> gradient()) *A -> gradient() += temp;
    else A -> setGradient(std::make_shared<tensor<t>>(temp));
    A->requiresGrad(true);
    owner.gradient()->toGPU();
    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}
    
template <typename t>
void reshapeNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "reshapeNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += owner.gradient()->reshaped(oldShape);
    else A -> setGradient(std::make_shared<tensor<t>>(owner.gradient()->reshaped(oldShape)));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void expNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "expNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * A -> exp();
    else A -> setGradient(std::make_shared<tensor<t>>(*(owner.gradient()) * A -> exp()));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void logNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "logNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(*A.get());
    temp.ones();
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * temp / *A.get();
    else A -> setGradient(std::make_shared<tensor<t>>(*(owner.gradient()) * temp / *A.get()));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void powNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "powNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *(owner.gradient()) * A -> pow(power - 1) * power;
    else A -> setGradient(std::make_shared<tensor<t>>(*(owner.gradient()) * A -> pow(power - 1) * power));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
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
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "reluNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape());
    reluGradKernel<<<cuda::ceil_div(temp.numElements(), 256), 256>>>(A->data(),temp.data(), temp.numElements());
    // cudaDeviceSynchronize();
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
    // A -> clearGradientFunction();
}

template <typename t>
void sigmoidNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "sigmoidNode init!\n";
    owner.gradient() -> requiresGrad(false);
    owner.requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> one(device::GPU, owner.getShape());
    one.ones();
    tensor<t> temp = owner * (one - owner);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * temp));
    A -> requiresGrad(true);
    owner.requiresGrad(true);
    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void tanhNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "tanhNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp = A->tanh();
    temp *= temp;
    tensor<t> one(device::GPU, A->getShape());
    one.ones();
    temp = one - temp;
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * temp));
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
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
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "geluNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape());
    A -> toGPU();
    geluGradKernel<<<cuda::ceil_div(temp.numElements(), 256), 256>>>(A->data(),temp.data(), temp.numElements());
    // cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    A -> toCPU();
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * temp;
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * temp));
    A -> requiresGrad(true);
    owner.gradient() -> toCPU();
    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
__global__ void broadcastSubtractKernel(t* A, t* B, t* out, size_t row, size_t col) {
    size_t idxX = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idxY = threadIdx.y + blockDim.y * blockIdx.y;

    if (idxX >= col ||  idxY >= row) return;

    out[blockIdx.z * row * col + idxY * col + idxX] = A[blockIdx.z * row * col + idxY * col + idxX] - B[blockIdx.z * row + idxY];
}

template <typename t>
void softmaxNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "softmaxNode init!\n";
    // std::cout << << "Backward, A = " << A.get() << '\n';
    // std::cout <<<<A->numElements()<<"run\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    owner.requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape());
    tensor<t> dotProd = (*owner.gradient() * owner).rowSum();
    dim3 blocks, threads;
    if (temp.getShape().size() == 2) {
        blocks = dim3(cuda::ceil_div(A->getShape()[1], 16), cuda::ceil_div(A->getShape()[0], 16));
        threads = dim3(16, 16);
        broadcastSubtractKernel<<<blocks, threads>>>(owner.gradient()->data(), dotProd.data(), temp.data(), A->getShape()[0], A->getShape()[1]);
    }
    if (temp.getShape().size() == 3) {
        blocks = dim3(cuda::ceil_div(A->getShape()[2], 16), cuda::ceil_div(A->getShape()[1], 16), A->getShape()[0]);
        threads = dim3(16, 16, 1);
        broadcastSubtractKernel<<<blocks, threads>>>(owner.gradient()->data(), dotProd.data(), temp.data(), A->getShape()[1], A->getShape()[2]);
    }    
    // cudaDeviceSynchronize();
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
    // A -> clearGradientFunction();
}

template <typename t>
__global__ void crossEntropyGradKernel(const t* pred, const t* targ, t* out, size_t storageLength, size_t rows) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    out[idx] = - (targ[idx] / (rows * pred[idx]));
}

template <typename t>
void crossEntropyLossNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "crossEntropyLossNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> temp(device::GPU, A->getShape());
    crossEntropyGradKernel<<<cuda::ceil_div(temp.numElements(), 256), 256>>>(A->data(), B->data(), temp.data(), temp.numElements(), ((temp.getShape().size() == 2) ? temp.getShape()[0] : temp.getShape()[0] * temp.getShape()[1]));
    // cudaDeviceSynchronize();
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
    // A -> clearGradientFunction();
}

template <typename t>
void batchNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "batchNode init!\n";
    A->requiresGrad(false);
    owner.gradient()->requiresGrad(false);
    // std::cout << << "batch Node: " << owner.DebugID() << ' ' << owner.gradient()->DebugID() << '\n';
    if (axis == 0) {    
        if (A -> gradient()) *A -> gradient() += owner.gradient()->colSum();
        else A -> setGradient(std::make_shared<tensor<t>>(owner.gradient()->colSum()));
    }
    else if (axis == 1) {
        if (A -> gradient()) *A -> gradient() += owner.gradient()->rowSum();
        else A -> setGradient(std::make_shared<tensor<t>>(owner.gradient()->rowSum()));
    }
    else {
        if (A -> gradient()) *A -> gradient() += owner.gradient()->batchSum();
        else A -> setGradient(std::make_shared<tensor<t>>(owner.gradient()->batchSum()));
    }
    A -> requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
__global__ void tokenEmbeddingNodeKernel(t* grad, const t* outGrad, const TokenID* token, const size_t len, const size_t dim) {
    size_t tokenIdx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dimIdx = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (tokenIdx >= len || dimIdx >= dim) return;

    atomicAdd(&grad[token[blockIdx.z * len + tokenIdx] * dim + dimIdx], outGrad[(blockIdx.z * len + tokenIdx) * dim + dimIdx]);
}

template <typename t>
void tokenEmbeddingNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "tokenEmbeddingNode init!\n";
    owner.gradient() -> requiresGrad(false);
    if (!batched) {
        size_t len = tokenIds.size();
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
        err = cudaMemcpy(temp, tokenIds.data(), len * sizeof(TokenID), cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        } 
        tokenEmbeddingNodeKernel<<<dim3(cuda::ceil_div(len, 16),cuda::ceil_div(weight->getShape()[1], 16)), dim3(16, 16)>>>(weight->gradient()->data(), owner.gradient()->data(), temp, len, weight->getShape()[1]);
        // cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
        cudaFree(temp);
        weight -> requiresGrad(true);
    }
    else {
        size_t batchSize = batchedTokenIDs.size();
        size_t len = batchedTokenIDs[0].size();
        std::vector<TokenID> singleBatch;
        singleBatch.reserve(batchSize * len);
        for (auto& i : batchedTokenIDs) {
            for (auto& j : i) {
                singleBatch.push_back(j);
            }
        }
        weight->requiresGrad(false);
        if (!weight->gradient()) {
            weight->setGradient(std::make_shared<tensor<t>>(device::GPU, weight->getShape()));
            weight->gradient()->zeros();
        }
        TokenID* temp;
        cudaError_t err = cudaMalloc(&temp, len * batchSize * sizeof(TokenID));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
        err = cudaMemcpy(temp, singleBatch.data(), len * batchSize * sizeof(TokenID), cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        } 
        tokenEmbeddingNodeKernel<<<dim3(cuda::ceil_div(len, 16),cuda::ceil_div(weight->getShape()[1], 16), batchSize), dim3(16, 16, 1)>>>(weight->gradient()->data(), owner.gradient()->data(), temp, len, weight->getShape()[1]);
        // cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
        cudaFree(temp);
        weight -> requiresGrad(true);
    }

    if (weight -> gradientFunction()) weight -> gradientFunction() -> backward(*weight.get());
    // weight -> clearGradientFunction();
}

template <typename t>
__global__ void positionEmbeddingNodeKernel(t* grad, const t* outGrad, const size_t storageLength) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= storageLength) return;

    grad[idx] = outGrad[idx];
}

template <typename t>
void positionEmbeddingNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "positionEmbeddingNode init!\n";
    owner.gradient() -> requiresGrad(false);
    weight->requiresGrad(false);
    if (!weight->gradient()) {
        weight->setGradient(std::make_shared<tensor<t>>(device::GPU, weight->getShape()));
        weight->gradient()->zeros();
    }
    positionEmbeddingNodeKernel<<<dim3(cuda::ceil_div(owner.gradient()->numElements(), 256)), dim3(256)>>>(weight->gradient()->data(), owner.gradient()->data(), owner.gradient()->numElements());
    //cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    weight -> requiresGrad(true);

    if (weight -> gradientFunction()) weight -> gradientFunction() -> backward(*weight.get());
    // weight -> clearGradientFunction();
}

template <typename t>
void singleHeadAttentionNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "singleHeadAttentionNode init!\n";
    input->requiresGrad(false);
    owner.gradient()->requiresGrad(false);
    wQuery->requiresGrad(false);
    wKey->requiresGrad(false);
    wVal->requiresGrad(false);
    auto dV = score->transposed().matMul(*owner.gradient());
    if (score-> gradient()) *score-> gradient() += owner.gradient()->matMul(V->transposed());
    else score-> setGradient(std::make_shared<tensor<t>>(owner.gradient()->matMul(V->transposed())));
    if (owner.gradient()->getShape().size() == 2) {
        if (wVal-> gradient()) *wVal-> gradient() += input->transposed().matMul(dV);
        else wVal-> setGradient(std::make_shared<tensor<t>>(input->transposed().matMul(dV)));
    }
    else if (owner.gradient()->getShape().size() == 3) {
        if (wVal-> gradient()) *wVal-> gradient() += input->transposed().matMul(dV).batchSum();
        else wVal-> setGradient(std::make_shared<tensor<t>>(input->transposed().matMul(dV).batchSum()));
    }
    tensor<t> tempSoftGrad(device::GPU, score->getShape());
    softmaxNode<t> temp(&tempSoftGrad);
    temp.cnt++;
    temp.backward(*score.get());
    score->requiresGrad(false);
    auto& softmaxGrad = *tempSoftGrad.gradient();
    softmaxGrad = softmaxGrad / sqrt(wQuery->getShape()[1]);
    auto dQ = softmaxGrad.matMul(*K.get());
    auto dK = softmaxGrad.transposed().matMul(*Q.get());
    if (owner.gradient()->getShape().size() == 2) {
        if (wKey-> gradient()) *wKey-> gradient() += input->transposed().matMul(dK);
        else wKey-> setGradient(std::make_shared<tensor<t>>(input->transposed().matMul(dK)));
        if (wQuery-> gradient()) *wQuery-> gradient() += input->transposed().matMul(dQ);
        else wQuery-> setGradient(std::make_shared<tensor<t>>(input->transposed().matMul(dQ)));
        if (input-> gradient()) *input-> gradient() += dV.matMul(wVal->transposed()) + dK.matMul(wKey->transposed()) + dQ.matMul(wQuery->transposed());
        else input-> setGradient(std::make_shared<tensor<t>>(dV.matMul(wVal->transposed()) + dK.matMul(wKey->transposed()) + dQ.matMul(wQuery->transposed())));
    }
    else if (owner.gradient()->getShape().size() == 3) {
        if (wKey-> gradient()) *wKey-> gradient() += input->transposed().matMul(dK).batchSum();
        else wKey-> setGradient(std::make_shared<tensor<t>>(input->transposed().matMul(dK).batchSum()));
        if (wQuery-> gradient()) *wQuery-> gradient() += input->transposed().matMul(dQ).batchSum();
        else wQuery-> setGradient(std::make_shared<tensor<t>>(input->transposed().matMul(dQ).batchSum()));
        if (input-> gradient()) *input-> gradient() += dV.matMul(wVal->transposed()) + dK.matMul(wKey->transposed()) + dQ.matMul(wQuery->transposed());
        else input-> setGradient(std::make_shared<tensor<t>>(dV.matMul(wVal->transposed()) + dK.matMul(wKey->transposed()) + dQ.matMul(wQuery->transposed())));
    }
    wQuery->requiresGrad(true);
    wKey->requiresGrad(true);
    wVal->requiresGrad(true);
    input->requiresGrad(true);
    score->clearGrad();
    if (input -> gradientFunction()) input -> gradientFunction() -> backward(*input.get());
    // input -> clearGradientFunction();
}

template <typename t>
__global__ void scatterKernel(t* grad, const t* ownerGrad, const TokenID* targ, const size_t targLength, const size_t vocabLen) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= targLength) return;

    grad[idx * vocabLen + targ[idx + blockIdx.y * targLength] + blockIdx.y * targLength * vocabLen] = ownerGrad[idx + blockIdx.y * targLength];
}

template <typename t>
void gatherNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "gatherNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    tensor<t> grad(device::GPU, A->getShape());
    grad.zeros();
    TokenID* temp;
    if (!batched) {
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
    }
    else {
        cudaError_t err = cudaMalloc(&temp, batchedB->size() * (*batchedB)[0].size() * sizeof(TokenID));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
        std::vector<TokenID> temp2;
        temp2.reserve(batchedB->size() * (*batchedB)[0].size());
        for (auto& i : *batchedB) {
            for (auto& j : i) temp2.push_back(j);
        }
        err = cudaMemcpy(temp, temp2.data(), batchedB->size() * (*batchedB)[0].size() * sizeof(TokenID), cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
        owner.gradient()->toGPU();
        scatterKernel<<<dim3(cuda::ceil_div((*batchedB)[0].size(), 256), batchedB->size()), dim3(256, 1)>>>(grad.data(), owner.gradient()->data(), temp, (*batchedB)[0].size(), grad.getShape()[2]);
    }
    //cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
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
    // A -> clearGradientFunction();
}

template <typename t>
void rowSumNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "rowSumNode init!\n";
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
    // A -> clearGradientFunction();
}

template <typename t>
void colSumNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "colSumNode init!\n";
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
    // A -> clearGradientFunction();
}

template <typename t>
__global__ void rowMaxNodeKernel(t* grad, const t* ownerGrad, const TokenID* targ, const size_t storageLen, const size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= storageLen) return;

    grad[targ[idx + blockIdx.y * storageLen] + idx * cols + blockIdx.y * storageLen * cols] = ownerGrad[idx + blockIdx.y * storageLen];
}

template <typename t>
void rowMaxNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "rowMaxNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);   
    tensor<t> temp(device::GPU, A -> getShape());
    temp.zeros();

    owner.gradient()->toGPU();
    cudaError_t err;
    if(temp.getShape().size() == 2) {
        TokenID* argVecG;
        err = cudaMalloc(&argVecG, A->getShape()[0] * sizeof(TokenID));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
        auto vec = A->argMax();
        std::vector<TokenID> argVec;
        argVec.reserve(A->getShape()[0]);
        for (auto& i : vec) {
            for (auto& j : i) argVec.push_back(j);
        }
        err = cudaMemcpy(argVecG, argVec.data(), A->getShape()[0] * sizeof(TokenID), cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
        rowMaxNodeKernel<<<cuda::ceil_div(temp.getShape()[0], 256), 256>>>(temp.data(), owner.gradient()->data(), argVecG, temp.getShape()[0], temp.getShape()[1]);
        // err = cudaDeviceSynchronize();
    }
    else if(temp.getShape().size() == 3) {
        TokenID* argVecG;
        err = cudaMalloc(&argVecG, A->getShape()[0] * A->getShape()[1] * sizeof(TokenID));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
        auto vec = A->argMax();
        std::vector<TokenID> argVec;
        argVec.reserve(A->getShape()[0] * A->getShape()[1]);
        for (auto& i : vec) {
            for (auto& j : i) argVec.push_back(j);
        }
        err = cudaMemcpy(argVecG, argVec.data(), A->getShape()[0] * A->getShape()[1] * sizeof(TokenID), cudaMemcpyDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: "
                    << cudaGetErrorString(err)
                    << '\n';
            std::abort();
        }
        rowMaxNodeKernel<<<dim3(cuda::ceil_div(temp.getShape()[1], 256), temp.getShape()[0]), dim3(256, 1)>>>(temp.data(), owner.gradient()->data(), argVecG, temp.getShape()[1], A->getShape()[2]);
        // err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    if (A-> gradient()) *A-> gradient() += temp;
    else A-> setGradient(std::make_shared<tensor<t>>(temp));
    A-> requiresGrad(true);

    if (A-> gradientFunction()) A-> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void scalarDivideNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "scalarDivideNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() / (sc);
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() / (sc)));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void scalarMultiplyNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "scalarMultiplyNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient() * (sc);
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient() * (sc)));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void scalarAddNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "scalarAddNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient();
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}

template <typename t>
void scalarSubtractNode<t>::backward(const tensor<t>& owner) {
    this -> cnt--;
    if (this -> cnt) return;
    if (debugGraph) std::cout << "scalarSubtractNode init!\n";
    owner.gradient() -> requiresGrad(false);
    A->requiresGrad(false);
    if (A -> gradient()) *A -> gradient() += *owner.gradient();
    else A -> setGradient(std::make_shared<tensor<t>>(*owner.gradient()));
    A->requiresGrad(true);

    if (A -> gradientFunction()) A -> gradientFunction() -> backward(*A.get());
    // A -> clearGradientFunction();
}