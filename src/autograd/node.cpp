#include "autograd/node.hpp"
#include "tensor/tensor.hpp"

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
    else B -> setGradient(new tensor(*owner.gradient()));
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