#include <vector>

template <typename t>
class tensor;

class node {
public:
    virtual ~node() = default;

    virtual void backward() = 0;
};

template<typename t>
class addNode : public node {
    tensor<t>* A;
    tensor<t>* B;
public:
    addNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    void backward() override {}
};

template<typename t>
class subtractNode : public node {
    tensor<t>* A;
    tensor<t>* B;
public:
    subtractNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    void backward() override {}
};

template<typename t>
class multiplyNode : public node {
    tensor<t>* A;
    tensor<t>* B;
public:
    multiplyNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    void backward() override {}
};

template<typename t>
class divideNode : public node {
    tensor<t>* A;
    tensor<t>* B;
public:
    divideNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    void backward() override {}
};

template<typename t>
class matMulNode : public node {
    tensor<t>* A;
    tensor<t>* B;
public:

    matMulNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    void backward() override {}
};

template<typename t>
class transposeNode : public node {
    tensor<t>* A;
public:

    transposeNode(tensor<t>* A) : A(A) {}

    void backward() override {}
};

template<typename t>
class sumNode : public node {
    tensor<t>* A;
public:

    sumNode(tensor<t>* A) : A(A) {}

    void backward() override {}
};

template<typename t>
class meanNode : public node {
    tensor<t>* A;
public:

    meanNode(tensor<t>* A) : A(A) {}

    void backward() override {}
};

template<typename t>
class reshapeNode : public node {
    tensor<t>* A;
    std::vector<size_t> oldShape; 
public:

    reshapeNode(tensor<t>* A, std::vector<size_t> oldShape) : A(A), oldShape(oldShape) {}

    void backward() override {}
};

template<typename t>
class expNode : public node {
    tensor<t>* A;
public:

    expNode(tensor<t>* A) : A(A) {}

    void backward() override {}
};

template<typename t>
class logNode : public node {
    tensor<t>* A;
public:

    logNode(tensor<t>* A) : A(A) {}

    void backward() override {}
};

template<typename t>
class powNode : public node {
    tensor<t>* A;
    t power;
public:

    powNode(tensor<t>* A, t power) : A(A), power(power) {}

    void backward() override {}
};

template<typename t>
class reluNode : public node {
    tensor<t>* A;
public:

    reluNode(tensor<t>* A) : A(A) {}

    void backward() override {}
};

template<typename t>
class sigmoidNode : public node {
    tensor<t>* A;
public:

    sigmoidNode(tensor<t>* A) : A(A) {}

    void backward() override {}
};

template<typename t>
class tanhNode : public node {
    tensor<t>* A;
public:

    tanhNode(tensor<t>* A) : A(A) {}

    void backward() override {}
};

template<typename t>
class geluNode : public node {
    tensor<t>* A;
public:

    geluNode(tensor<t>* A) : A(A) {}

    void backward() override {}
};