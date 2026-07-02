template <typename t>
class tensor;

class node {
public:
    virtual ~node() = default;

    virtual void backward() = 0;
};

template<typename t>
class addNode : public node {
public:
    addNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    tensor<t>* A;
    tensor<t>* B;

    void backward() override {}
};

template<typename t>
class subtractNode : public node {
public:
    subtractNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    tensor<t>* A;
    tensor<t>* B;

    void backward() override {}
};

template<typename t>
class multiplyNode : public node {
public:
    multiplyNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    tensor<t>* A;
    tensor<t>* B;

    void backward() override {}
};

template<typename t>
class divideNode : public node {
public:
    divideNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    tensor<t>* A;
    tensor<t>* B;

    void backward() override {}
};

template<typename t>
class matMulNode : public node {
public:

    matMulNode(tensor<t>* A, tensor<t>* B) : A(A), B(B) {}

    tensor<t>* A;
    tensor<t>* B;

    void backward() override {}
};

