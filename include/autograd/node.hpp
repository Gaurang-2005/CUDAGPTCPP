#pragma once

#include <vector>
#include <memory>

template <typename t>
class tensor;

template<typename t>
class node {
public:
    virtual ~node() = default;

    virtual void backward(const tensor<t>& owner) = 0;
};

template <typename t>
class tensorRef {
    const tensor<t>* borrowed = nullptr;
    const std::shared_ptr<tensor<t>> rValue;
public:
    tensorRef(const std::shared_ptr<tensor<t>> rValue) : rValue(std::move(rValue)), borrowed(rValue.get()) {}
    tensorRef(const tensor<t>* borrowed) : borrowed(borrowed) {}
    const tensor<t>* operator->() const {return borrowed;}   
    const tensor<t>* get() const {return borrowed;}
};

template<typename t>
class addNode : public node<t> {
    const tensorRef<t> A;
    const tensorRef<t> B;
public:
    addNode(const tensor<t>* A, const tensor<t>* B) : A(A), B(B) {}
    addNode(const std::shared_ptr<tensor<t>> A, const tensor<t>* B) : A(A), B(B) {}
    addNode(const tensor<t>* A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    addNode(const std::shared_ptr<tensor<t>> A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class subtractNode : public node<t> {
    const tensorRef<t> A;
    const tensorRef<t> B;
public:
    subtractNode(const tensor<t>* A, const tensor<t>* B) : A(A), B(B) {}
    subtractNode(const std::shared_ptr<tensor<t>> A, const tensor<t>* B) : A(A), B(B) {}
    subtractNode(const tensor<t>* A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    subtractNode(const std::shared_ptr<tensor<t>> A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class multiplyNode : public node<t> {
    const tensorRef<t> A;
    const tensorRef<t> B;
public:
    multiplyNode(const tensor<t>* A, const tensor<t>* B) : A(A), B(B) {}
    multiplyNode(const std::shared_ptr<tensor<t>> A, const tensor<t>* B) : A(A), B(B) {}
    multiplyNode(const tensor<t>* A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    multiplyNode(const std::shared_ptr<tensor<t>> A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class divideNode : public node<t> {
    const tensorRef<t> A;
    const tensorRef<t> B;
public:
    divideNode(const tensor<t>* A, const tensor<t>* B) : A(A), B(B) {}
    divideNode(const std::shared_ptr<tensor<t>> A, const tensor<t>* B) : A(A), B(B) {}
    divideNode(const tensor<t>* A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    divideNode(const std::shared_ptr<tensor<t>> A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class matMulNode : public node<t> {
    const tensorRef<t> A;
    const tensorRef<t> B;
public:
    matMulNode(const tensor<t>* A, const tensor<t>* B) : A(A), B(B) {}
    matMulNode(const std::shared_ptr<tensor<t>> A, const tensor<t>* B) : A(A), B(B) {}
    matMulNode(const tensor<t>* A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    matMulNode(const std::shared_ptr<tensor<t>> A, const std::shared_ptr<tensor<t>> B) : A(A), B(B) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class transposeNode : public node<t> {
    const tensorRef<t> A;
public:
    transposeNode(const tensor<t>* A) : A(A) {}
    transposeNode(const std::shared_ptr<tensor<t>> A) : A(A) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class sumNode : public node<t> {
    const tensorRef<t> A;
public:
    sumNode(const tensor<t>* A) : A(A) {}
    sumNode(const std::shared_ptr<tensor<t>> A) : A(A) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class meanNode : public node<t> {
    const tensorRef<t> A;
public:

    meanNode(const tensor<t>* A) : A(A) {}
    meanNode(const std::shared_ptr<tensor<t>> A) : A(A) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class reshapeNode : public node<t> {
    const tensorRef<t> A;
    const std::vector<size_t> oldShape; 
public:
    reshapeNode(const tensor<t>* A, std::vector<size_t> oldShape) : A(A), oldShape(oldShape) {}
    reshapeNode(const std::shared_ptr<tensor<t>> A, std::vector<size_t> oldShape) : A(A), oldShape(oldShape) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class expNode : public node<t> {
    const tensorRef<t> A;
public:
    expNode(const tensor<t>* A) : A(A) {}
    expNode(const std::shared_ptr<tensor<t>> A) : A(A) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class logNode : public node<t> {
    const tensorRef<t> A;
public:
    logNode(const tensor<t>* A) : A(A) {}
    logNode(const std::shared_ptr<tensor<t>> A) : A(A) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class powNode : public node<t> {
    const tensorRef<t> A;
    const t power;
public:
    powNode(const tensor<t>* A, t power) : A(A), power(power) {}
    powNode(const std::shared_ptr<tensor<t>> A, t power) : A(A), power(power) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class reluNode : public node<t> {
    const tensorRef<t> A;
public:
    reluNode(const tensor<t>* A) : A(A) {}
    reluNode(const std::shared_ptr<tensor<t>> A) : A(A) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class sigmoidNode : public node<t> {
    const tensorRef<t> A;
public:
    sigmoidNode(const tensor<t>* A) : A(A) {}
    sigmoidNode(const std::shared_ptr<tensor<t>> A) : A(A) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class tanhNode : public node<t> {
    const tensorRef<t> A;
public:
    tanhNode(const tensor<t>* A) : A(A) {}
    tanhNode(const std::shared_ptr<tensor<t>> A) : A(A) {}
    virtual void backward(const tensor<t>& owner) override;
};

template<typename t>
class geluNode : public node<t> {
    const tensorRef<t> A;
public:
    geluNode(const tensor<t>* A) : A(A) {}
    geluNode(const std::shared_ptr<tensor<t>> A) : A(A) {}
    virtual void backward(const tensor<t>& owner) override;
};