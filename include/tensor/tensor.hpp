#pragma once

#include <vector>
#include <concepts>
#include <cassert>
#include "autograd/node.hpp"

enum class device {
    CPU,
    GPU
};

template <typename t>
class tensor {
    std::vector<size_t> shape;
    size_t storageLength = 1;
    mutable t* tens = nullptr;
    mutable device dev;
    mutable bool isGradEnabled = false;
    mutable std::shared_ptr<node<t>> gradFunction = nullptr;
    mutable tensor* grad = nullptr;
    bool isIdentity = false;
public:
    template <typename ... Args>
    requires (std::integral<Args> && ...)
    tensor(device dev, Args...args) : shape({static_cast<size_t>(args)...}), dev(dev) {

        for (auto& i : shape) {
            storageLength*=i;
        }
        if (dev == device::CPU)
        tens = new t[storageLength]{};

        else if (dev == device::GPU) {
            constructorAllocate();
        }
    }

    tensor(device dev, std::initializer_list<std::initializer_list<t>> list);

    template <typename ... Args>
    requires (std::integral<Args> && ...)
    tensor(Args...args) : tensor(device::CPU, args...) {}
    tensor() : tensor(device::CPU, 1, 1) {}
    t* data() {
        return tens;
    }
    tensor*& gradient() {
        return grad;
    }
    tensor* gradient() const {
        return grad;
    }
    const t* data() const {
        return tens;
    }
    std::shared_ptr<node<t>> gradientFunction() {
        return gradFunction;
    }
    const std::shared_ptr<node<t>> gradientFunction() const {
        return gradFunction;
    }
    void setGradient(tensor<t>* gradient) const {
        grad = gradient;
    }
    void setGradientFunction(std::shared_ptr<node<t>> gradFunction) const {
        this->gradFunction = gradFunction;
    }
    void constructorAllocate();
    void toGPU() const;
    void toCPU() const;

    tensor(const tensor& other);
    tensor(tensor&& other) noexcept;
    tensor& operator=(const tensor& other);
    tensor& operator=(tensor&& other) noexcept;
    ~tensor();

    template <typename ... Args>
    requires (std::integral<Args> && ...)
    t& operator()(Args...args) {
        assert(sizeof...(args) == shape.size());
        size_t arg[] = {static_cast<size_t>(args)...};
        size_t index = 0;
        for (size_t i = 0; i < sizeof...(args); ++i) {
            assert(arg[i] < shape[i]);
            index = index * shape[i] + arg[i];
        }
        return tens[index];
    }

    template <typename ... Args>
    requires (std::integral<Args> && ...)
    const t& operator()(Args...args) const {
        assert(sizeof...(args) == shape.size());
        size_t arg[] = {static_cast<size_t>(args)...};
        size_t index = 0;
        for (size_t i = 0; i < sizeof...(args); ++i) {
            assert(arg[i] < shape[i]);
            index = index * shape[i] + arg[i];
        }
        return tens[index];
    }

    size_t rank() const {
        return shape.size();
    }
    size_t numElements() const {
        return storageLength;
    }
    const std::vector<size_t>& getShape() const {
        return shape;
    }
    device getDevice() const {
        return dev;
    }
    bool isCPU() const {
        return dev == device::CPU;
    }
    bool isGPU() const {
        return dev == device::GPU;
    }

    void fill(t val);
    void zeros() {
        fill(0);
    }
    void ones() {
        fill(1);
    }
    void random();
    bool isEmpty() const {
        return storageLength == 0;
    }

    tensor& operator+=(const tensor& other);
    tensor& operator-=(const tensor& other);
    tensor operator+(const tensor& other) const &;
    tensor operator-(const tensor& other) const &;
    tensor operator+(const tensor& other) &&;
    tensor operator-(const tensor& other) &&;
    tensor operator+(tensor&& other) const &;
    tensor operator-(tensor&& other) const &;
    tensor operator+(tensor&& other) &&;
    tensor operator-(tensor&& other) &&;
    tensor& operator*=(const tensor& other);
    tensor& operator/=(const tensor& other);
    tensor operator*(const tensor& other) const &;
    tensor operator/(const tensor& other) const &;
    tensor operator*(const tensor& other) &&;
    tensor operator/(const tensor& other) &&;
    tensor operator*(tensor&& other) const &;
    tensor operator/(tensor&& other) const &;
    tensor operator*(tensor&& other) &&;
    tensor operator/(tensor&& other) &&;

    template <typename ... Args>
    requires (std::integral<Args> && ...)
    void reshape(Args...args) {
        if (isGradEnabled) throw std::invalid_argument("Cannot use in-place operations when autograd is enabled");
        std::vector<size_t> newShape = {static_cast<size_t>(args)...};
        size_t newStorageLength = 1;
        for (auto& i : newShape) {
            newStorageLength *= i;
        }
        assert(newStorageLength == storageLength);
        shape = newShape;
    }

    template <typename ... Args>
    requires (std::integral<Args> && ...)
    tensor reshaped(Args...args) const & {
        tensor<t> temp(*this);
        temp.shape = std::vector<size_t>({static_cast<size_t>(args)...});
        size_t newStorageLength = 1;
        for (auto& i : temp.shape) {
            newStorageLength *= i;
        }
        assert(newStorageLength == storageLength);
        if (isGradEnabled) {
            temp.gradFunction = std::make_shared<reshapeNode<t>>(this, shape);
            temp.isGradEnabled = true;
        }
        return temp;
    }

    template <typename ... Args>
    requires (std::integral<Args> && ...)
    tensor reshaped(Args...args) const && {
        tensor<t> temp(*this);
        temp.shape = std::vector<size_t>({static_cast<size_t>(args)...});
        size_t newStorageLength = 1;
        for (auto& i : temp.shape) {
            newStorageLength *= i;
        }
        assert(newStorageLength == storageLength);
        if (isGradEnabled) {
            std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
            temp.gradFunction = std::make_shared<reshapeNode<t>>(first, shape);
            temp.isGradEnabled = true;
        }
        return temp;
    }

    void print() const;
    tensor transposed() const &;
    tensor transposed() &&;
    tensor& transpose();

    tensor matMul(const tensor<t>& other) const &;
    tensor matMul(const tensor<t>& other) &&;
    tensor matMul(tensor<t>&& other) const &;
    tensor matMul(tensor<t>&& other) &&;

    void requiresGrad(bool val) const {
        isGradEnabled = val;
    }

    bool requiresGrad() const {
        return isGradEnabled;
    }

    void identity();
    tensor sum() const &;
    tensor sum() &&;
    tensor mean() const & {
        tensor<t> out;
        if (isGradEnabled) {
            isGradEnabled = false;
            out = sum();
            out.toCPU();
            out.tens[0] /= storageLength;
            isGradEnabled = true;
        }
        else {
            out = sum();
            out.toCPU();
            out.tens[0] /= storageLength;
        }
        if (isGradEnabled) {
            out.isGradEnabled = true;
            out.gradFunction = std::make_shared<meanNode<t>>(this);
        }
        return out;
    }
    tensor mean() && {
        tensor<t> out;
        if (isGradEnabled) {
            isGradEnabled = false;
            out = sum();
            out.toCPU();
            out.tens[0] /= storageLength;
            isGradEnabled = true;
        }
        else {
            out = sum();
            out.toCPU();
            out.tens[0] /= storageLength;
        }
        if (isGradEnabled) {
            std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(*this));
            out.isGradEnabled = true;
            out.gradFunction = std::make_shared<meanNode<t>>(first);
        }
        return out;
    }

    void backward() {
        if (!isGradEnabled) throw std::invalid_argument("Gradient is not enabled on this tensor, so backward failed!");

        if (grad) {
            delete grad;
        }
        grad = new tensor(device::GPU, gradFunction -> shape()[0], gradFunction -> shape()[1]);
        grad->ones();

        if (gradFunction) gradFunction -> backward(*this);
    }

    tensor operator-() const;

    tensor operator*(t val) const;
    tensor operator/(t val) const ;

    tensor exp() const &;
    tensor exp() &&;
    tensor pow(t power) const &;
    tensor pow(t power) &&;
    tensor log() const &;
    tensor log() &&;

    tensor rowSum() const;
    tensor rowMax() const;
    tensor colSum() const;

    // //Activation Functions
    tensor ReLU() const &;
    tensor ReLU() &&;
    tensor sigmoid() const &;
    tensor sigmoid() &&;
    tensor tanh() const &;
    tensor tanh() &&;
    tensor gelu() const &;
    tensor gelu() &&;
    tensor softmax() const &;
    tensor softmax() &&;

    void clearGrad() {
        if (grad) delete grad;
    }

    tensor batch(size_t batchSize, int axis = 0) const &;
    tensor batch(size_t batchSize, int axis = 0) &&;

    friend tensor operator*(t other, const tensor<t>& me) {
        return me * other;
    }
};