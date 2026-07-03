#include <memory>
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
    t* tens = nullptr;
    device dev;
    bool isGradEnabled = false;
    node* gradFunction = nullptr;
    tensor* grad = nullptr;
public:
    template <typename ... Args>
    requires (std::integral<Args> && ...)
    tensor(device dev, Args...args) : shape({static_cast<size_t>(args)...}), dev(dev) {

        for (auto& i : shape) {
            storageLength*=i;
        }
        if (dev == device::CPU)
        tens = new t[storageLength];

        else if (dev == device::GPU) {
            constructorAllocate();
        }
    }
    template <typename ... Args>
    requires (std::integral<Args> && ...)
    tensor(Args...args) : tensor(device::CPU, args...) {}
    tensor() : tensor(device::CPU, 1, 1) {}
    t* data() {
        return tens;
    }

    const t* data() const {
        return tens;
    }
    void constructorAllocate();
    void toGPU();
    void toCPU();

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
    size_t numelements() const {
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

    tensor& operator+=(tensor& other);
    tensor& operator-=(tensor& other);
    tensor operator+(tensor& other);
    tensor operator-(tensor& other);
    tensor& operator*=(tensor& other);
    tensor& operator/=(tensor& other);
    tensor operator*(tensor& other);
    tensor operator/(tensor& other);

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
    tensor reshaped(Args...args) const {
        tensor<t> temp(*this);
        temp.shape = std::vector<size_t>({static_cast<size_t>(args)...});
        size_t newStorageLength = 1;
        for (auto& i : temp.shape) {
            newStorageLength *= i;
        }
        assert(newStorageLength == storageLength);
        if (isGradEnabled) {
            temp.gradFunction = new reshapeNode<t>(this, shape);
            temp.isGradEnabled = true;
        }
        return temp;
    }

    void print() const;
    tensor transposed();
    tensor& transpose();

    tensor matMul(tensor<t>& other);

    void requiresGrad(bool val) {
        isGradEnabled = val;
    }

    bool requiresGrad() const {
        return isGradEnabled;
    }

    void backward() {
        if (gradFunction) gradFunction->backward();
    }

    tensor sum();

    tensor mean() {
        tensor<t> out = sum();
        out.tens[0] /= storageLength;
        if (isGradEnabled) {
            delete out.gradFunction;
            out.gradFunction = new meanNode<t>(this);
        }
        return out;
    }
};