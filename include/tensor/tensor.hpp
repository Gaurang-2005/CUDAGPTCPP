#include <memory>
#include <vector>
#include <concepts>
#include <cassert>

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
public:
    template <typename ... Args>
    requires (std::integral<Args> && ...)
    tensor(device dev, Args...args) : shape({args...}), dev(dev) {

        for (auto& i : shape) {
            storageLength*=i;
        }
        if (dev == device::CPU)
        tens = new t[storageLength];

        else if (dev == device::GPU) {
            toGPU();
        }
    }
    template <typename ... Args>
    requires (std::integral<Args> && ...)
    tensor(Args...args) : tensor(device::CPU, args...) {}

    t* data() {
        return tens;
    }

    const t* data() const {
        return tens;
    }

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
        constant(0);
    }
    void ones() {
        constant(1);
    }
    void random();
    bool isEmpty() const {
        return storageLength == 0;
    }

    tensor& operator+=(const tensor& other);
    tensor& operator-=(const tensor& other);
    tensor operator+(const tensor& other);
    tensor operator-(const tensor& other);
};