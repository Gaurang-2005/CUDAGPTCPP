#include <memory>
#include <vector>
#include <concepts>

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

    void toGPU();
    void toCPU();

    tensor(const tensor& other);
    tensor(tensor&& other) noexcept;
    tensor& operator=(const tensor& other);
    tensor& operator=(tensor&& other) noexcept;
    ~tensor(); 
};