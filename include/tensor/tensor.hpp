#include <memory>
#include <vector>
#include <concepts>

template <typename t>
class tensor {
    std::vector<size_t> shape;
    size_t storageLength = 1;

public:
    std::unique_ptr<t[]> tens;
    template <typename ... Args>
    requires (std::integral<Args> && ...)
    tensor(Args...args) : shape({args...}) {

        for (auto& i : shape) {
            storageLength*=i;
        }

        tens = std::make_unique(new t[storageLength]);
    }
};