#include <iostream>
#include "tensor/tensor.hpp"

int main() {

    try {

        // ~8 GB tensor
        // 2,000,000,000 floats × 4 bytes ≈ 8 GB

        std::cout << "Allocating huge tensor on CPU...\n";

        tensor<float> A(2000000000LL);

        std::cout << "Allocation successful.\n";

        float* ptr = A.data();

        std::cout << "Filling tensor...\n";

        for (long long i = 0; i < 2000000000LL; i++) {
            ptr[i] = static_cast<float>(i % 100);
        }

        std::cout << "CPU fill complete.\n";

        std::cout << "Moving tensor to GPU...\n";

        A.toGPU();

        std::cout << "Transfer complete.\n";

        std::cout << "Moving tensor back to CPU...\n";

        A.toCPU();

        std::cout << "Transfer back complete.\n";

        std::cout << "First element: " << A.data()[0] << '\n';
        std::cout << "Last element: " << A.data()[1999999999LL] << '\n';

    }
    catch (...) {
        std::cout << "Exception caught.\n";
    }

    return 0;
}