#include <iostream>
#include "tensor/tensor.hpp"

int main() {

    try {

        std::cout << "Creating tensor A (2x3)...\n";

        tensor<float> A(2, 3);

        A(0,0) = 1;
        A(0,1) = 2;
        A(0,2) = 3;
        A(1,0) = 4;
        A(1,1) = 5;
        A(1,2) = 6;

        std::cout << "\nTensor A:\n";
        A.print();

        //-------------------------------------------------

        std::cout << "\nMoving A to GPU...\n";
        A.toGPU();

        //-------------------------------------------------

        std::cout << "\nCreating tensor B (2x3)...\n";

        tensor<float> B(2,3);
        B.fill(10);

        std::cout << "\nTensor B:\n";
        B.toCPU();
        B.print();

        //-------------------------------------------------

        std::cout << "\nTesting operator+\n";

        tensor<float> C = A + B;

        C.toCPU();

        std::cout << "\nTensor C = A + B:\n";
        C.print();

        //-------------------------------------------------

        std::cout << "\nTesting operator+=\n";

        A += B;

        A.toCPU();

        std::cout << "\nTensor A after A += B:\n";
        A.print();

        //-------------------------------------------------

        std::cout << "\nTesting random()\n";

        tensor<float> D(3,3);

        D.random();
        D.toCPU();

        D.print();

        //-------------------------------------------------

        std::cout << "\nTesting reshape()\n";

        D.reshape(1,9);

        D.print();

        //-------------------------------------------------

        std::cout << "\nTesting transposed()\n";

        tensor<float> E = C.transposed();

        E.toCPU();

        E.print();

        //-------------------------------------------------

        std::cout << "\nTesting in-place transpose()\n";

        C.transpose();

        C.toCPU();

        C.print();

        //-------------------------------------------------

        std::cout << "\nTesting copy constructor\n";

        tensor<float> F(C);

        F.toCPU();

        F.print();

        //-------------------------------------------------

        std::cout << "\nTesting move constructor\n";

        tensor<float> G(std::move(F));

        G.toCPU();

        G.print();

        //-------------------------------------------------

        std::cout << "\nAll tests completed successfully.\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
    }

    return 0;
}