#include <iostream>
#include "tensor/tensor.hpp"

int main() {

    std::cout << "Creating Matrix A (2x3)...\n";

    tensor<float> A(2, 3);

    A(0,0) = 1;
    A(0,1) = 2;
    A(0,2) = 3;

    A(1,0) = 4;
    A(1,1) = 5;
    A(1,2) = 6;

    A.print();

    std::cout << "\nCreating Matrix B (3x2)...\n";

    tensor<float> B(3, 2);

    B(0,0) = 7;
    B(0,1) = 8;

    B(1,0) = 9;
    B(1,1) = 10;

    B(2,0) = 11;
    B(2,1) = 12;

    B.print();

    std::cout << "\nTesting Matrix Multiplication\n";

    tensor<float> C = A.matMul(B);

    C.toCPU();

    std::cout << "\nResult Matrix C = A x B\n";

    C.print();

    /*
        Expected Output:

        58   64
        139 154
    */

    std::cout << "\nTesting Identity Matrix...\n";

    tensor<float> I(2,2);

    I(0,0) = 1;
    I(0,1) = 0;

    I(1,0) = 0;
    I(1,1) = 1;

    tensor<float> D = C.matMul(I);

    D.toCPU();

    std::cout << "\nC x I:\n";

    D.print();

    std::cout << "\nTesting invalid dimensions...\n";

    try {

        tensor<float> X(2,3);
        tensor<float> Y(2,4);

        tensor<float> Z = X.matMul(Y);

    } catch (const std::exception& e) {

        std::cout << "Caught exception:\n";
        std::cout << e.what() << '\n';
    }

    std::cout << "\nTesting large matrix multiplication (512x512)...\n";

    tensor<float> L1(512,512);
    tensor<float> L2(512,512);

    L1.random();
    L2.random();

    auto L3 = L1.matMul(L2);

    std::cout << "Large matrix multiplication completed successfully.\n";

    std::cout << "\nAll tests passed.\n";

    return 0;
}