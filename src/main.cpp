#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>

#include "tensor/tensor.hpp"

template<typename T>
bool verifyMatMul(const tensor<T>& A,
                  const tensor<T>& B,
                  const tensor<T>& C,
                  T eps = static_cast<T>(1e-3))
{
    tensor<T> Ac = A;
    tensor<T> Bc = B;
    tensor<T> Cc = C;

    Ac.toCPU();
    Bc.toCPU();
    Cc.toCPU();

    const T* a = Ac.data();
    const T* b = Bc.data();
    const T* c = Cc.data();

    size_t M = Ac.getShape()[0];
    size_t K = Ac.getShape()[1];
    size_t N = Bc.getShape()[1];

    for (size_t row = 0; row < M; row++)
    {
        for (size_t col = 0; col < N; col++)
        {
            T expected = 0;

            for (size_t k = 0; k < K; k++)
            {
                expected +=
                    a[row * K + k] *
                    b[k * N + col];
            }

            T actual = c[row * N + col];

            if (std::abs(expected - actual) > eps)
            {
                std::cout << "\nVerification FAILED\n";
                std::cout << "Mismatch at (" << row
                          << ", " << col << ")\n";

                std::cout << std::setprecision(8);
                std::cout << "Expected : " << expected << '\n';
                std::cout << "Actual   : " << actual << '\n';

                return false;
            }
        }
    }

    return true;
}

template<typename T>
void benchmarkMatMul(size_t M,
                     size_t K,
                     size_t N,
                     bool verify)
{
    std::cout << "\n=========================================\n";
    std::cout << "Benchmarking "
              << M << " x " << K
              << " * "
              << K << " x " << N
              << '\n';
    std::cout << "=========================================\n";

    tensor<T> A(M, K);
    tensor<T> B(K, N);

    std::cout << "Initializing matrices...\n";

    A.random();
    B.random();

    std::cout << "Warmup run...\n";

    {
        tensor<T> warmup = A.matMul(B);
    }

    std::cout << "Benchmarking...\n";

    auto start = std::chrono::high_resolution_clock::now();

    tensor<T> C = A.matMul(B);

    auto end = std::chrono::high_resolution_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(K) *
        static_cast<double>(N);

    double gflops = flops / (seconds * 1e9);

    std::cout << "\nResults:\n";
    std::cout << "Execution Time : "
              << seconds * 1000.0
              << " ms\n";

    std::cout << "Performance    : "
              << gflops
              << " GFLOPS\n";

    if (verify)
    {
        std::cout << "\nVerifying result...\n";

        bool ok = verifyMatMul(A, B, C);

        std::cout << "Verification   : "
                  << (ok ? "PASSED ✅" : "FAILED ❌")
                  << '\n';
    }
    else
    {
        std::cout << "\nVerification   : Skipped\n";
    }
}

// int main()
// {
//     std::cout << "=========================================\n";
//     std::cout << " CUDA-GPT Matrix Multiplication Test Suite\n";
//     std::cout << "=========================================\n";

//     std::cout << "\n========== Correctness Tests ==========\n";

//     // Small square matrices
//     benchmarkMatMul<float>(2, 2, 2, true);
//     benchmarkMatMul<float>(3, 3, 3, true);
//     benchmarkMatMul<float>(7, 7, 7, true);

//     // Tile boundary tests
//     benchmarkMatMul<float>(15, 15, 15, true);
//     benchmarkMatMul<float>(16, 16, 16, true);
//     benchmarkMatMul<float>(17, 17, 17, true);

//     benchmarkMatMul<float>(31, 31, 31, true);
//     benchmarkMatMul<float>(32, 32, 32, true);
//     benchmarkMatMul<float>(33, 33, 33, true);

//     // Rectangular matrices
//     benchmarkMatMul<float>(8, 16, 4, true);
//     benchmarkMatMul<float>(16, 8, 32, true);
//     benchmarkMatMul<float>(17, 19, 23, true);
//     benchmarkMatMul<float>(31, 47, 29, true);
//     benchmarkMatMul<float>(64, 31, 17, true);
//     benchmarkMatMul<float>(127, 65, 33, true);
//     benchmarkMatMul<float>(129, 128, 131, true);
//     benchmarkMatMul<float>(257, 129, 65, true);

//     std::cout << "\n========== Performance Benchmarks ==========\n";

//     benchmarkMatMul<float>(256, 256, 256, true);
//     benchmarkMatMul<float>(512, 512, 512, false);
//     benchmarkMatMul<float>(1024, 1024, 1024, false);
//     benchmarkMatMul<float>(2048, 2048, 2048, false);
//     benchmarkMatMul<float>(10000, 10000, 10000, false);

//     std::cout << "\n=========================================\n";
//     std::cout << "All tests completed successfully.\n";
//     std::cout << "=========================================\n";

//     return 0;
// }

double sumCheck(double* data, size_t storageLength) {
    double sum = 0;
    for (int i = 0; i < storageLength; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    tensor<double> test(4096, 4096);
    // for (int i = 1; i <= test.numelements(); i++) {
    //     test.data()[i-1] = i;
    // }
    test.ones();
    tensor<double> sumGPU = test.sum();
    test.toCPU();
    double sumCPU = sumCheck(test.data(), test.numelements());

    std::cout << "sumCPU: " << sumCPU << " and sumGPU: " << sumGPU.data()[0] << std::endl;
}