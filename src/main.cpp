#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>

#include "tensor/tensor.hpp"
#include "loss/loss.hpp"

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

void separator(const std::string& title) {
    std::cout << "\n========================================\n";
    std::cout << title << '\n';
    std::cout << "========================================\n";
}


// using std::cout;
// using std::endl;

// int main() {

//     cout << "\n========================================\n";
//     cout << "TEST 1 : ADD -> EXP -> SUM\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         tensor<float> B(device::GPU,2,2);
//         cout<<"passed\n";

//         A.ones();
//         B.ones();
//         cout<<"passed\n";
//         A.requiresGrad(true);
//         B.requiresGrad(true);
//         cout<<"passed\n";
//         auto out = (A + B).exp().sum();
//         cout<<"passed\n";
//         out.backward();
//         cout<<"passed\n";
//         cout << "A.grad\n";
//         A.gradient()->print();

//         cout << "B.grad\n";
//         B.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 2 : LOG -> POW -> MEAN\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         A.fill(2);

//         A.requiresGrad(true);

//         auto out = A.log().pow(2).mean();
//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 3 : GELU -> SIGMOID -> TANH -> SUM\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         A.fill(0.5f);

//         A.requiresGrad(true);

//         auto out = A.gelu().sigmoid().tanh().sum();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 4 : RELU NEGATIVE INPUT\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         A.fill(-2);

//         A.requiresGrad(true);

//         auto out = A.ReLU().sum();
//         out.backward();

//         cout << "Expected gradient = 0\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 5 : SIGMOID(0)\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         A.zeros();

//         A.requiresGrad(true);

//         auto out = A.sigmoid().sum();
//         out.backward();

//         cout << "Expected ~= 0.25\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 6 : TANH(0)\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         A.zeros();

//         A.requiresGrad(true);

//         auto out = A.tanh().sum();
//         out.backward();

//         cout << "Expected = 1\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 7 : EXP(LOG(A))\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         A.fill(3);

//         A.requiresGrad(true);

//         auto out = A.log().exp().sum();

//         out.backward();

//         cout << "Expected = 1\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 8 : POW(3)\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         A.fill(2);

//         A.requiresGrad(true);

//         auto out = A.pow(3).sum();

//         out.backward();

//         cout << "Expected = 12\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 9 : RESHAPE -> TANH -> MEAN\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         A.fill(1);

//         A.requiresGrad(true);

//         auto out = A.reshaped(1,4).tanh().mean();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 10 : TRANSPOSE -> MATMUL -> SUM\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         tensor<float> B(device::GPU,2,2);

//         A.ones();
//         B.ones();

//         A.requiresGrad(true);
//         B.requiresGrad(true);

//         auto out = A.transposed().matMul(B).sum();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();

//         cout << "B.grad\n";
//         B.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 11 : DIAMOND GRAPH\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.fill(2);
//         A.requiresGrad(true);

//         auto B = A.exp();
//         auto C = A.log();

//         auto out = (B + C).sum();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 12 : TRIPLE BRANCH\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.fill(2);

//         A.requiresGrad(true);

//         auto out =
//             (
//                 A.exp()
//                 +
//                 A.pow(2)
//                 +
//                 A.log()
//             ).mean();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 13 : VERY DEEP CHAIN\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.fill(1.5f);
//         A.requiresGrad(true);

//         auto out =
//             A.exp()
//             .log()
//             .pow(2)
//             .gelu()
//             .sigmoid()
//             .tanh()
//             .pow(2)
//             .exp()
//             .log()
//             .mean();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 14 : HUGE BRANCH GRAPH\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.fill(1);

//         A.requiresGrad(true);

//         auto out =
//             (
//                 A.exp()
//                 +
//                 A.sigmoid()
//                 +
//                 A.tanh()
//                 +
//                 A.gelu()
//                 +
//                 A.pow(2)
//                 +
//                 A.log()
//             ).sum();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 15 : MATMUL CHAIN\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);
//         tensor<float> B(device::GPU,2,2);
//         tensor<float> C(device::GPU,2,2);

//         A.ones();
//         B.ones();
//         C.ones();

//         A.requiresGrad(true);
//         B.requiresGrad(true);
//         C.requiresGrad(true);

//         auto out =
//             A.matMul(B)
//              .matMul(C)
//              .gelu()
//              .mean();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();

//         cout << "B.grad\n";
//         B.gradient()->print();

//         cout << "C.grad\n";
//         C.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 16 : MULTIPLE RESHAPES\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.fill(2);

//         A.requiresGrad(true);

//         auto out =
//             A.reshaped(1,4)
//              .reshaped(2,2)
//              .reshaped(4,1)
//              .mean();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 17 : DOUBLE TRANSPOSE\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.fill(3);

//         A.requiresGrad(true);

//         auto out =
//             A.transposed()
//              .transposed()
//              .sum();

//         out.backward();

//         cout << "Expected = 1\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 18 : LONG EXP LOG CHAIN\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.fill(2);

//         A.requiresGrad(true);

//         auto out =
//             A.exp()
//              .log()
//              .exp()
//              .log()
//              .exp()
//              .log()
//              .exp()
//              .log()
//              .sum();

//         out.backward();

//         cout << "Expected = 1\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 19 : ACTIVATION CASCADE\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.random();

//         A.requiresGrad(true);

//         auto out =
//             A.ReLU()
//              .sigmoid()
//              .tanh()
//              .gelu()
//              .sigmoid()
//              .mean();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 20 : MASSIVE ACCUMULATION\n";
//     cout << "========================================\n";

//     {
//         cout <<"passed\n";
//         tensor<float> A(device::GPU,2,2);
//         cout <<"passed\n";
//         A.ones();

//         A.requiresGrad(true);
//         cout <<"passed\n";
//         auto out =
//             A+A+A+A+A+A+A+A+A+A;
//         cout << "passed\n";
//         tensor<float> fresh = out.sum();
//         cout << "passed\n";
//         fresh.backward();

//         cout << "Expected = 10\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 21 : DEEP BINARY TREE\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.fill(2);

//         A.requiresGrad(true);

//         auto B=A+A;
//         auto C=B+B;
//         auto D=C+C;
//         auto E=D+D;
//         auto F=E+E;

//         auto out=F.sum();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 22 : RANDOM VALUES\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,2,2);

//         A.random();

//         A.requiresGrad(true);

//         auto out =
//             (
//                 A.exp()
//                 *
//                 A.sigmoid()
//                 /
//                 A.gelu()
//             ).mean();

//         out.backward();

//         cout << "A.grad\n";
//         A.gradient()->print();
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 23 : LARGE TENSOR\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,512,512);

//         A.ones();

//         A.requiresGrad(true);

//         auto out =
//             A.exp()
//              .log()
//              .pow(2)
//              .mean();
//         out.backward();

//         cout << "Large tensor completed.\n";
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 24 : REPEATED GRAPH CREATION\n";
//     cout << "========================================\n";

//     {
//         for(int i=0;i<1000;i++)
//         {
//             tensor<float> A(device::GPU,2,2);

//             A.fill(2);

//             A.requiresGrad(true);

//             auto out =
//                 A.exp()
//                  .gelu()
//                  .pow(2)
//                  .sigmoid()
//                  .mean();
//             out.backward();
//         }

//         cout << "Completed 1000 graphs.\n";
//     }

//     cout << "\n========================================\n";
//     cout << "TEST 25 : GPU MEMORY STRESS\n";
//     cout << "========================================\n";

//     // {
//     //     for(int i=0;i<200;i++)
//     //     {
//     //         tensor<float> A(device::GPU,1024,1024);

//     //         A.random();

//     //         A.requiresGrad(true);

//     //         auto out =
//     //             (
//     //                 A.exp()
//     //                  .gelu()
//     //                  .pow(2)
//     //                  .sigmoid()
//     //                  .tanh()
//     //             ).mean();

//     //         out.backward();

//     //         if(i%20==0)
//     //             cout << "Iteration " << i << endl;
//     //     }

//     //     cout << "Memory stress completed.\n";
//     // }
//     cout << "\n========================================\n";
//     cout << "TEST : SOFTMAX FORWARD\n";
//     cout << "========================================\n";
//     {
//         tensor<float> A(device::GPU, 1, 3);

//         A.toCPU();
//         A(0,0) = 1;
//         A(0,1) = 2;
//         A(0,2) = 3;
//         A.toGPU();

//         auto out = A.softmax();

//         cout << "Expected approximately:\n";
//         cout << "0.0900306 0.244728 0.665241\n";

//         out.print();
//         cout << "\n========================================\n";
//         cout << "ALL STRESS TESTS FINISHED\n";
//         cout << "========================================\n";
//     }
//     cout << "\n========================================\n";
//     cout << "TEST : SOFTMAX BACKWARD\n";
//     cout << "========================================\n";
//     {
//         tensor<float> A(device::GPU,1,3);

//         A.toCPU();

//         A(0,0)=1;
//         A(0,1)=2;
//         A(0,2)=3;

//         A.toGPU();

//         A.requiresGrad(true);

//         auto out = A.softmax();

//         auto out2 = out.sum();

//         out2.backward();

//         cout << "Expected:\n";
//         cout << "0 0 0\n";

//         A.gradient()->print();
//     }
//     cout << "\n========================================\n";
//     cout << "TEST : RANDOM SOFTMAX\n";
//     cout << "========================================\n";

//     {
//         tensor<float> A(device::GPU,64,128);

//         A.random();

//         auto out = A.softmax();

//         auto sums = out.rowSum();

//         cout << "Every value should be close to 1\n";

//         sums.print();

//     }
//     {
//         tensor<float> A(device::GPU,1,3);

//         A.toCPU();
//         A(0,0)=100;
//         A(0,1)=101;
//         A(0,2)=102;
//         A.toGPU();

//         A.softmax().print();

//     }
//     {
//         tensor<float> A(device::GPU, 1, 3);

//         A.toCPU();
//         A(0,0) = 1;
//         A(0,1) = 2;
//         A(0,2) = 3;
//         A.toGPU();

//         A.requiresGrad(true);

//         auto out = A.softmax();
//         auto out2 = out.pow(2).sum();

//         out2.backward();

//         A.gradient()->print();
//     }
//     std::cout << "\n========================================\n";
//     std::cout << "TEST : CROSS ENTROPY\n";
//     std::cout << "========================================\n";

//     {
//         tensor<float> pred(device::CPU, 1, 3);

//         pred.data()[0] = 0.1f;
//         pred.data()[1] = 0.8f;
//         pred.data()[2] = 0.1f;

//         pred.toGPU();

//         pred.requiresGrad(true);

//         tensor<float> target(device::CPU, 1, 3);

//         target.data()[0] = 0.f;
//         target.data()[1] = 1.f;
//         target.data()[2] = 0.f;

//         target.toGPU();

//         auto loss = crossEntropyLoss(pred, target);

//         std::cout << "Expected Loss ~= 0.22314355\n";
//         loss.print();
        
//         loss.backward();

//         std::cout << "Expected Gradient:\n";
//         std::cout << "0 -1.25 0\n";

//         pred.gradient()->print();
//     }

//     return 0;
// }

#include "nn/module.hpp"
#include "nn/optimizer.hpp"
#include "loss/loss.hpp"

// int main() {
//     sequential<double> model(
//         linear<double>(device::GPU, 4, 2),
//         relu<double>(),
//         linear<double>(device::GPU, 1, 4)
//     );

//     tensor<double> input(device::CPU, 4, 2);
//     tensor<double> target(device::CPU, 4, 1);
//     input(0, 0) = 0;
//     input(0, 1) = 0;
//     input(1, 0) = 1;
//     input(1, 1) = 0;
//     input(2, 0) = 0;
//     input(2, 1) = 1;
//     input(3, 0) = 1;
//     input(3, 1) = 1;
//     target(0, 0) = 0;
//     target(1, 0) = 1;
//     target(2, 0) = 1;
//     target(3, 0) = 0;
//     input.toGPU();
//     target.toGPU();
//     tensor<double> test(device::GPU, 2 ,1);
//     SGD<double> opti(model.parameters(), 0.0001);
//     for (int i = 0; i < 1000; i++) {
//         if (!(i%10))std::cout<<"Iteration: " << i << std::endl;
//         auto out = model.forward(input);
//         auto loss = crossEntropyLoss(out, target);
//         if (!i) {
//             input.print();
//             out.print();
//             loss.print();
//         }
//         loss.backward();
//         opti.step();
//         opti.zeroGrad();
//         if (!(i % 100)){
//             out.gradient()->print();
//             input.print();
//             out.print();
//         }
//     }
// }


// int main()
// {
//     layernorm<float> ln(device::CPU, 4);

//     tensor<float> x(device::CPU, 2, 4);

//     x(0,0)=1.0f;
//     x(0,1)=2.0f;
//     x(0,2)=3.0f;
//     x(0,3)=4.0f;

//     x(1,0)=5.0f;
//     x(1,1)=6.0f;
//     x(1,2)=7.0f;
//     x(1,3)=8.0f;

//     x.requiresGrad(true);

//     auto y = ln.forward(x);

//     std::cout << "Forward:\n";
//     y.print();

//     tensor<float> target(device::CPU,2,4);
//     target.fill(0.0f);

//     tensor<float> loss = MSE(y, target);

//     std::cout << "Loss:\n";
//     loss.print();

//     loss.backward();

//     std::cout << "\nInput Gradient\n";
//     x.gradient()->print();

//     std::cout << "\nGamma Gradient\n";
//     ln.parameters()[0]->gradient()->print();

//     std::cout << "\nBeta Gradient\n";
//     ln.parameters()[1]->gradient()->print();
// }

int main()
{
    embedding<float> emb(device::GPU, 4, 3);

    // Weight matrix
    emb.parameters()[0]->print();
    std::cout<<"passed! "<< std::endl;
    size_t tokens[] =
    {
        2,
        0,
        2
    };
    std::cout<<"passed! "<< std::endl;
    auto out = emb.forward(tokens, sizeof(tokens)/sizeof(tokens[0]));
    std::cout<<"passed! "<< std::endl;
    std::cout << "Forward:\n";
    out.print();

    // Give output a fake gradient
    out.setGradient(new tensor<float>(
        device::GPU,
        {
            {1,2,3},
            {4,5,6},
            {7,8,9}
        }));

    out.backward();

    std::cout << "\nWeight Gradient:\n";
    emb.parameters()[0]->gradient()->print();

    return 0;
}