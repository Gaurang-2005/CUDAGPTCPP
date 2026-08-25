#include "loss/loss.hpp"

template tensor<float> crossEntropyLoss(
    const tensor<float>&,
    const std::vector<TokenID>&
);

template tensor<double> crossEntropyLoss(
    const tensor<double>&,
    const std::vector<TokenID>&
);

template tensor<__half> crossEntropyLoss(
    const tensor<__half>&,
    const std::vector<TokenID>&
);

template tensor<__nv_bfloat16> crossEntropyLoss(
    const tensor<__nv_bfloat16>&,
    const std::vector<TokenID>&
);

template tensor<float> crossEntropyLoss(
    const tensor<float>&,
    const std::vector<std::vector<TokenID>>&
);

template tensor<double> crossEntropyLoss(
    const tensor<double>&,
    const std::vector<std::vector<TokenID>>&
);

template tensor<__half> crossEntropyLoss(
    const tensor<__half>&,
    const std::vector<std::vector<TokenID>>&
);

template tensor<__nv_bfloat16> crossEntropyLoss(
    const tensor<__nv_bfloat16>&,
    const std::vector<std::vector<TokenID>>&
);

template <typename t>
__global__ void crossEntropyLossLLMKernel(t* out, const t* logits, const TokenID* targ, const size_t targLength, const size_t vocabLen) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= targLength) return;

    out[idx] = logits[idx * vocabLen + targ[idx]];
}

template <typename t>
__global__ void crossEntropyLossLLM3DKernel(t* out, const t* logits, const TokenID* targ, const size_t targLength, const size_t vocabLen) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= targLength) return;

    out[blockIdx.y * targLength + idx] = logits[blockIdx.y * targLength * vocabLen + idx * vocabLen + targ[blockIdx.y * targLength + idx]];
}

template <typename t>
tensor<t> crossEntropyLoss(const tensor<t>& logits, const std::vector<TokenID>& target) {
    if (logits.getShape()[0] != target.size()) throw std::invalid_argument("Prediction and target should have same batch size!");
    if (!logits.requiresGrad()) throw std::invalid_argument("LLM needs ");
    logits.requiresGrad(false);
    auto maxes = logits.rowMax();
    logits.requiresGrad(true);
    auto maxes2 = maxes;

    auto shifted = logits - std::move(maxes).batch(logits.getShape()[1], 1);      

    auto sumExp = std::move(shifted).exp().rowSum();

    auto logSumExp = std::move(sumExp).log() + std::move(maxes2);
    TokenID* temp;
    cudaError_t err = cudaMalloc(&temp, target.size() * sizeof(TokenID));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    err = cudaMemcpy(temp, target.data(), target.size() * sizeof(TokenID), cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    tensor<t> out(device::GPU, target.size(), 1);
    crossEntropyLossLLMKernel<<<(target.size() + 255) / 256, 256>>>(out.data(), logits.data(), temp, target.size(), logits.getShape()[1]);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    cudaFree(temp);
    out.requiresGrad(true);
    out.setGradientFunction(std::make_shared<gatherNode<t>>(&logits, &target));
    return (std::move(logSumExp) - std::move(out)).mean();
}

template <typename t>
tensor<t> crossEntropyLoss(const tensor<t>& logits, const std::vector<std::vector<TokenID>>& target) {
    if (logits.getShape()[0] != target.size()) throw std::invalid_argument("Prediction and target should have same batch size!");
    logits.requiresGrad(false);
    auto maxes = logits.rowMax();
    logits.requiresGrad(true);
    auto maxes2 = maxes;
    auto shifted = logits - std::move(maxes).batch(logits.getShape()[2], 1);     
    //shifted.print(); 
    auto sumExp = std::move(shifted).exp().rowSum();
    auto logSumExp = std::move(sumExp).log() + std::move(maxes2);
    tensor<t> out(device::GPU, logits.getShape()[0], logits.getShape()[1], 1);
    std::vector<TokenID> targComb;
    targComb.reserve(target.size() * target[0].size());
    for (auto& i : target) {
        for (auto& j : i) {
            targComb.push_back(j);
        }
    }
    TokenID* temp;
    cudaError_t err = cudaMalloc(&temp, targComb.size() * sizeof(TokenID));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                << cudaGetErrorString(err)
                << '\n';
    }
    err = cudaMemcpy(temp, targComb.data(), targComb.size() * sizeof(TokenID), cudaMemcpyDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    crossEntropyLossLLM3DKernel<<<dim3((target[0].size() + 255) / 256, target.size()), dim3(256, 1)>>>(out.data(), logits.data(), temp, target[0].size(), logits.getShape()[2]);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                << cudaGetErrorString(err)
                << '\n';
        std::abort();
    }
    cudaFree(temp);

    out.requiresGrad(true);
    out.setGradientFunction(std::make_shared<gatherNode<t>>(&logits, &target));
    return (std::move(logSumExp) - std::move(out)).mean();
}