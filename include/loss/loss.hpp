#pragma once

#include "tensor/tensor.hpp"
#include "autograd/node.hpp"

template <typename t>
tensor<t> MSE(const tensor<t>& prediction, const tensor<t>& target) {
    if (prediction.numElements() != target.numElements()) throw std::invalid_argument("Prediction and target should have same size!");
    return (prediction - target).pow(2).mean();
}

template <typename t>
tensor<t> crossEntropyLoss(const tensor<t>& prediction, const tensor<t>& target) {
    if (prediction.numElements() != target.numElements()) throw std::invalid_argument("Prediction and target should have same size!");
    if (prediction.requiresGrad()) {
        prediction.requiresGrad(false);
        tensor<t> out = (target * (-prediction.log())).rowSum().mean();
        prediction.requiresGrad(true);
        out.requiresGrad(true);
        out.setGradientFunction(std::make_shared<crossEntropyLossNode<t>>(&prediction, &target));
        return out;
    }
    return (target * (-prediction.log())).rowSum().mean();
}

template <typename t>
tensor<t> crossEntropyLoss(const tensor<t>& prediction, const tensor<t>&& target) = delete;

template <typename t>
tensor<t> crossEntropyLoss(const tensor<t>&& prediction, const tensor<t>& target) = delete;

template <typename t>
tensor<t> crossEntropyLoss(const tensor<t>&& prediction, const tensor<t>&& target) = delete;
