#pragma once

#include "tensor/tensor.hpp"
#include <thread>

template <typename t>
class module {
public:
    virtual ~module() = default;

    virtual tensor<t> forward(const tensor<t>& input) = 0;
    virtual tensor<t> forward(tensor<t>&& input) = 0;

    virtual std::vector<tensor<t>*> parameters() = 0;
};

template <typename t>
class linear : public module<t> {
    tensor<t> weights;
    tensor<t> bias;
public:
    linear(device dev, size_t neurons, size_t inputs) : bias(dev, 1, neurons), weights(dev, inputs, neurons) {
        weights.requiresGrad(true);
        bias.requiresGrad(true);
        weights.random();
        bias.random();
    }
    tensor<t> forward(const tensor<t>& input) override {
        assert(input.getShape()[1] == weights.getShape()[0]);
        return input.matMul(weights) + bias.batch(input.getShape()[0]);
    }
    tensor<t> forward(tensor<t>&& input) override {
        assert(input.getShape()[1] == weights.getShape()[0]);
        return std::move(input).matMul(weights) + bias.batch(input.getShape()[0]);
    }
    std::vector<tensor<t>*> parameters() override {
        return std::vector<tensor<t>*>({&weights, &bias});
    }
};

template <typename t>
class relu : public module<t> {
public:
    tensor<t> forward(const tensor<t>& input) override {
        return input.ReLU();
    }
    tensor<t> forward(tensor<t>&& input) override {
        return std::move(input).ReLU();
    }
    std::vector<tensor<t>*> parameters() override {
        return std::vector<tensor<t>*>({});
    }
};

template <typename t>
class gelu : public module<t> {
public:
    tensor<t> forward(const tensor<t>& input) override {
        return input.gelu();
    }
    tensor<t> forward(tensor<t>&& input) override {
        return std::move(input).gelu();
    }
    std::vector<tensor<t>*> parameters() override {
        return std::vector<tensor<t>*>({});
    }
};

template <typename t>
class softmax : public module<t> {
public:
    tensor<t> forward(const tensor<t>& input) override {
        return input.softmax();
    }
    tensor<t> forward(tensor<t>&& input) override {
        return std::move(input).softmax();
    }
    std::vector<tensor<t>*> parameters() override {
        return std::vector<tensor<t>*>({});
    }
};

template <typename t>
class sigmoid : public module<t> {
public:
    tensor<t> forward(const tensor<t>& input) override {
        return input.sigmoid();
    }
    tensor<t> forward(tensor<t>&& input) override {
        return std::move(input).sigmoid();
    }
    std::vector<tensor<t>*> parameters() override {
        return std::vector<tensor<t>*>({});
    }
};

template <typename t>
class layernorm : public module<t> {
    tensor<t> gamma;
    tensor<t> beta;
    tensor<t> epsilon;
public:
    layernorm(device dev, size_t inputs, t eps = t(1e-5)) : gamma(dev, 1, inputs), beta(dev, 1, inputs), epsilon(dev, 1, 1) {
        gamma.ones();
        beta.zeros();
        epsilon.fill(eps);

        gamma.requiresGrad(true);
        beta.requiresGrad(true);
        epsilon.requiresGrad(false);
    }

    tensor<t> forward(const tensor<t>& input) override {
        epsilon.requiresGrad(false);
        input.requiresGrad(false);
        auto mean = input.rowSum() / input.getShape()[1];
        auto centered = input - mean.batch(input.getShape()[1], 1);
        auto var = centered.pow(2).rowSum() / input.getShape()[1];
        auto std = (var + epsilon.batch(var.getShape()[0])).batch(centered.getShape()[1], 1).pow(-0.5);
        auto norm = centered * std;
        gamma.requiresGrad(false);
        beta.requiresGrad(false);
        auto out = norm * gamma.batch(norm.getShape()[0]) + beta.batch(norm.getShape()[0]);
        gamma.requiresGrad(true);
        beta.requiresGrad(true);
        input.requiresGrad(true);
        out.requiresGrad(true);
        out.setGradientFunction(std::make_shared<layerNormNode<t>>(&gamma, &beta, std::make_shared<tensor<t>>(std::move(norm)), std::make_shared<tensor<t>>(std::move(std)), &input));
        return out;
    }
    tensor<t> forward(tensor<t>&& input) override {
        epsilon.requiresGrad(false);
        input.requiresGrad(false);
        auto mean = input.rowSum() / input.getShape()[1];
        auto centered = input - mean.batch(input.getShape()[1], 1);
        auto var = centered.pow(2).rowSum() / input.getShape()[1];
        auto std = (var + epsilon.batch(var.getShape()[0])).batch(centered.getShape()[1], 1).pow(-0.5);
        auto norm = centered * std;
        gamma.requiresGrad(false);
        beta.requiresGrad(false);
        auto out = norm * gamma.batch(norm.getShape()[0]) + beta.batch(norm.getShape()[0]);
        gamma.requiresGrad(true);
        beta.requiresGrad(true);
        input.requiresGrad(true);
        out.requiresGrad(true);
        std::shared_ptr<tensor<t>> first = std::make_shared<tensor<t>>(std::move(input));
        out.setGradientFunction(std::make_shared<layerNormNode<t>>(&gamma, &beta, std::make_shared<tensor<t>>(std::move(norm)), std::make_shared<tensor<t>>(std::move(std)), first));
        return out;
    }
    std::vector<tensor<t>*> parameters() override {
        return std::vector<tensor<t>*>({&gamma, &beta});
    }    
};

template <typename t>
class tokenEmbedding {
    tensor<t> weight;

public:
    tokenEmbedding(device dev, size_t vocabSize, size_t embeddingDim) : weight(dev, vocabSize, embeddingDim) {
        weight.random();
        weight.requiresGrad(true);
    }  
    tensor<t> forward(const size_t* input, size_t len);
    tensor<t>* parameters() {
        return &weight;
    }
};

template <typename t>
class positionEmbedding {
    tensor<t> weight;

public:
    positionEmbedding(device dev, size_t vocabSize, size_t embeddingDim) : weight(dev, vocabSize, embeddingDim) {
        weight.random();
        weight.requiresGrad(true);
    }  
    tensor<t> forward(size_t len);
    tensor<t>* parameters() {
        return &weight;
    }
};

template <typename t>
class singleHeadAttention : public module<t> {
public:
    tensor<t> forward(const tensor<t>& input) override {
    }
    tensor<t> forward(tensor<t>&& input) override {
    }
    std::vector<tensor<t>*> parameters() override {
        return std::vector<tensor<t>*>({});
    }
};

template <typename t>
class sequential : public module<t> {
    std::vector<std::unique_ptr<module<t>>> modules; 
public:
    template <typename...Args>
    requires (std::derived_from<std::decay_t<Args>, module<t>> && ...)
    sequential(Args&&...args) {
        (
            modules.push_back(std::make_unique<std::decay_t<Args>>(std::forward<Args>(args))),
            ...
        );
    }

    tensor<t> forward(const tensor<t>& input) override {
        tensor<t> temp = input;
        for (auto& i : modules) {
            tensor<t> temp2 = i -> forward(std::move(temp));
            tensor<t> tempdel = std::move(temp);
            temp = temp2;
        }

        return temp;
    }

    tensor<t> forward(tensor<t>&& input) override {
        tensor<t> temp = std::move(input);
        for (auto& i : modules) {
            tensor<t> temp2 = i -> forward(std::move(temp));
            tensor<t> tempdel = std::move(temp);
            temp = temp2;
        }

        return temp;
    }
    std::vector<tensor<t>*> parameters() override {
        std::vector<tensor<t>*> out;

        for (auto& i : modules) {
            std::vector<tensor<t>*> temp = i -> parameters();
            for (auto& j : temp) out.push_back(j);
        }
        return out;
    }
};

