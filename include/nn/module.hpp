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
    tensor<t> wQuery;
    tensor<t> wKey;
    tensor<t> wVal;

    tensor<t> scaledDotProductAttention(const tensor<t>& q, const tensor<t>& k, const tensor<t>& v, std::shared_ptr<tensor<t>>& score) const;
public:
    singleHeadAttention(device dev, size_t embedDim) : wQuery(dev, embedDim, embedDim), wKey(dev, embedDim, embedDim), wVal(dev, embedDim, embedDim) {
        wQuery.random();
        wKey.random();
        wVal.random();
        wQuery.requiresGrad(true);
        wKey.requiresGrad(true);
        wVal.requiresGrad(true);
    }
    tensor<t> forward(const tensor<t>& input) override {
        wQuery.requiresGrad(false);
        wKey.requiresGrad(false);
        wVal.requiresGrad(false);
        input.requiresGrad(false);
        auto Q = std::make_shared<tensor<t>>(input.matMul(wQuery));
        auto K = std::make_shared<tensor<t>>(input.matMul(wKey));
        auto V = std::make_shared<tensor<t>>(input.matMul(wVal));
        wQuery.requiresGrad(true);
        wKey.requiresGrad(true);
        wVal.requiresGrad(true);
        input.requiresGrad(true);
        std::shared_ptr<tensor<t>> score;
        auto out = scaledDotProductAttention(*Q, *K, *V, score);
        out.requiresGrad(true);
        out.setGradientFunction(std::make_shared<singleHeadAttentionNode<t>>(Q, K, V, &input, &wQuery, &wKey, &wVal, score));
        return out;
    }
    tensor<t> forward(tensor<t>&& input) override {
        wQuery.requiresGrad(false);
        wKey.requiresGrad(false);
        wVal.requiresGrad(false);
        input.requiresGrad(false);
        auto Q = std::make_shared<tensor<t>>(input.matMul(wQuery));
        auto K = std::make_shared<tensor<t>>(input.matMul(wKey));
        auto V = std::make_shared<tensor<t>>(input.matMul(wVal));
        wQuery.requiresGrad(true);
        wKey.requiresGrad(true);
        wVal.requiresGrad(true);
        input.requiresGrad(true);
        std::shared_ptr<tensor<t>> score;
        auto out = scaledDotProductAttention(*Q, *K, *V, score);
        out.requiresGrad(true);
        out.setGradientFunction(std::make_shared<singleHeadAttentionNode<t>>(Q, K, V, std::make_shared<tensor<t>>(std::move(input)), &wQuery, &wKey, &wVal, score));
        return out;
    }
    std::vector<tensor<t>*> parameters() override {
        return std::vector<tensor<t>*>({&wQuery, &wKey, &wVal});
    }
};

template <typename t>
class residual : public module<t> {
     module<t>& branch;
public:
    template <typename Module>
    requires std::derived_from<std::decay_t<Module>, module<t>>
    residual(Module& mod) : branch(mod) {}                                          
    tensor<t> forward(const tensor<t>& input) override {
        return input + branch.forward(input);
    }
    tensor<t> forward(tensor<t>&& input) override {
        auto input2 = input;
        return std::move(input) + branch.forward(std::move(input2));
    }
    std::vector<tensor<t>*> parameters() override {
        return branch.parameters();
    }
};

template <typename t>
class feedForward : public module<t> {
    linear<t> layer1;
    gelu<t> activation;
    linear<t> layer2;
public:
    feedForward(device dev, size_t embedDim) : layer1(dev, embedDim * 4, embedDim), layer2(dev, embedDim, embedDim * 4) {}
    tensor<t> forward(const tensor<t>& input) override {
        return layer2.forward(activation.forward(layer1.forward(input)));
    }
    tensor<t> forward(tensor<t>&& input) override {
        return layer2.forward(activation.forward(layer1.forward(std::move(input))));
    }
    std::vector<tensor<t>*> parameters() override {
        std::vector<tensor<t>*> out;
        for (auto& i : layer1.parameters()) out.push_back(i);
        for (auto& i : layer2.parameters()) out.push_back(i);
        return out;
    }
};

template <typename t>
class transformerBlock : public module<t> {
    layernorm<t> layerNorm1;
    singleHeadAttention<t> attention;
    layernorm<t> layerNorm2;
    feedForward<t> ff;
public:
    transformerBlock(device dev, size_t embedDim) : layerNorm1(dev, embedDim), attention(dev, embedDim), layerNorm2(dev, embedDim), ff(dev, embedDim) {}
    tensor<t> forward(const tensor<t>& input) override {
        auto y = attention.forward(layerNorm1.forward(input)) + input;
        auto y2 = y;
        return ff.forward(layerNorm2.forward(std::move(y))) + std::move(y2) ;
    }
    tensor<t> forward(tensor<t>&& input) override {
        auto input2 = input;
        auto y = attention.forward(layerNorm1.forward(std::move(input))) + std::move(input2);
        auto y2 = y;
        return ff.forward(layerNorm2.forward(std::move(y))) + std::move(y2) ;
    }
    std::vector<tensor<t>*> parameters() override {
        std::vector<tensor<t>*> out;
        for (auto& i : layerNorm1.parameters()) out.push_back(i);
        for (auto& i : attention.parameters()) out.push_back(i);
        for (auto& i : layerNorm2.parameters()) out.push_back(i);
        for (auto& i : ff.parameters()) out.push_back(i);
        return out;
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

