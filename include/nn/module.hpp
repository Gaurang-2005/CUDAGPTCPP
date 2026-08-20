#pragma once

#include "tensor/tensor.hpp"
#include "tokenizer/types.hpp"

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
        if (input.rank() == 2) {
            assert(input.getShape()[1] == weights.getShape()[0]);
            return input.matMul(weights) + bias;
        }
        else if (input.rank() == 3) {
            assert(input.getShape()[2] == weights.getShape()[0]);
            return input.matMul(weights) + bias;
        }
        else {
            throw std::invalid_argument("linear only supports rank-2 and rank-3 tensors");
        }
    }
    tensor<t> forward(tensor<t>&& input) override {
        size_t sh1 = input.getShape()[0];
        size_t sh2 = input.getShape()[1];
        if (input.rank() == 2) {
            assert(input.getShape()[1] == weights.getShape()[0]);
            return std::move(input).matMul(weights) + bias;
        }
        else if (input.rank() == 3) {
            assert(input.getShape()[2] == weights.getShape()[0]);
            return std::move(input).matMul(weights) + bias;
        }
        else {
            throw std::invalid_argument("linear only supports rank-2 and rank-3 tensors");
        }
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
    t epsilon;
public:
    layernorm(device dev, size_t inputs, t eps = t(1e-5)) : gamma(dev, 1, inputs), beta(dev, 1, inputs), epsilon(eps) {
        gamma.ones();
        beta.zeros();

        gamma.requiresGrad(true);
        beta.requiresGrad(true);
    }

    tensor<t> forward(const tensor<t>& input) override {
        tensor<t> out;
        if (input.getShape().size() == 2) {
            tensor<t> centered;
            tensor<t> var;
            if constexpr (std::is_same_v<t, __half>) {
                centered = input - (input.rowSum() / __double2half(input.getShape()[1]));
                auto centcpy = centered;
                var = std::move(centcpy).pow(2).rowSum() / __double2half(input.getShape()[1]);
            }            
            else {
                centered = input - (input.rowSum() / input.getShape()[1]);
                auto centcpy = centered;
                var = std::move(centcpy).pow(2).rowSum() / input.getShape()[1];
            }    
            auto varShape = var.getShape();
            auto std = (std::move(var) + epsilon).pow(-0.5);
            auto norm = std::move(centered) * std::move(std);
            auto normShape = norm.getShape();
            out = std::move(norm) * gamma + beta;
        }
        else if (input.getShape().size() == 3) {
            tensor<t> centered;
            tensor<t> var;
            if constexpr (std::is_same_v<t, __half>) {
                centered = input - (input.rowSum() / __double2half(input.getShape()[2]));
                auto centcpy = centered;
                var = std::move(centcpy).pow(2).rowSum() / __double2half(input.getShape()[2]);
            }            
            else {
                centered = input - (input.rowSum() / input.getShape()[2]);
                auto centcpy = centered;
                var = std::move(centcpy).pow(2).rowSum() / input.getShape()[2];
            }    
            auto varShape = var.getShape();
            auto std = (std::move(var) + epsilon).pow(-0.5);
            auto norm = std::move(centered) * std::move(std);
            auto normShape = norm.getShape();
            out = std::move(norm) * gamma + beta;
        }
        else {
            throw std::invalid_argument("LayerNorm only supports rank-2 and rank-3 tensors.");
        }
        return out;
    }
    tensor<t> forward(tensor<t>&& input) override {
        tensor<t> out;
        if (input.getShape().size() == 2) {
            auto inShape = input.getShape();
            auto input1 = input;
            tensor<t> centered;
            tensor<t> var;
            if constexpr (std::is_same_v<t, __half>) {
                centered = std::move(input1) - (std::move(input).rowSum() / __double2half(inShape[1]));
                auto cent2 = centered;
                var = std::move(cent2).pow(2).rowSum() / __double2half(inShape[1]);
            }            
            else {
                centered = std::move(input1) - (std::move(input).rowSum() / inShape[1]);
                auto cent2 = centered;
                var = std::move(cent2).pow(2).rowSum() / inShape[1];
            }    
            auto varShape = var.getShape();
            auto std = (std::move(var) + epsilon).pow(-0.5);
            auto norm = std::move(centered) * std::move(std);
            auto normShape = norm.getShape();
            out = std::move(norm) * gamma + beta;
        }
        else if (input.getShape().size() == 3) {
            auto inShape = input.getShape();
            auto input1 = input;
            tensor<t> centered;
            tensor<t> var;
            if constexpr (std::is_same_v<t, __half>) {
                centered = std::move(input1) - (std::move(input).rowSum() / __double2half(inShape[2]));
                auto cent2 = centered;
                var = std::move(cent2).pow(2).rowSum() / __double2half(inShape[2]);
            }            
            else {
                centered = std::move(input1) - (std::move(input).rowSum() / inShape[2]);
                auto cent2 = centered;
                var = std::move(cent2).pow(2).rowSum() / inShape[2];
            } 
            auto varShape = var.getShape();
            auto std = (std::move(var) + epsilon).pow(-0.5);
            auto norm = std::move(centered) * std::move(std);
            auto normShape = norm.getShape();
            out = std::move(norm) * gamma + beta;
        }
        else {
            throw std::invalid_argument("LayerNorm only supports rank-2 and rank-3 tensors.");
        }

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
    tensor<t> forward(const std::vector<TokenID>& input);
    tensor<t> forward(const std::vector<std::vector<TokenID>>& input);
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
    tensor<t> forward(size_t len, size_t batchSize = 1);
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
        std::shared_ptr<tensor<t>> Q, K, V;
        if (input.getShape().size() == 2) {
            Q = std::make_shared<tensor<t>>(input.matMul(wQuery));
            K = std::make_shared<tensor<t>>(input.matMul(wKey));
            V = std::make_shared<tensor<t>>(input.matMul(wVal));
        }
        else if (input.getShape().size() == 3) {
            Q = std::make_shared<tensor<t>>(input.matMul(wQuery));
            K = std::make_shared<tensor<t>>(input.matMul(wKey));
            V = std::make_shared<tensor<t>>(input.matMul(wVal));
        }
        else throw std::invalid_argument("Attention only supports 2D or 3D");
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
        std::shared_ptr<tensor<t>> Q, K, V;
        if (input.getShape().size() == 2) {
            Q = std::make_shared<tensor<t>>(input.matMul(wQuery));
            K = std::make_shared<tensor<t>>(input.matMul(wKey));
            V = std::make_shared<tensor<t>>(input.matMul(wVal));
        }
        else if (input.getShape().size() == 3) {
            Q = std::make_shared<tensor<t>>(input.matMul(wQuery));
            K = std::make_shared<tensor<t>>(input.matMul(wKey));
            V = std::make_shared<tensor<t>>(input.matMul(wVal));
        }
        else throw std::invalid_argument("Attention only supports 2D or 3D");
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
class sequential {
    std::vector<std::unique_ptr<module<t>>> modules; 
    std::vector<tensor<t>> outs;
public:
    template <typename...Args>
    requires (std::derived_from<std::decay_t<Args>, module<t>> && ...)
    sequential(Args&&...args) {
        outs.reserve(sizeof...(args));
        modules.reserve(sizeof...(args));
        (
            modules.push_back(std::make_unique<std::decay_t<Args>>(std::forward<Args>(args))),
            ...
        );
    }

    tensor<t>& forward(const tensor<t>& input) {
        outs.clear();
        outs.push_back(modules[0] -> forward(input));
        for (int i = 1; i < modules.size(); i++) {
            outs.push_back(modules[i] -> forward(outs[i - 1]));
        }

        return outs.back();
    }

    tensor<t>& forward(tensor<t>&& input) {
        outs.clear();
        outs.push_back(modules[0] -> forward(std::move(input)));
        for (size_t i = 1; i < modules.size(); i++) {
            outs.push_back(modules[i] -> forward(outs[i - 1]));
        }

        return outs.back();
    }
    std::vector<tensor<t>*> parameters() {
        std::vector<tensor<t>*> out;

        for (auto& i : modules) {
            std::vector<tensor<t>*> temp = i -> parameters();
            for (auto& j : temp) out.push_back(j);
        }
        return out;
    }
};

