#pragma once

#include "tensor/tensor.hpp"

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

