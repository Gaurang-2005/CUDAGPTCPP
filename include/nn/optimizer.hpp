#pragma once

#include "nn/module.hpp" 

template <typename t>
class optimizer {
protected:
    std::vector<tensor<t>*> parameters;
public:
    optimizer(const std::vector<tensor<t>*>& parameters) : parameters(parameters) {}

    virtual void step() = 0;

    void add(const tensor<t>* parameter) {
        parameters.push_back(parameter);
    }

    void zeroGrad() {
        for (auto& i : parameters) i -> gradient() -> zeros();
    }
};

template <typename t>
class SGD : public optimizer<t> {
    t learningRate;
public:
    SGD(const std::vector<tensor<t>*>& parameters, t val = 0.001) : optimizer<t>(parameters), learningRate(val) {}
    void setLearningrate(t val) {
        learningRate = val;
    }
    void step() override {
        for (auto& i : this -> parameters) {
            i->requiresGrad(false);
            i->toGPU();
            i->gradient()->toGPU();
            *i -= learningRate * (*i -> gradient());
            i->requiresGrad(true);
        }
    }
};