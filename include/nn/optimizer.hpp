#pragma once

#include "nn/module.hpp" 
#include <cmath>

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
    void clearGrad() {
        for (auto& i : parameters) i -> clearGrad();
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

template <typename t>
class Adam : public optimizer<t> {
    t learningRate;
    t beta1;
    t beta2;
    std::vector<tensor<t>> m;
    std::vector<tensor<t>> v;
    size_t st = 0;
    t epsilon = 1e-8;
public:
    Adam(const std::vector<tensor<t>*>& parameters, t val = 0.001) : optimizer<t>(parameters), learningRate(val), beta1(0.9), beta2(0.999) {
        for (auto& i : parameters) {
            tensor<t> mt(i -> getDevice(), i -> getShape());
            mt.zeros();

            tensor<t> vt(i -> getDevice(), i -> getShape());
            vt.zeros();

            m.push_back(std::move(mt));
            v.push_back(std::move(vt));            
        }
    }
    void setLearningrate(t val) {
        learningRate = val;
    }
    void step() override {
        st++;
        t bias1 = 1 - std::pow(beta1, st);
        t bias2 = 1 - std::pow(beta2, st);
        for (size_t i = 0; i < this -> parameters.size(); i++) {         
            m[i].toGPU();
            v[i].toGPU();
            m[i] = m[i] * beta1 + *(this -> parameters[i] -> gradient()) * (1 - beta1);
            v[i] = v[i] * beta2 + *(this -> parameters[i] -> gradient()) * *(this -> parameters[i] -> gradient()) * (1 - beta2);
            this -> parameters[i] -> requiresGrad(false);
            *(this -> parameters[i]) = *(this -> parameters[i]) - ((m[i] / bias1) / ((v[i] / bias2).pow(0.5) + epsilon)) * learningRate;
            this -> parameters[i] -> requiresGrad(true);
        }
    }
};