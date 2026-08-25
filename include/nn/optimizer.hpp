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
        for (auto& i : parameters) {
            if (i -> gradient()) i -> gradient() -> zeros();
        }
    }
    void clearGrad() {
        for (auto& i : parameters) {
            i -> clearGrad();
            i -> clearGradientFunction();
        }
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
    float beta1;
    float beta2;
    std::vector<tensor<t>> m;
    std::vector<tensor<t>> v;
    size_t st = 0;
    float epsilon = 1e-8f;
public:
    Adam(const std::vector<tensor<t>*>& parameters, t val = 0.0001) : optimizer<t>(parameters), learningRate(val), beta1(0.9f), beta2(0.999f) {
        for (auto& i : parameters) {
            tensor<t> mt(i -> getDevice(), i -> getShape());
            mt.zeros();
            mt.requiresGrad(false);
            tensor<t> vt(i -> getDevice(), i -> getShape());
            vt.zeros();
            vt.requiresGrad(false);
            m.push_back(std::move(mt));
            v.push_back(std::move(vt));            
        }
    }
    void setLearningrate(t val) {
        learningRate = val;
    }
    void step() override {
        st++;
        float bias1 = 1.0f - std::pow(beta1, st);
        float bias2 = 1.0f - std::pow(beta2, st);

        for (size_t i = 0; i < this->parameters.size(); i++) {
            if (!this->parameters[i]->gradient()) {
                std::cout << "WARNING: param " << i << " has no gradient this step\n";
                continue;
            }
            m[i].toGPU();
            v[i].toGPU();
            this->parameters[i]->requiresGrad(false);

            if constexpr (std::is_same_v<t, __half>) {
                m[i] = m[i] * __float2half(beta1) + *(this->parameters[i]->gradient()) * (__float2half(1.0f) - __float2half(beta1));
                v[i] = v[i] * __float2half(beta2) + *(this->parameters[i]->gradient()) * *(this->parameters[i]->gradient()) * (__float2half(1.0f) - __float2half(beta2));
            }
            else if constexpr (std::is_same_v<t, __nv_bfloat16>) {
                m[i] = m[i] * __float2bfloat16(beta1) + *(this->parameters[i]->gradient()) * (__float2bfloat16(1.0f) - __float2bfloat16(beta1));
                v[i] = v[i] * __float2bfloat16(beta2) + *(this->parameters[i]->gradient()) * *(this->parameters[i]->gradient()) * (__float2bfloat16(1.0f) - __float2bfloat16(beta2));
            }
            else {
                m[i] = m[i] * beta1 + *(this->parameters[i]->gradient()) * (1.0f - beta1);
                v[i] = v[i] * beta2 + *(this->parameters[i]->gradient()) * *(this->parameters[i]->gradient()) * (1.0f - beta2);
            }

            if constexpr (std::is_same_v<t, __half>) {
                *(this->parameters[i]) = *(this->parameters[i]) - ((m[i] / __float2half(bias1)) / ((v[i] / __float2half(bias2)).pow(__float2half(0.5f)) + __float2half(epsilon))) * learningRate;
            }
            else if constexpr (std::is_same_v<t, __nv_bfloat16>) {
                *(this->parameters[i]) = *(this->parameters[i]) - ((m[i] / __float2bfloat16(bias1)) / ((v[i] / __float2bfloat16(bias2)).pow(__float2bfloat16(0.5f)) + __float2bfloat16(epsilon))) * learningRate;
            }
            else {
                *(this->parameters[i]) = *(this->parameters[i]) - ((m[i] / static_cast<t>(bias1)) / ((v[i] / static_cast<t>(bias2)).pow(t(0.5)) + static_cast<t>(epsilon))) * learningRate;
            }
            this->parameters[i]->requiresGrad(true);
        }
    }
};