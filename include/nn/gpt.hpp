#pragma once

#include "tensor/tensor.hpp"
#include "nn/module.hpp"
#include "tokenizer/types.hpp"
#include <vector>

template <typename t>
class GPT {
    tokenEmbedding<t> tokenEmb;
    positionEmbedding<t> posEmb;
    std::vector<transformerBlock<t>> blocks;
    layernorm<t> finalNorm;
    linear<t> lmHead;

public:
    GPT(device dev, size_t vocabSize, size_t contextLength, size_t embedDim, size_t numLayers) : tokenEmb(dev, vocabSize, embedDim), posEmb(dev, contextLength, embedDim), finalNorm(dev, embedDim), lmHead(dev, vocabSize, embedDim) {
        blocks.reserve(numLayers);

        for (size_t i = 0; i < numLayers; ++i) blocks.emplace_back(dev, embedDim);        
    }
    tensor<t> forward(const std::vector<TokenID>& input) {
        std::vector<tensor<t>> output;
        output.reserve(blocks.size() + 1);
        output.push_back(tokenEmb.forward(input) + posEmb.forward(input.size()));
        for (auto& block : blocks) {
            output.push_back(block.forward(std::move(output.back())));
        }
        return lmHead.forward(finalNorm.forward(std::move(output.back())));
    }
    tensor<t> forward(const std::vector<std::vector<TokenID>>& input) {
        if (input.empty())
            throw std::invalid_argument("Input batch cannot be empty.");
        std::vector<tensor<t>> output;
        output.reserve(blocks.size() + 1);
        output.push_back(tokenEmb.forward(input) + posEmb.forward(input[0].size()));
        for (auto& block : blocks) {
            output.push_back(block.forward(std::move(output.back())));
        }
        return lmHead.forward(finalNorm.forward(std::move(output.back())));
    }    
    std::vector<tensor<t>*> parameters() {
        std::vector<tensor<t>*> out;
        out.push_back(tokenEmb.parameters());
        out.push_back(posEmb.parameters());
        for (auto& block : blocks) {
            auto temp = block.parameters();
            for (auto& i : temp) out.push_back(i);
        }
        out.push_back(finalNorm.parameters()[0]);
        out.push_back(finalNorm.parameters()[1]);
        out.push_back(lmHead.parameters()[0]);
        out.push_back(lmHead.parameters()[1]);
        return out;
    }
};