#pragma once

#include "tensor/tensor.hpp"
#include "nn/module.hpp"
#include "tokenizer/types.hpp"
#include <vector>
#include <fstream> 

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
        output.push_back(tokenEmb.forward(input) + posEmb.forward(input[0].size(), input.size()));
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
    void save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);

        if (!file) {
            throw std::runtime_error("Could not open model file for writing: " + filename);
        }

        // Magic number / file identifier
        const char magic[] = "CUDAGPT";
        file.write(magic, sizeof(magic));

        // Number of parameters
        uint64_t numParams = parameters().size();
        file.write(reinterpret_cast<const char*>(&numParams),
                sizeof(numParams));

        for (auto& param : parameters()) {

            // Save shape
            const auto& shape = param->getShape();

            uint64_t ndim = shape.size();
            file.write(reinterpret_cast<const char*>(&ndim),
                    sizeof(ndim));

            for (auto dim : shape) {
                uint64_t d = dim;
                file.write(reinterpret_cast<const char*>(&d),
                        sizeof(d));
            }

            // Make sure data is on CPU before accessing it
            param->toCPU();

            uint64_t numElements = 1;
            for (auto dim : shape)
                numElements *= dim;

            file.write(
                reinterpret_cast<const char*>(param->data()),
                numElements * sizeof(t)
            );

            // Put parameter back on GPU
            param->toGPU();
        }

        file.close();

        if (!file) {
            throw std::runtime_error("Error while writing model file.");
        }
    }
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file) {
            throw std::runtime_error("Could not open model file: " + filename);
        }

        // Check magic number
        char magic[sizeof("CUDAGPT")];
        file.read(magic, sizeof(magic));

        if (std::string(magic) != "CUDAGPT") {
            throw std::runtime_error("Invalid CUDA-GPT model file.");
        }

        uint64_t numParams;
        file.read(reinterpret_cast<char*>(&numParams),
                sizeof(numParams));

        if (numParams != parameters().size()) {
            throw std::runtime_error(
                "Parameter count mismatch. "
                "Saved model: " + std::to_string(numParams) +
                ", current model: " + std::to_string(parameters().size())
            );
        }

        for (size_t i = 0; i < numParams; ++i) {

            auto param = parameters()[i];

            // Read shape
            uint64_t ndim;
            file.read(reinterpret_cast<char*>(&ndim),
                    sizeof(ndim));

            std::vector<size_t> savedShape(ndim);

            for (uint64_t j = 0; j < ndim; ++j) {
                uint64_t dim;
                file.read(reinterpret_cast<char*>(&dim),
                        sizeof(dim));

                savedShape[j] = dim;
            }

            // Verify shape
            if (savedShape != param->getShape()) {
                throw std::runtime_error(
                    "Shape mismatch for parameter " +
                    std::to_string(i)
                );
            }

            uint64_t numElements = 1;

            for (auto dim : savedShape)
                numElements *= dim;

            // Read into CPU tensor
            param->toCPU();

            file.read(
                reinterpret_cast<char*>(param->data()),
                numElements * sizeof(t)
            );

            if (!file) {
                throw std::runtime_error(
                    "Unexpected end of model file."
                );
            }

            // Move trained weights back to GPU
            param->toGPU();
        }

        file.close();
    }
};