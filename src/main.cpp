#include "tensor/tensor.hpp"
#include "nn/module.hpp"
#include "nn/optimizer.hpp"
#include "loss/loss.hpp"
#include <fstream>
#include <iostream>

void checkpoint(const char* name) {
    std::cout
        << "\n===== " << name << " =====\n"
        << "Live tensors: "
        << tensorsCreated - tensorsDestroyed
        << '\n';
}

uint32_t readInt(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));

    return ((value & 0x000000FF) << 24) |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0xFF000000) >> 24);
}

float accuracy(const tensor<float>& pred,
               const tensor<float>& labels) {
    pred.toCPU();
    size_t correct = 0;

    for (size_t i = 0; i < pred.getShape()[0]; i++) {
        int best = 0;

        for (int j = 1; j < 10; j++)
            if (pred(i, j) > pred(i, best))
                best = j;

        if (labels(i, best) == 1.0f)
            correct++;
    }

    return 100.0f * correct / pred.getShape()[0];
}

void mnist() {
    sequential<float> model{
        linear<float>(device::GPU, 256, 784),
        relu<float>(),
        linear<float>(device::GPU, 128, 256),
        relu<float>(),
        linear<float>(device::GPU, 10, 128),
        softmax<float>()
    };
    bool cont = false;
    // while (!cont) std::cin >> cont;
    // cont = false;
    std::ifstream imageFile(
        "datasets/mnist/train-images.idx3-ubyte",
        std::ios::binary);

    std::ifstream labelFile(
        "datasets/mnist/train-labels.idx1-ubyte",
        std::ios::binary);

    std::ifstream testImageFile(
        "datasets/mnist/t10k-images.idx3-ubyte",
        std::ios::binary);

    std::ifstream testLabelFile(
        "datasets/mnist/t10k-labels.idx1-ubyte",
        std::ios::binary);

    if (!imageFile.is_open() || !labelFile.is_open()) {
        std::cout << "Failed to open MNIST files\n";
    }

    uint32_t imageMagic = readInt(imageFile);
    uint32_t numImages  = readInt(imageFile);
    uint32_t rows       = readInt(imageFile);
    uint32_t cols       = readInt(imageFile);

    uint32_t labelMagic = readInt(labelFile);
    uint32_t numLabels  = readInt(labelFile);

    uint32_t testImageMagic = readInt(testImageFile);
    uint32_t testNumImages  = readInt(testImageFile);
    uint32_t testRows       = readInt(testImageFile);
    uint32_t testCols       = readInt(testImageFile);

    uint32_t testLabelMagic = readInt(testLabelFile);
    uint32_t testNumLabels  = readInt(testLabelFile);

    std::cout << "Images : " << numImages << '\n';
    std::cout << "Rows   : " << rows << '\n';
    std::cout << "Cols   : " << cols << '\n';
    std::cout << "Labels : " << numLabels << '\n';
    // while (!cont) std::cin >> cont;
    // cont = false;
    tensor<float> images(device::CPU, numImages, rows * cols);
    tensor<float> labels(device::CPU, numLabels, 10);
    tensor<float> testImages(device::CPU, testNumImages, testRows * testCols);
    tensor<float> testLabels(device::CPU, testNumLabels, 10);

    for (uint32_t i = 0; i < numImages; i++) {

        for (uint32_t j = 0; j < rows * cols; j++) {

            unsigned char pixel;
            imageFile.read(reinterpret_cast<char*>(&pixel), 1);

            images(i, j) = pixel / 255.0f;
        }
    }

    for (uint32_t i = 0; i < numLabels; i++) {

        unsigned char label;
        labelFile.read(reinterpret_cast<char*>(&label), 1);

        labels(i, static_cast<int>(label)) = 1;
    }

    for (uint32_t i = 0; i < testNumImages; i++) {
        for (uint32_t j = 0; j < testRows * testCols; j++) {
            unsigned char pixel;
            testImageFile.read(reinterpret_cast<char*>(&pixel), 1);

            testImages(i, j) = pixel / 255.0f;
        }
    }

    // Read test labels
    for (uint32_t i = 0; i < testNumLabels; i++) {
        unsigned char label;
        testLabelFile.read(reinterpret_cast<char*>(&label), 1);

        testLabels(i, static_cast<int>(label)) = 1;
    }
    std::cout << "\nFirst label: "
              << labels(0, 0)
              << "\n";

    std::cout << "First 20 pixels:\n";


    // images.toGPU();
    std::cout<<'\n';
    Adam<float> optim(model.parameters(), 0.001);
    std::cout<<"training\n";
    // checkpoint("before training loop");
    for (int i = 0; i < 10000; i++) {
        if (i == 330) optim.setLearningrate(0.001);
        // while (!cont) std::cin >> cont;
        // cont = false;
        // std::cout<<"loop start!\n";
        // checkpoint("before forward");
        auto& out = model.forward(images); 
        // checkpoint("after forward");  
        auto loss = crossEntropyLoss(out, labels);
        // checkpoint("after loss");
        loss.toCPU();
        std::cout<< "epoch " << i + 1 << ": " << loss(0, 0) << std::endl;
        loss.toGPU();
        // checkpoint("before backward"); 
        loss.backward();
        // checkpoint("after backward");  
        optim.step();
        // checkpoint("after step");  
        optim.clearGrad();
        // checkpoint("after clearGrad");
        auto pred = model.forward(testImages);
        std::cout << "Accuracy of current model: "<<accuracy(pred, testLabels)<<'\n';
    }

}





#include "nn/gpt.hpp"
#include "tensor/tensor.hpp"
#include "loss/loss.hpp"
#include "tokenizer/tokenizer.hpp"
#include "nn/optimizer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

std::string intToString(std::vector<TokenID> dat) {
    std::string out;
    for (auto& i : dat) out.push_back(char(i + 'A'));

    return out;
}

void ABCD() {
    std::string trainText = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::cout<<trainText<<std::endl;
    std::cout << "size of dataset: "<< trainText.length() << '\n';
    std::vector<TokenID> trainInput;

    for (auto i : trainText) trainInput.push_back((i) - 65);
    for (auto& i : trainInput) std::cout << i <<' ';
    std::cout << '\n';
    std::cout<<"tokenizer ready!!\n";
    std::cout<<"encoded input ready!!\n";
    int context = 8;
    std::cout << "Encoded token count = " << trainInput.size() << '\n';
    GPT<float> model(device::GPU, 26, context, 128, 1);
    Adam<float> opti(model.parameters(), 0.0001); 
    int trainLen = context;
    for (int epoch = 0; epoch < 100; epoch++) {
        std::cout << "epoch: " << epoch << '\n';
        for (int i = trainLen; i < trainInput.size() - 1; i++) {
            auto dataset = std::vector<TokenID>(trainInput.begin() + i - trainLen, trainInput.begin() + i);
            auto target = std::vector<TokenID>(trainInput.begin() + i - trainLen + 1, trainInput.begin() + i + 1);
            auto out = model.forward(dataset);
            auto loss = crossEntropyLoss(out, target);
            if (i == 24) {
                loss.print();
            }
                
            loss.backward();
            opti.step();
            opti.zeroGrad();
            auto ids = out.argMax();
            if (i % 1 == 0) {
            }
        }
    }
    std::cout << '\n';
    for (int i = trainLen; i < trainInput.size() - 1; i++) {
        auto dataset = std::vector<TokenID>(trainInput.begin() + i - trainLen, trainInput.begin() + i);
        std::cout << "Input: " << intToString(dataset) << '\n';
        auto logits = model.forward(dataset);
        std::cout << "Pred: " << intToString(logits.argMax()[0]) << "\n\n";
    }
    auto dataset = std::vector<TokenID>(trainInput.begin(), trainInput.begin() + 1);
    std::cout << "Input: " << intToString(dataset) << '\n';
    auto logits = model.forward(dataset);
    std::cout << "Pred: " << intToString(logits.argMax()[0]) << "\n\n";
}

std::string readDataset(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset: " + filename);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

void tinyShake() {
    std::string trainText = readDataset("datasets/tiny shakespeare/train.csv");
    BPE<float> tokenizer(4096);
    // std::vector<std::string> doc;
    // doc.push_back(trainText);

    // tokenizer.train(doc);
    // tokenizer.save("tokenizerSave/tinyShakespeare/token.bin");
    tokenizer.load("tokenizerSave/tinyShakespeare/token.bin");   
    // std::ofstream inputOut("tempSave/input.bin", std::ios::binary);
    // if (inputOut) {
    //     inputOut.write(reinterpret_cast<char*>(&input), sizeof(input));
    // }
    std::vector<TokenID> input = tokenizer.encode(trainText);
    std::vector<std::vector<TokenID>> inputDoc;
    size_t context = 1024;
    size_t batchSize;
    for (int i = context; i < input.size() - context; ++i) {
        inputDoc.emplace_back(input.begin() + i - context, input.begin() + i);
    }
    std::ofstream inputOut("tempSave/input.bin", std::ios::binary);
    batchSize = inputDoc.size();
    if (inputOut) {
        inputOut.write(reinterpret_cast<char*>(&batchSize), sizeof(size_t));
        inputOut.write(reinterpret_cast<char*>(&inputDoc), batchSize * context * sizeof(TokenID));
    }
    for (int i = 0; i < inputDoc.size(); ++i) {
        for (int j = 0; j < inputDoc[i].size(); ++j) {
            std::cout << inputDoc[i][j] << ' ';
        }
        std::cout << std::endl;
    }
}

int main() {
    tinyShake();
}