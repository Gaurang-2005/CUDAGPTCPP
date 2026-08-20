
#include "nn/gpt.hpp"
#include "tokenizer/tokenizer.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
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
    int context = 4;
    std::cout << "Encoded token count = " << trainInput.size() << '\n';
    GPT<float> model(device::GPU, 26, context, 128, 1);
    Adam<float> opti(model.parameters(), 0.001); 
    int trainLen = 4;
    std::vector<std::vector<TokenID>> inputDat;
    std::vector<std::vector<TokenID>> targDat;
    for (int i = trainLen; i < trainInput.size() - 1; i++) {
        inputDat.push_back(std::vector<TokenID>(trainInput.begin() + i - trainLen, trainInput.begin() + i));
        targDat.push_back(std::vector<TokenID>(trainInput.begin() + i - trainLen + 1, trainInput.begin() + i + 1));
    }
    for (int epoch = 0; epoch < 1000; epoch++) {
        std::cout << "epoch: " << epoch << " loss: ";
        auto out = model.forward(inputDat);
        auto loss = crossEntropyLoss(out, targDat);
        loss.toCPU();
        std::cout << loss(0, 0) << '\n';

                
            loss.backward();
            auto params = model.parameters();
            // for (auto& i : params) {
            //     i -> gradient() -> print();
            // }
            opti.step();
            opti.clearGrad();
            // auto ids = out.argMax();

            // for (int i = trainLen; i < trainInput.size() - 1; i++) {
            //     auto dataset = std::vector<TokenID>(trainInput.begin() + i - trainLen, trainInput.begin() + i);
            //     std::cout << "Input: " << intToString(dataset) << '\n';
            //     auto logits = model.forward(dataset);
            //     std::cout << "Pred: " << intToString(logits.argMax()[0]) << "\n\n";
            // }

        }
    
    std::cout << '\n';
    for (int i = trainLen; i < trainInput.size() - 1; i++) {
        auto dataset = std::vector<TokenID>(trainInput.begin() + i - trainLen, trainInput.begin() + i);
        std::cout << "Input: " << intToString(dataset) << '\n';
        auto logits = model.forward(dataset);
        std::cout << "Pred: " << intToString(logits.argMax()[0]) << "\n\n";
    }
    // auto dataset = std::vector<TokenID>(trainInput.begin(), trainInput.begin() + 1);
    // std::cout << "Input: " << intToString(dataset) << '\n';
    // auto logits = model.forward(dataset);
    // std::cout << "Pred: " << intToString(logits.argMax()[0]) << "\n\n";
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

void saveBatchedInput(const std::string& filename, const std::vector<std::vector<TokenID>>& inputDoc) {
    std::ofstream output(filename, std::ios::binary);

    if (!output.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    size_t batchSize = inputDoc.size();
    size_t seqLen = inputDoc[0].size();

    // Save metadata
    output.write(reinterpret_cast<char*>(&batchSize), sizeof(size_t));
    output.write(reinterpret_cast<char*>(&seqLen), sizeof(size_t));

    // Save token data
    for (const auto& seq : inputDoc) {
        output.write(
            reinterpret_cast<const char*>(seq.data()),
            seqLen * sizeof(TokenID)
        );
    }

    output.close();
}


std::vector<std::vector<TokenID>> readBatchedInput(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    size_t batchSize;
    size_t seqLen;

    // Read metadata
    input.read(reinterpret_cast<char*>(&batchSize), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&seqLen), sizeof(size_t));

    std::vector<std::vector<TokenID>> inputDoc(
        batchSize,
        std::vector<TokenID>(seqLen)
    );

    // Read token data
    for (auto& seq : inputDoc) {
        input.read(
            reinterpret_cast<char*>(seq.data()),
            seqLen * sizeof(TokenID)
        );
    }

    input.close();

    return inputDoc;
}

template <typename t>
float validationLoss(
    GPT<t>& model,
    const std::vector<std::vector<TokenID>>& valInput,
    const std::vector<std::vector<TokenID>>& valTarget,
    size_t batchSize
) {
    std::cout << "valSize: " << valInput.size() << '\n';
    t totalLoss = 0.0;
    size_t numBatches = 0;
    for (size_t start = 0;
         start + batchSize <= valInput.size();
         start += batchSize) {

        std::vector<std::vector<TokenID>> in(
            valInput.begin() + start,
            valInput.begin() + start + batchSize
        );

        std::vector<std::vector<TokenID>> target(
            valTarget.begin() + start,
            valTarget.begin() + start + batchSize
        );

        auto out = model.forward(in);
        // out.print();
        auto loss = crossEntropyLoss<t>(out, target);
        // loss.print();
        // loss is a (1,1) tensor in your implementation
        loss.toCPU();
        ++numBatches;
        if constexpr (std::is_same_v<t, __half>) {
            totalLoss += loss.data()[0] / __double2half(numBatches);
        }
        else {
            totalLoss += loss.data()[0] / numBatches;
        }
    }

    return totalLoss;
}
#include <chrono>
using modelDType = float;
void tinyShake() {
    // std::string trainText = readDataset("datasets/tiny shakespeare/train.csv");
    BPE tokenizer(4096);
    // std::vector<std::string> doc;
    // doc.push_back(trainText);
    // tokenizer.train(doc);
    // tokenizer.save("datasets/tiny shakespeare/token.bin");
    tokenizer.load("datasets/tiny shakespeare/token.bin");   
    // std::ofstream inputOut("tempSave/input.bin", std::ios::binary);
    // if (inputOut) {
    //     inputOut.write(reinterpret_cast<char*>(&input), sizeof(input));
    // }
    // std::vector<TokenID> input = tokenizer.encode(trainText);
    std::vector<std::vector<TokenID>> inputDoc;
    std::vector<std::vector<TokenID>> targetDoc;
    size_t context = 64;
    size_t batchSize;
    // for (int i = context; i < input.size() - context + 1; ++i) {
    //     inputDoc.emplace_back(input.begin() + i - context, input.begin() + i);
    //     targetDoc.emplace_back(input.begin() + i - context + 1, input.begin() + i + 1);
    // }
    // saveBatchedInput("datasets/tiny shakespeare/input.bin", inputDoc);
    // saveBatchedInput("datasets/tiny shakespeare/target.bin", targetDoc);
    inputDoc = readBatchedInput("datasets/tiny shakespeare/input.bin");
    targetDoc = readBatchedInput("datasets/tiny shakespeare/target.bin");
    batchSize = inputDoc.size();
    // for (int i = 0; i < inputDoc.size(); ++i) {
    //     for (int j = 0; j < inputDoc[i].size(); ++j) {
    //         std::cout << inputDoc[i][j] << ' ';
    //     }
    //     std::cout << std::endl;
    // }
    GPT<modelDType> model(device::GPU, (4096), context, 384, 2);
    Adam<modelDType> opti(model.parameters(), 0.1);
    batchSize = 160;
    std::vector<std::vector<TokenID>> validationInput;
    std::vector<std::vector<TokenID>> validationTarget;
    // std::cout << "test token size: " << testTokens.size() << '\n';
    // for (size_t i = 0; i + context < testTokens.size(); ++i) {
    //     testInput.emplace_back(
    //         testTokens.begin() + i,
    //         testTokens.begin() + i + context
    //     );

    //     testTarget.emplace_back(
    //         testTokens.begin() + i + 1,
    //         testTokens.begin() + i + context + 1
    //     );
    // }
    bool achieved = false;
    validationInput = readBatchedInput("datasets/tiny shakespeare/validationInput.bin");
    validationTarget = readBatchedInput("datasets/tiny shakespeare/validationTarget.bin");
    // model.load("datasets/tiny shakespeare/model (1).bin");
    for (int i = 0; i < 100; i++) {
        std::cout << "epoch: " << i <<'\n';
        int temp = 0;
        size_t remain = (inputDoc.end() - inputDoc.begin()) / batchSize;
        int valcnt = 0;
        for (auto batch = 0; batch < inputDoc.size() - batchSize; batch += batchSize) {
            {
                // std::cout << "live Tensors: " << tensorsCreated - tensorsDestroyed << '\n';
                // while (temp) {}
                // temp++;
                std::cout << "Data left: " << remain-- << '\n';
                auto in = std::vector<std::vector<TokenID>>(inputDoc.begin() + batch, inputDoc.begin() + batch + batchSize);
                auto out = model.forward(in);
                auto targ = std::vector<std::vector<TokenID>>(targetDoc.begin() + batch, targetDoc.begin() + batch + batchSize);
                auto loss = crossEntropyLoss<modelDType>(out, targ);
                // auto start = std::chrono::steady_clock::now();
                loss.backward();
                // auto end = std::chrono::steady_clock::now();
                // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                // std::cout << "Execution time: " << elapsed.count() << " ms" << std::endl;
                opti.step();
                opti.clearGrad();
                loss.clearGradientFunction();
                if (!(valcnt % 500)) {}
                
                loss.print();

                model.save("datasets/tiny shakespeare/model.bin");

                // auto out2 = model.forward(in);
                // auto predictions = out2.argMax();
                // int total = 0;
                // int correct = 0;
                // for (size_t b = 0; b < predictions.size(); ++b) {
                //     for (size_t t = 0; t < predictions[b].size(); ++t) {

                //         if (predictions[b][t] == targ[b][t])
                //             ++correct;

                //         ++total;
                //     }
                // }
                // modelDType accuracy =
                //     static_cast<modelDType>(correct) / static_cast<modelDType>(total);

                // std::cout << "Test accuracy: "
                //         << accuracy * 100.0f
                //         << "%\n";


            }
            if ((valcnt == 0)) {
                auto valLoss = validationLoss(model, validationInput, validationTarget, batchSize);
                std::cout << "Validation: " << valLoss << '\n';
                if (valLoss < 2) {
                    achieved = true;
                    break;
                }
            }
            valcnt++; 
        }
        if (achieved) break;
    }
    model.save("datasets/tiny shakespeare/model.bin");

    std::vector<std::vector<TokenID>> testInput;
    std::vector<std::vector<TokenID>> testTarget;

    std::string testText =
        readDataset("datasets/tiny shakespeare/test.csv");

    std::vector<TokenID> testTokens =
        tokenizer.encode(testText);

    for (size_t i = 0; i + context < testTokens.size(); ++i) {

        testInput.emplace_back(
            testTokens.begin() + i,
            testTokens.begin() + i + context
        );

        testTarget.emplace_back(
            testTokens.begin() + i + 1,
            testTokens.begin() + i + context + 1
        );
    }

    size_t correct = 0;
    size_t total = 0;

    for (size_t i = 0; i < testInput.size(); i += batchSize) {

        size_t currentBatchSize =
            std::min(batchSize, testInput.size() - i);

        std::vector<std::vector<TokenID>> in(
            testInput.begin() + i,
            testInput.begin() + i + currentBatchSize
        );

        std::vector<std::vector<TokenID>> target(
            testTarget.begin() + i,
            testTarget.begin() + i + currentBatchSize
        );

        auto out = model.forward(in);
        auto predictions = out.argMax();

        for (size_t b = 0; b < predictions.size(); ++b) {
            for (size_t t = 0; t < predictions[b].size(); ++t) {

                if (predictions[b][t] == target[b][t])
                    ++correct;

                ++total;
            }
        }
        break;
    }

    modelDType accuracy =
        static_cast<modelDType>(correct) / static_cast<modelDType>(total);

    std::cout << "Test accuracy: "
            << __half2float(accuracy) * 100.0f
            << "%\n";
}


// int main() {
//     std::string trainText = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
//     std::cout<<trainText<<std::endl;
//     std::cout << "size of dataset: "<< trainText.length() << '\n';
//     std::vector<TokenID> trainInput;

//     for (auto i : trainText) trainInput.push_back((i) - 65);
//     for (auto& i : trainInput) std::cout << i <<' ';
//     std::cout << '\n';
//     // BPE<float> tokenizer(26);
//     // std::vector<std::string> doc;
//     // doc.push_back(trainText);

//     // tokenizer.train(doc);
//     // tokenizer.save("tokenizerSave/tinyShakespeare/token.bin");
//     // tokenizer.load("tokenizerSave/tinyShakespeare/token.bin");
//     std::cout<<"tokenizer ready!!\n";
//     std::cout<<"encoded input ready!!\n";
//     int context = 4;
//     // auto trainInput = tokenizer.encode(trainText);
//     std::cout << "Encoded token count = " << trainInput.size() << '\n';
//     GPT<float> model(device::GPU, 26, context, 128, 1);
//     std::vector<std::string> str;
//     int cnt = 0;
//     Adam<float> opti(model.parameters(), 0.001); 
//     int trainLen = context;
//     for (int epoch = 0; epoch < 1000; epoch++) {
//         std::cout << "epoch: " << epoch << '\n';
//         for (int i = trainLen; i < trainInput.size() - 1; i++) {
//             auto dataset = std::vector<TokenID>(trainInput.begin() + i - trainLen, trainInput.begin() + i);
//             auto target = std::vector<TokenID>(trainInput.begin() + i - trainLen + 1, trainInput.begin() + i + 1);
//             auto out = model.forward(dataset);
//             // std::cout<<"OUTPUT: \n";
//             // out.print();
//             // std::cout << "forward passed\n";
//             auto loss = crossEntropyLoss(out, target);
//             // std::cout << "loss passed\n";
//                 // out.print();
//             std::cout<<"LOSS: \n";
//             loss.print();
//             // std::cout<<"PARAMETERS: \n";
//             // for (auto& i : model.parameters()) i -> print();
//             // if (epoch == 600) opti.setLearningrate(1e-5);
//             // model.grad();
//             loss.backward();
//             // std::cout <<"OUTPUT GRADIENT: \n";
//             // loss.gradient() -> print();
//             // std::cout<<"PARAMETERS GRADIENTS: \n";
//             // // for (auto& i : model.parameters()) i -> gradient() -> print();
//             // model.grad();
//             opti.step();
//             // opti.zeroGrad();
//             // std::cout << "opti passed\n";
//             auto ids = out.argMax()[0];
//             // std::cout << "argmax passed\n";
//             //for (auto& i : ids) std::cout << i <<' ';
//             if (i % 1 == 0) {
//                 // std::cout << "Input  : " << intToString(dataset) << '\n';
//                 // std::cout << "Target : " << intToString(target) << '\n';
//                 // std::cout << "Pred   : " << intToString(ids) << '\n';
//             }
//         }
//     }
//     std::cout << '\n';
//     for (int i = trainLen; i < trainInput.size() - 1; i++) {
//         auto dataset = std::vector<TokenID>(trainInput.begin() + i - trainLen, trainInput.begin() + i);
//         std::cout << "Input: " << intToString(dataset) << '\n';
//         auto logits = model.forward(dataset);
//         std::cout << "Pred: " << intToString(logits.argMax()[0]) << "\n\n";
//     }
//     auto dataset = std::vector<TokenID>(trainInput.begin(), trainInput.begin() + 1);
//     std::cout << "Input: " << intToString(dataset) << '\n';
//     auto logits = model.forward(dataset);
//     std::cout << "Pred: " << intToString(logits.argMax()[0]) << "\n\n";
// }
int main() {
    tinyShake();
}


