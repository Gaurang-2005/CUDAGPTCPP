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

void run() {
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
    SGD<float> optim(model.parameters(), 0.1);
    std::cout<<"training\n";
    // checkpoint("before training loop");
    for (int i = 0; i < 10000; i++) {
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


// #include "tokenizer/tokenizer.hpp"
// int main() {
//     //run();
//     // std::cout
//     //     << "Created : " << tensorsCreated << '\n'
//     //     << "Destroyed: " << tensorsDestroyed << '\n'
//     //     << "Live     : " << tensorsCreated - tensorsDestroyed << '\n';
//     std::vector<std::string> file;
//     file.push_back("Hello world! Hello world! Hello world!");

//     file.push_back("The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.");

//     file.push_back("Byte Pair Encoding repeatedly merges the most frequent adjacent pair of symbols until the vocabulary reaches the desired size.");

//     file.push_back("Artificial intelligence, machine learning, deep learning, transformers, attention mechanisms, embeddings, optimization, CUDA programming, C++, Python, Linux.");

//     file.push_back("India is the world's most populous country. It has many languages including Hindi, English, Tamil, Telugu, Bengali, Marathi, Punjabi, Gujarati, Kannada, and Malayalam.");

//     file.push_back("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");

//     file.push_back("abcabcabcabcabcabcabcabcabcabcabcabcabcabc");

//     file.push_back("hello hello hello hello hello hello hello hello");

//     file.push_back("token token tokenization tokenizer tokenize tokens tokenization");

//     file.push_back("0 1 2 3 4 5 6 7 8 9");

//     file.push_back("10 20 30 40 50 60 70 80 90 100");

//     file.push_back("1234567890");

//     file.push_back("3.14159265358979323846");

//     file.push_back("-42 +73 -999 1e9");

//     file.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZ");

//     file.push_back("abcdefghijklmnopqrstuvwxyz");

//     file.push_back("MixedCase mixedCASE MiXeDcAsE");

//     file.push_back("! @ # $ % ^ & * ( ) _ + - = { } [ ] : ; \" ' < > , . ? / \\ | ~");

//     file.push_back("This     line     contains     multiple     spaces.");

//     file.push_back("Tabs\tshould\talso\tbe\thandled.");

//     file.push_back("Line one.\nLine two.\nLine three.");

//     file.push_back("Machine learning models learn from data. Machine learning models learn from data. Machine learning models learn from data.");

//     file.push_back("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.");

//     file.push_back("The rain in Spain stays mainly in the plain.");

//     file.push_back("CUDA kernels execute thousands of threads in parallel for high throughput matrix multiplication.");

//     file.push_back("Neural networks consist of layers, activations, weights, biases, gradients, optimizers, and loss functions.");

//     file.push_back("BPE learns subwords like token, tokenizer, tokenize, tokenization, and tokens.");

//     file.push_back("One fish two fish red fish blue fish.");

//     file.push_back("To be, or not to be, that is the question.");

//     file.push_back("The tokenizer should satisfy decode(encode(text)) == text.");
//     BPE tokenizer(10000);
//     std::cout << '\n';
//     tokenizer.train(file);
    
//     auto out = tokenizer.encode(file[0]);
//     for (size_t j = 0; j < out.size(); ++j) {
//         std::cout << out[j] << ' ';
//     } 
//     std::cout << '\n';
//     std::cout << tokenizer.decode(out) << '\n';
// }


#include "nn/gpt.hpp"
#include "tensor/tensor.hpp"
#include "loss/loss.hpp"
#include "tokenizer/tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

std::string readDataset(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset: " + filename);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

int main() {
    std::string trainText = readDataset("datasets/tiny shakespeare/train.csv");

    std::cout << "size of dataset: "<< trainText.length() << '\n';

    BPE tokenizer(1024);
    // std::vector<std::string> doc;
    // doc.push_back(trainText);

    // tokenizer.train(doc);
    // tokenizer.save("tokenizerSave/tinyShakespeare/token.bin");
    tokenizer.load("tokenizerSave/tinyShakespeare/token.bin");

    auto trainInput = tokenizer.encode(trainText);
    GPT<float> model(device::GPU, 1024, 10240, 128, 4);

    auto out = model.forward(trainInput);

}