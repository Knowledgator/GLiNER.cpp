#include <iostream>
#include <vector>
#include <string>

#include "gliner_config.h"
#include "processor.h"
#include "decoder.h"
#include "model.h"
#include "tokenizer_utils.h"

void testWhitespaceTokenSplitter() {
    WhitespaceTokenSplitter splitter;

    std::string text1 = "Hello world_this-is a_test!";
    auto result1 = splitter.call(text1);

    std::cout << "Test WhitespaceTokenSplitter:" << std::endl;
    for (const auto& word : result1) {
        std::cout << "Word: " << std::get<0>(word) << ", Start: " << std::get<1>(word) << ", End: " << std::get<2>(word) << std::endl;
    }
}

void testProcessor() {
    WhitespaceTokenSplitter splitter;
    Config config{12, 512};  // Set your max_width and max_length
    auto blob = LoadBytesFromFile("tokenizer.json");

    // Create the tokenizer
    auto tokenizer = Tokenizer::FromBlobJSON(blob);

    // Create Processor object
    SpanProcessor processor(config, *tokenizer, splitter);

    // Test tokenizeText
    std::string input = "Hello world";
    auto [tokens, startIdx, endIdx] = processor.tokenizeText(input);

    std::cout << "\nTest Processor - Tokenize Text:" << std::endl;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i] << " (" << startIdx[i] << ", " << endIdx[i] << ")" << std::endl;
    }

    // Test batchTokenizeText
    std::vector<std::string> batchInput{"Hello world", "I love C++"};
    auto [batchTokens, batchStartIdx, batchEndIdx] = processor.batchTokenizeText(batchInput);

    std::cout << "\nTest Processor - Batch Tokenize Text:" << std::endl;
    for (size_t i = 0; i < batchTokens.size(); ++i) {
        std::cout << "Batch " << i << " Tokens: ";
        for (const auto& token : batchTokens[i]) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    }
}

void testModelInference() {
    Config config{12, 512};  // Set your max_width and max_length
    WhitespaceTokenSplitter splitter;
    auto blob = LoadBytesFromFile("tokenizer.json");
    
    // Create the tokenizer
    auto tokenizer = Tokenizer::FromBlobJSON(blob);

    // Create Processor and SpanDecoder
    SpanProcessor processor(config, *tokenizer, splitter);
    SpanDecoder decoder(config);

    // Create Model
    Model model("./model.onnx", config, processor, decoder);

    // Test case: Inference on a sample input
    std::vector<std::string> texts = {"Kyiv is the capital of Ukraine."};
    std::vector<std::string> entities = {"city", "country", "river", "person", "car"};

    auto output = model.inference(texts, entities);

    std::cout << "\nTest Model Inference:" << std::endl;
    for (size_t batch = 0; batch < output.size(); ++batch) {
        std::cout << "Batch " << batch << ":\n";
        for (const auto& span : output[batch]) {
            std::cout << "  Span: [" << span.startIdx << ", " << span.endIdx << "], "
                      << "Class: " << span.classLabel << ", "
                      << "Text: " << span.text << ", "
                      << "Prob: " << span.prob << std::endl;
        }
    }
}

int main() {
    try {
        // Run individual tests
        testWhitespaceTokenSplitter();
        testProcessor();
        testModelInference();

        std::cout << "\nAll tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }

    return 0;
}