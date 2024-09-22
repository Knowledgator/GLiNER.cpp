#include <iostream>
#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "GLiNER/gliner_config.hpp"
#include "GLiNER/processor.hpp"
#include "GLiNER/decoder.hpp"
#include "GLiNER/model.hpp"
#include "GLiNER/tokenizer_utils.hpp"

bool compare_tokens(gliner::Token t1, gliner::Token t2) {
    return t1.text == t2.text && t1.start == t2.start && t1.end == t2.end;
}

TEST(TestTopic, TestWhitespaceTokenSplitter) {
    std::vector<gliner::Token> res_map = {
        {0, 5, "Hello"}, 
        {6, 19, "world_this-is"}, 
        {20, 26, "a_test"},
        {26, 27, "!"},
    };

    gliner::WhitespaceTokenSplitter splitter;

    std::string text1 = "Hello world_this-is a_test!";
    auto result1 = splitter.call(text1);
    std::cout << "Results: " << result1.size() << " / " << res_map.size() << std::endl;

    std::cout << "Test WhitespaceTokenSplitter:" << std::endl;
    for (size_t i = 0; i < result1.size(); i++) {
        auto word = result1[i];
        auto expected_word = res_map[i];
        std::cout << "Actual: Word: " << word.text << ", Start: " << word.start << ", End: " << word.end << std::endl;
        std::cout << "Expected: Word: " << expected_word.text << ", Start: " << expected_word.start << ", End: " << expected_word.end << std::endl;
        EXPECT_EQ(compare_tokens(word, res_map[i]), true);
    }
}

TEST(TestTopic, TestProcessor) {
    gliner::WhitespaceTokenSplitter splitter;
    gliner::Config config{12, 512};
    auto blob = gliner::LoadBytesFromFile("./gliner_small-v2.1/tokenizer.json");

    // Create the tokenizer
    auto tokenizer = Tokenizer::FromBlobJSON(blob);

    // Create Processor object
    gliner::SpanProcessor processor(config, *tokenizer, splitter);

    // Test tokenizeText
    std::string input = "Hello world";
    auto tokens = processor.tokenizeText(input);

    std::cout << "\nTest Processor - Tokenize Text:" << std::endl;
    for (auto token : tokens) {
        std::cout << token.text << " (" << token.start << ", " << token.end << ")" << std::endl;
    }

    // Test batchTokenizeText
    std::vector<std::string> batchInput{"Hello world", "I love C++"};
    auto batchTokens = processor.batchTokenizeText(batchInput);

    std::cout << "\nTest Processor - Batch Tokenize Text:" << std::endl;
    for (size_t i = 0; i < batchTokens.size(); ++i) {
        std::cout << "Batch " << i << " Tokens: ";
        for (const auto& token : batchTokens[i]) {
            std::cout << token.text << " ";
        }
        std::cout << std::endl;
    }
}

bool compare_spans(const gliner::Span& a, const gliner::Span& b) {
    return (
        a.startIdx == b.startIdx
        && a.endIdx == b.endIdx
        && a.classLabel == b.classLabel
        && a.text == b.text
        && abs(a.prob - b.prob) < 0.1
    );
}

TEST(TestTopic, TestModelInference) {
    std::vector<std::vector<gliner::Span>> res_map = {{
        {.startIdx = 0, .endIdx = 4, .text = "Kyiv", .classLabel = "city", .prob = 0.9},
        {.startIdx = 23, .endIdx = 30, .text = "Ukraine", .classLabel = "country", .prob = 0.9},
    }};

    gliner::Config config{12, 512};  // Set your max_width and max_length
    gliner::WhitespaceTokenSplitter splitter;
    auto blob = gliner::LoadBytesFromFile("./gliner_small-v2.1/tokenizer.json");
    
    // Create the tokenizer
    auto tokenizer = Tokenizer::FromBlobJSON(blob);

    // Create Processor and SpanDecoder
    gliner::SpanProcessor processor(config, *tokenizer, splitter);
    gliner::SpanDecoder decoder(config);

    // Create Model
    gliner::Model model("./gliner_small-v2.1/onnx/model.onnx", config, processor, decoder);

    // Test case: Inference on a sample input
    std::vector<std::string> texts = {"Kyiv is the capital of Ukraine."};
    std::vector<std::string> entities = {"city", "country", "river", "person", "car"};

    auto output = model.inference(texts, entities);

    std::cout << "\nTest Model Inference:" << std::endl;
    for (size_t batch = 0; batch < output.size(); ++batch) {
        std::cout << "Batch " << batch << ":\n";
        EXPECT_EQ(output[batch].size(), res_map[batch].size());
        for (size_t i = 0; i < output[batch].size(); i++) {
            auto span = output[batch][i];
            auto expected = res_map[batch][i];
            std::cout << "  Actual: Span: [" << span.startIdx << ", " << span.endIdx << "], "
                      << "Class: " << span.classLabel << ", "
                      << "Text: " << span.text << ", "
                      << "Prob: " << span.prob << std::endl;
            std::cout << "  Expected: Span: [" << expected.startIdx << ", " << expected.endIdx << "], "
                      << "Class: " << expected.classLabel << ", "
                      << "Text: " << expected.text << ", "
                      << "Prob: " << expected.prob << std::endl;
            EXPECT_EQ(compare_spans(span, expected), true);
        }
    }
}