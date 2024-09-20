#include <iostream>
#include <vector>
#include <string>

#include "GLiNER/gliner_config.hpp"
#include "GLiNER/processor.hpp"
#include "GLiNER/decoder.hpp"
#include "GLiNER/model.hpp"
#include "GLiNER/tokenizer_utils.hpp"

int main() {
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

    // A sample input
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

    return 0;
}