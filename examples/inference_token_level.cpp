#include <iostream>
#include <vector>
#include <string>

#include "GLiNER/gliner_config.hpp"
#include "GLiNER/model.hpp"

int main() {
    gliner::Config config{12, 512, gliner::TOKEN_LEVEL};  // Set your maxWidth, maxLength and modelType
    // Create Model
    gliner::Model model("./gliner-multitask-large-v0.5/onnx/model.onnx", "./gliner-multitask-large-v0.5/tokenizer.json", config);
    // Provide the path to the model, the path to the tokenizer, and the configuration.

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