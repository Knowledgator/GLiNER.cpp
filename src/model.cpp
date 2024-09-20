#include <iostream>

#include "GLiNER/model.hpp"

using namespace gliner;

Model::Model(const std::string& path, Config config, SpanProcessor proc, SpanDecoder decoder) :
    model_path(path), config(config), processor(proc), decoder(decoder),
    env(ORT_LOGGING_LEVEL_WARNING, "test"), session_options(),
    session(env, model_path.data(), session_options) 
{
    input_names = {"input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"};
    output_names = {"logits"};
}

bool checkInputs(const std::vector<std::string>& texts, const std::vector<std::string>& entities) {
    return !texts.empty() && !entities.empty();
}

std::vector<std::vector<Span>> Model::inference(const std::vector<std::string>& texts, const std::vector<std::string>& entities,
                                        bool flatNer,
                                        float threshold,
                                        bool multiLabel) {
    if (!checkInputs(texts, entities)) {
        std::cerr << "WARNING! Empty texts or entities." << std::endl;
        return {};
    }
                            
    BatchOutput batch = processor.prepareBatch(texts, entities);
    std::vector<Ort::Value> input_tensors;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {batch.batchSize, batch.numTokens};

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, batch.inputsIds.data(), batch.inputsIds.size(), shape.data(), shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, batch.attentionMasks.data(), batch.attentionMasks.size(), shape.data(), shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, batch.wordsMasks.data(), batch.wordsMasks.size(), shape.data(), shape.size()));
    
    std::vector<int64_t> text_lengths_shape = {batch.batchSize, 1};

    std::vector<int64_t> textLengths;
    textLengths.reserve(batch.batchPrompts.size());
    for (auto p : batch.batchPrompts) {
        textLengths.push_back(p.textLength);
    }

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, textLengths.data(), textLengths.size(), text_lengths_shape.data(), text_lengths_shape.size()));

    int64_t numSpans = batch.numWords*batch.maxWidth;

    std::vector<int64_t> span_shape = {batch.batchSize, numSpans, 2};
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, batch.spanIdxs.data(), batch.spanIdxs.size(), span_shape.data(), span_shape.size()));    
    
    std::vector<int64_t> span_mask_shape = {batch.batchSize, numSpans};
    input_tensors.push_back(Ort::Value::CreateTensor<bool>(memory_info, reinterpret_cast<bool*>(batch.spanMasks.data()), batch.spanMasks.size(), span_mask_shape.data(), span_mask_shape.size()));
    
    std::vector<Ort::Value> modelOutputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

    Ort::Value& output_tensor = modelOutputs[0];
    Ort::TensorTypeAndShapeInfo output_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();

    // Compute total number of elements
    int64_t total_elements = 1;
    for (int64_t i : output_shape) {
        total_elements *= i;
    }

    const float* output_data = output_tensor.GetTensorData<float>();
    std::vector<float> modelOutput(output_data, output_data + total_elements);

    return decoder.decode(
        batch,
        config.max_width,
        texts,
        entities,
        modelOutput,
        flatNer, threshold, multiLabel
    );
}
