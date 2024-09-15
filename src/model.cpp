
#include "model.h"

Model::Model(const char* path, Config config, SpanProcessor proc, SpanDecoder decoder) :
    model_path(path), config(config), processor(proc), decoder(decoder),
    env(ORT_LOGGING_LEVEL_WARNING, "test"), session_options(),
    session(env, model_path, session_options) {}

void Model::prepareInput(const BatchOutput& batch, std::vector<Ort::Value>& input_tensors, InputData& inputData) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {batch.batchSize, batch.numTokens};

    inputData.inputsIds = batch.inputsIds;
    inputData.attentionMasks = batch.attentionMasks;
    inputData.wordsMasks = batch.wordsMasks;

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.inputsIds.data(), inputData.inputsIds.size(), shape.data(), shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.attentionMasks.data(), inputData.attentionMasks.size(), shape.data(), shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.wordsMasks.data(), inputData.wordsMasks.size(), shape.data(), shape.size()));
    
    std::vector<int64_t> text_lengths_shape = {batch.batchSize, 1};

    inputData.textLengths = batch.textLengths;

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.textLengths.data(), inputData.textLengths.size(), text_lengths_shape.data(), text_lengths_shape.size()));

    int64_t numSpans = batch.numWords*batch.maxWidth;

    std::vector<int64_t> span_shape = {batch.batchSize, numSpans, 2};
    std::vector<int64_t> span_mask_shape = {batch.batchSize, numSpans};

    inputData.spanIdxs = batch.spanIdxs;

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.spanIdxs.data(), inputData.spanIdxs.size(), span_shape.data(), span_shape.size()));
    
    inputData.spanMasksBool.reserve(batch.spanMasks.size());
    for (const auto& mask : batch.spanMasks) {
        inputData.spanMasksBool.push_back(mask != 0);
    };

    input_tensors.push_back(Ort::Value::CreateTensor<bool>(memory_info, reinterpret_cast<bool*>(inputData.spanMasksBool.data()), inputData.spanMasksBool.size(), span_mask_shape.data(), span_mask_shape.size()));
}

std::vector<std::vector<Span>> Model::inference(const std::vector<std::string>& texts, const std::vector<std::string>& entities,
                                        bool flatNer,
                                        float threshold,
                                        bool multiLabel) {
    BatchOutput batch = processor.prepareBatch(texts, entities);
    std::vector<Ort::Value> input_tensors;
    InputData inputData;

    prepareInput(batch, input_tensors, inputData);

    std::vector<const char*> input_names = {"input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"};
    std::vector<const char*> output_names = {"logits"}; 
    
    std::vector<Ort::Value> modelOutputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

    Ort::Value& output_tensor = modelOutputs[0];
    Ort::TensorTypeAndShapeInfo output_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();

    // Compute total number of elements
    size_t total_elements = 1;
    for (size_t i = 0; i < output_shape.size(); ++i) {
        total_elements *= output_shape[i];
    }

    const float* output_data = output_tensor.GetTensorData<float>();
    std::vector<float> modelOutput(output_data, output_data + total_elements);

    int batchSize = batch.textLengths.size();
    int maxWidth = config.max_width;
    auto max_iterator = std::max_element(batch.textLengths.begin(), batch.textLengths.end());
    int inputLength = *max_iterator;
    int numEntities = entities.size();

    std::vector<std::vector<Span>>  outputSpans = decoder.decode(
        batchSize, inputLength, maxWidth, numEntities,
        texts,
        batch.batchWordsStartIdx, batch.batchWordsEndIdx, 
        batch.idToClass,
        modelOutput,
        flatNer, threshold, multiLabel
    );
    return outputSpans;
}
