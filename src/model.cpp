#include <iostream>

#include "GLiNER/model.hpp"

using namespace gliner;

Model::Model(
    const std::string& model_path, const std::string& tokenizer_path, const Config& config
) : modelPath(model_path), config(config)
{
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "gliner");
    sessionOptions = new Ort::SessionOptions();
    session = new Ort::Session(*env, modelPath.data(), *sessionOptions);
    initialize(tokenizer_path);
}

Model::Model(
    const std::string& path, const std::string& tokenizer_path, const Config& config, const int device_id
) : modelPath(path), config(config)
{
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "gliner");
    sessionOptions = new Ort::SessionOptions();
    useDevice(sessionOptions, device_id);
    session = new Ort::Session(*env, modelPath.data(), *sessionOptions);
    initialize(tokenizer_path);
}

Model::Model(
    const std::string& path, const std::string& tokenizer_path, const Config& config, const Ort::Env& env, const Ort::SessionOptions& session_options
) : modelPath(path), config(config)
{
    session = new Ort::Session(env, modelPath.data(), session_options);
    initialize(tokenizer_path);
}

Model::~Model() {
    if (env != nullptr) {
        delete env;
    }

    if (sessionOptions != nullptr) {
        delete sessionOptions;
    }

    delete session;
    delete processor;
    delete decoder;
}

bool Model::checkInputs(const std::vector<std::string>& texts, const std::vector<std::string>& entities) {
    return !texts.empty() && !entities.empty();
}

void Model::initialize(const std::string& tokenizer_path) {
    switch (config.modelType){
    case TOKEN_LEVEL:
        processor = new TokenProcessor(config, tokenizer_path);
        decoder = new TokenDecoder();
        inputNames = {"input_ids", "attention_mask", "words_mask", "text_lengths"};
        outputNames = {"logits"};
        break;
    case SPAN_LEVEL:
        processor = new SpanProcessor(config, tokenizer_path);
        decoder = new SpanDecoder();
        inputNames = {"input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"};
        outputNames = {"logits"};
        break;
    }
}

void Model::useDevice(Ort::SessionOptions* session_options, const int device_id) {
    if (device_id >= 0) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options->AppendExecutionProvider_CUDA(cuda_options);
    }
}

int64_t Model::count_total_elements(std::vector<int64_t>& output_shape) {
    int64_t total_elements = 1;
    for (int64_t i : output_shape) {
        total_elements *= i;
    }
    return total_elements;
}

void Model::run(const std::vector<Ort::Value>& input_tensors, std::vector<float>& output) {
    std::vector<Ort::Value> modelOutputs = session->Run(
        Ort::RunOptions(), inputNames.data(), 
        input_tensors.data(), inputNames.size(), 
        outputNames.data(), outputNames.size()
    );
    Ort::Value& output_tensor = modelOutputs[0];
    Ort::TensorTypeAndShapeInfo output_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();

    const float* output_data = output_tensor.GetTensorData<float>();
    output = std::vector<float>(output_data, output_data + count_total_elements(output_shape));
}

std::vector<std::vector<Span>> Model::inference(
    const std::vector<std::string>& texts, const std::vector<std::string>& entities, bool flatNer, float threshold, bool multiLabel
) {
    if (!checkInputs(texts, entities)) {
        std::cerr << "WARNING! Empty texts or entities." << std::endl;
        return {};
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> output;

    // std::vector<Prompt> prompts; // TODO: delete
    Batch* batch = processor->prepareBatch(texts, entities);

    std::vector<Ort::Value> input_tensors;
    batch->tensors(input_tensors, memory_info);
    run(input_tensors, output);

    auto decoded = decoder->decode(
        batch, texts, entities, output, flatNer, threshold, multiLabel
    );
    delete batch;
    return decoded;
}
