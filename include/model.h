#pragma once

#include <onnxruntime_cxx_api.h>

#include <vector>
#include <string>

#include "gliner_structs.h"
#include "processor.h"
#include "decoder.h"

class Model {
protected:
    const char* model_path;
    Config config;
    SpanProcessor processor;
    SpanDecoder decoder;
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;

public:
    Model(const char* path, Config config, SpanProcessor proc, SpanDecoder decoder);
    void prepareInput(const BatchOutput& batch, std::vector<Ort::Value>& input_tensors, InputData& inputData);
    std::vector<std::vector<Span>> inference(const std::vector<std::string>& texts, const std::vector<std::string>& entities,
                                             bool flatNer = false, float threshold = 0.5, bool multiLabel = false);
};
