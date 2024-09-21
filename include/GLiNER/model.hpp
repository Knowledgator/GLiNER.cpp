#pragma once

#include <onnxruntime_cxx_api.h>

#include <vector>
#include <string>

#include "gliner_structs.hpp"
#include "processor.hpp"
#include "decoder.hpp"

namespace gliner {
    class Model {
    protected:
        const std::string& model_path;
        Config config;
        SpanProcessor processor;
        SpanDecoder decoder;
        Ort::Env env;
        Ort::SessionOptions session_options;
        Ort::Session session;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;

    public:
        Model(const std::string& path, const Config& config, const SpanProcessor& proc, const SpanDecoder& decoder);
        Model(
            const std::string& path, const Config& config, const SpanProcessor& proc, const SpanDecoder& decoder, 
            const Ort::Env& env, const Ort::SessionOptions& so
        );
        std::vector<std::vector<Span>> inference(const std::vector<std::string>& texts, const std::vector<std::string>& entities,
                                                bool flatNer = false, float threshold = 0.5, bool multiLabel = false);
    };
}
