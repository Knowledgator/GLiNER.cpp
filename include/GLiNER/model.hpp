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
        const std::string& modelPath;
        Config config;
        Ort::Env *env = nullptr;
        Ort::SessionOptions *sessionOptions = nullptr;
        Ort::Session *session;
        Processor *processor;
        Decoder *decoder;
        std::vector<const char*> inputNames;
        std::vector<const char*> outputNames;

        static bool checkInputs(const std::vector<std::string>& texts, const std::vector<std::string>& entities);
        void initialize(const std::string& tokenizer_path);
        void useDevice(Ort::SessionOptions* session_options, const int device_id);
    public:
        Model(
            const std::string& path, const std::string& tokenizer_path, const Config& config
        );
        Model(
            const std::string& path, const std::string& tokenizer_path, const Config& config, const int device_id
            
        );
        Model(
            const std::string& path, const std::string& tokenizer_path, const Config& config, const Ort::Env& env, const Ort::SessionOptions& session_options
        );
        ~Model();

        static int64_t count_total_elements(std::vector<int64_t>& output_shape);
        void run(const std::vector<Ort::Value>& input_tensors, std::vector<float>& output);
        std::vector<std::vector<Span>> inference(
            const std::vector<std::string>& texts, const std::vector<std::string>& entities, 
            bool flatNer = false, float threshold = 0.5, bool multiLabel = false
        );
    };
}
