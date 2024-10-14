#pragma once

#include <tokenizers_cpp.h>

#include <vector>
#include <string>

#include "gliner_config.hpp"
#include "gliner_structs.hpp"
#include "tokenizer_utils.hpp"

namespace gliner {
    class Processor {
    protected:
        Config config;
        std::unique_ptr<tokenizers::Tokenizer> tokenizer;
        WhitespaceTokenSplitter wordSplitter;

        void encodeInputs(const std::vector<Prompt>& prompts, Batch* output);
        virtual void prepareTextInputs(
            const std::vector<std::string>& entities, Batch* output, std::vector<Prompt>& prompts
        );
    public:
        Processor(const Config& config, const std::string& tokenizer_path);
        virtual ~Processor() {};
        std::vector<Token> tokenizeText(const std::string& text);
        std::vector<std::vector<Token>> batchTokenizeText(const std::vector<std::string>& texts);
        
        virtual Batch* prepareBatch(
            const std::vector<std::string>& texts, const std::vector<std::string>& entities
        ) = 0; 
    };

    class SpanProcessor : public Processor {
    protected:
        void prepareSpans(const std::vector<Prompt>& prompts, SpanBatch* output);
    public:
        SpanProcessor(const Config& config, const std::string& tokenizer_path);
        virtual ~SpanProcessor() {};
        virtual Batch* prepareBatch(
            const std::vector<std::string>& texts, const std::vector<std::string>& entities
        ); 
    };

    class TokenProcessor : public Processor {
    public:
        TokenProcessor(const Config& config, const std::string& tokenizer_path);
        virtual ~TokenProcessor() {};
        virtual Batch* prepareBatch(
            const std::vector<std::string>& texts, const std::vector<std::string>& entities
        ); 
    };
}