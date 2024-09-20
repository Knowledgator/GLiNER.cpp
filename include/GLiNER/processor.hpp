#pragma once

#include <tokenizers_cpp.h>

#include <vector>
#include <string>
#include <tuple>

#include "gliner_config.hpp"
#include "gliner_structs.hpp"
#include "tokenizer_utils.hpp"

using tokenizers::Tokenizer;

namespace gliner {
    class SpanProcessor {
    protected:
        Config config;
        WhitespaceTokenSplitter wordSplitter;
        Tokenizer& nonConstTokenizer;

        void encodeInputs(const std::vector<Prompt>& prompts, BatchOutput& output);
        void prepareSpans(const std::vector<Prompt>& prompts, int64_t MaxWidth, int64_t numWords, BatchOutput& output);
    public:
        SpanProcessor(const Config& config, Tokenizer& tokenizer, const WhitespaceTokenSplitter& wordSplitter);
        std::vector<Token> tokenizeText(const std::string& text);
        std::vector<std::vector<Token>> batchTokenizeText(const std::vector<std::string>& texts);
        std::vector<Prompt> prepareTextInputs(const std::vector<std::vector<Token>>& tokens, const std::vector<std::string>& entities);
        BatchOutput prepareBatch(const std::vector<std::string>& texts, const std::vector<std::string>& entities);
    };
}