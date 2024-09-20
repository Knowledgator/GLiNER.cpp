#pragma once

#include <vector>
#include <string>
#include <regex>
#include <unordered_map>
#include <cstdint>

namespace gliner {
    struct Token {
        size_t start;
        size_t end;
        std::string text;
    };

    struct Prompt {
        int64_t textLength;
        int64_t promptLength;
        std::vector<std::string> prompt;
    };

    struct BatchOutput {
        int64_t batchSize;
        int64_t numTokens;
        int64_t numWords;
        int64_t maxWidth;
        std::vector<int64_t> inputsIds;
        std::vector<int64_t> attentionMasks;
        std::vector<int64_t> wordsMasks;
        std::vector<int64_t> spanIdxs;
        std::vector<uint8_t> spanMasks;
        std::unordered_map<int, std::string> idToClass;
        std::vector<std::vector<Token>> batchTokens;
        std::vector<Prompt> batchPrompts;
    };

    struct Span {
        int startIdx;
        int endIdx;
        std::string text;
        std::string classLabel;
        float prob;
    };
}