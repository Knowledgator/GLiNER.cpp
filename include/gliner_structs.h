#pragma once

#include <vector>
#include <string>
#include <regex>

struct BatchOutput {
    int64_t batchSize;
    int64_t numTokens;
    int64_t numWords;
    int64_t maxWidth;
    std::vector<int64_t> inputsIds;
    std::vector<int64_t> attentionMasks;
    std::vector<int64_t> wordsMasks;
    std::vector<int64_t> textLengths;
    std::vector<int64_t> spanIdxs;
    std::vector<uint8_t> spanMasks;
    std::unordered_map<int, std::string> idToClass;
    std::vector<std::vector<std::string>> batchTokens;
    std::vector<std::vector<int>> batchWordsStartIdx;
    std::vector<std::vector<int>> batchWordsEndIdx;
};

struct InputData {
    std::vector<int64_t> inputsIds;
    std::vector<int64_t> attentionMasks;
    std::vector<int64_t> wordsMasks;
    std::vector<int64_t> textLengths;
    std::vector<int64_t> spanIdxs;
    std::vector<uint8_t> spanMasksBool;
};

struct Span {
    int startIdx;
    int endIdx;
    std::string text;
    std::string classLabel;
    float prob;
};