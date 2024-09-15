#pragma once

#include <vector>
#include <string>
#include <regex>

#include "gliner_config.h"
#include "gliner_structs.h"

class Decoder {
protected:
    Config config;

public:
    Decoder(const Config& config);
    
    virtual std::vector<std::vector<int>> greedySearch(const std::vector<std::vector<int>>& spans,
                                                       const std::vector<float>& scores,
                                                       bool flatNer = true, bool multiLabel = false);

    bool isNested(const std::vector<int>& idx1, const std::vector<int>& idx2);
    bool hasOverlapping(const std::vector<int>& idx1, const std::vector<int>& idx2, bool multiLabel = false);
    bool hasOverlappingNested(const std::vector<int>& idx1, const std::vector<int>& idx2, bool multiLabel = false);
};

class SpanDecoder : public Decoder {
public:
    SpanDecoder(const Config& config);

    std::vector<std::vector<Span>> decode(
        int batchSize,
        int inputLength,
        int maxWidth,
        int numEntities,
        const std::vector<std::string>& texts,
        const std::vector<std::vector<int>>& batchWordsStartIdx,
        const std::vector<std::vector<int>>& batchWordsEndIdx,
        const std::unordered_map<int, std::string>& idToClass,
        const std::vector<float>& modelOutput,
        bool flatNer = false,
        float threshold = 0.5,
        bool multiLabel = false
    );
};