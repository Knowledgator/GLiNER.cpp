#include <iostream>
#include <cmath>
#include <string>
#include <vector>

#include "decoder.h"

float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

Decoder::Decoder(const Config& config) :
            config(config) {}

bool Decoder::isNested(const std::vector<int>& idx1, const std::vector<int>& idx2) {
    return (idx1[0] <= idx2[0] && idx1[1] >= idx2[1]) || (idx2[0] <= idx1[0] && idx2[1] >= idx1[1]);
}

// Check for any overlap between two spans
bool Decoder::hasOverlapping(const std::vector<int>& idx1, const std::vector<int>& idx2, bool multiLabel) {
    if (std::vector<int>(idx1.begin(), idx1.begin() + 2) == std::vector<int>(idx2.begin(), idx2.begin() + 2)) {
        return !multiLabel;
    }
    if (idx1[0] > idx2[1] || idx2[0] > idx1[1]) {
        return false;
    }
    return true;
}

// Check if spans overlap but are not nested inside each other
bool Decoder::hasOverlappingNested(const std::vector<int>& idx1, const std::vector<int>& idx2, bool multiLabel) {
    if (std::vector<int>(idx1.begin(), idx1.begin() + 2) == std::vector<int>(idx2.begin(), idx2.begin() + 2)) {
        return !multiLabel;
    }
    if (idx1[0] > idx2[1] || idx2[0] > idx1[1] || isNested(idx1, idx2)) {
        return false;
    }
    return true;
}

std::vector<std::vector<int>> Decoder::greedySearch(const std::vector<std::vector<int>>& spans,
                                            const std::vector<float>& scores,
                                            bool flatNer, bool multiLabel) {
    std::function<bool(const std::vector<int>&, const std::vector<int>&)> hasOv;
    if (flatNer) {
        hasOv = [this, multiLabel](const std::vector<int>& idx1, const std::vector<int>& idx2) {
            return this->hasOverlapping(idx1, idx2, multiLabel);;
        };
    } else {
        hasOv = [this, multiLabel](const std::vector<int>& idx1, const std::vector<int>& idx2) {
            return this->hasOverlappingNested(idx1, idx2, multiLabel);
        };
    }

    // Create a vector of indices
    std::vector<size_t> indices(spans.size());
    for (size_t i = 0; i < spans.size(); ++i) {
        indices[i] = i;
    }

    // Sort indices based on corresponding scores in descending order
    std::sort(indices.begin(), indices.end(), [&scores](size_t a, size_t b) {
        return scores[a] > scores[b];
    });

    std::vector<std::vector<int>> newList;

    for (size_t idx : indices) {
        const auto& b = spans[idx];
        bool flag = false;
        for (const auto& newSpan : newList) {
            if (hasOv(b, newSpan)) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            newList.push_back(b);
        }
    }

    // Sort by start position
    std::sort(newList.begin(), newList.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        return a[0] < b[0];
    });

    return newList;
}



SpanDecoder::SpanDecoder(const Config& config) : Decoder(config) {}

std::vector<std::vector<Span>> SpanDecoder::decode(
    int batchSize,
    int inputLength,
    int maxWidth,
    int numEntities,
    const std::vector<std::string>& texts,
    const std::vector<std::vector<int>>& batchWordsStartIdx,
    const std::vector<std::vector<int>>& batchWordsEndIdx,
    const std::unordered_map<int, std::string>& idToClass,
    const std::vector<float>& modelOutput,
    bool flatNer,
    float threshold,
    bool multiLabel
) {
    // Initialize spans for each batch
    std::vector<std::vector<Span>> spans(batchSize);

    int batchPadding = inputLength * maxWidth * numEntities;
    int startTokenPadding = maxWidth * numEntities;
    int endTokenPadding = numEntities;

    // Process the model output
    for (size_t id = 0; id < modelOutput.size(); ++id) {
        float value = modelOutput[id];
        int batch = id / batchPadding;
        int startToken = (id / startTokenPadding) % inputLength;
        int endToken = startToken + ((id / endTokenPadding) % maxWidth);
        int entity = id % numEntities;
        float prob = sigmoid(value);
        
        // std::cout<<prob<<std::endl;

        if (prob >= threshold &&
            batch < batchSize &&
            startToken < batchWordsStartIdx[batch].size() &&
            endToken < batchWordsEndIdx[batch].size()) {

            Span span;
            span.startIdx = batchWordsStartIdx[batch][startToken];
            span.endIdx = batchWordsEndIdx[batch][endToken];
            std::string spanText = texts[batch].substr(span.startIdx, span.endIdx - span.startIdx + 1);
            span.text = spanText;

            // Adjusting for 1-based indexing in idToClass
            auto it = idToClass.find(entity + 1);
            if (it != idToClass.end()) {
                span.classLabel = it->second;
            } else {
                span.classLabel = "Unknown";
            }
            span.prob = prob;

            spans[batch].push_back(span);
        }
    }    

    // Apply greedy search to each batch
    std::vector<std::vector<Span>> allSelectedSpans(batchSize);

    for (int batch = 0; batch < batchSize; ++batch) {
        const std::vector<Span>& resI = spans[batch];

        // Extract spans and scores for greedySearch
        std::vector<std::vector<int>> spansForGreedySearch;
        std::vector<float> scoresForGreedySearch;

        for (const Span& s : resI) {
            spansForGreedySearch.push_back({s.startIdx, s.endIdx});
            scoresForGreedySearch.push_back(s.prob);
        }

        // Call greedySearch
        std::vector<std::vector<int>> selectedSpansIndices = Decoder::greedySearch(spansForGreedySearch, scoresForGreedySearch, flatNer, multiLabel);

        // Reconstruct selected Spans
        std::vector<Span> selectedSpans;

        for (const auto& indices : selectedSpansIndices) {
            int start = indices[0];
            int end = indices[1];

            // Find the Span in resI with matching start and end indices
            for (const Span& s : resI) {
                if (s.startIdx == start && s.endIdx == end) {
                    selectedSpans.push_back(s);
                    break;  // Assuming no duplicates
                }
            }
        }
        allSelectedSpans[batch] = selectedSpans;
    }

    return allSelectedSpans;
}
