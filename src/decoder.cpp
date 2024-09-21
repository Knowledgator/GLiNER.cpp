#include <cmath>
#include <string>
#include <vector>

#include "GLiNER/decoder.hpp"

using namespace gliner;

float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

SpanDecoder::SpanDecoder(const Config& config) : config(config) {}

bool SpanDecoder::isNested(const Span& s1, const Span& s2) {
    return (s1.startIdx <= s2.startIdx && s2.endIdx <= s1.endIdx) || (s2.startIdx <= s1.startIdx && s1.endIdx <= s2.endIdx);
}

// Check for any overlap between two spans
bool SpanDecoder::hasOverlapping(const Span& s1, const Span& s2, bool multiLabel) {
    if (s1.startIdx == s2.startIdx && s1.endIdx == s2.endIdx) {
        return !multiLabel;
    }
    if (s1.startIdx > s2.endIdx || s2.startIdx > s1.endIdx) {
        return false;
    }
    return true;
}

// Check if spans overlap but are not nested inside each other
bool SpanDecoder::hasOverlappingNested(const Span& s1, const Span& s2, bool multiLabel) {
    return hasOverlapping(s1, s2, multiLabel) || isNested(s1, s2);
}

std::vector<Span> SpanDecoder::greedySearch(
    const std::vector<Span>& spans, bool flatNer, bool multiLabel
) { // expected sorted spans by start/end position
    if (spans.empty()) {
        return {};
    }

    std::function<bool(const Span&, const Span&, bool)> hasOv;
    if (flatNer) {
        hasOv = hasOverlapping;
    } else {
        hasOv = hasOverlappingNested;
    }

    std::vector<Span> newList;
    newList.reserve(spans.size());

    size_t prev = 0, next = 1;
    for (; next < spans.size(); next++) {
        if (!hasOv(spans[prev], spans[next], multiLabel)) {
            newList.push_back(spans[prev]);
            prev = next;
        } else {
            if (spans[prev].prob < spans[next].prob) { // get span with higher score on overlap
                prev = next;
            }
        }
    }
    newList.push_back(spans[prev]);
    return newList;
}

std::vector<std::vector<Span>> SpanDecoder::decode(
    const BatchOutput& batch,
    int maxWidth,
    const std::vector<std::string>& texts,
    const std::vector<std::string>& entities,
    const std::vector<float>& modelOutput,
    bool flatNer,
    float threshold,
    bool multiLabel
) {
    auto tokens = batch.batchTokens;
    int batchSize = batch.batchPrompts.size();
    int inputLength = batch.numWords;
    int numEntities = entities.size();

    int startTokenPadding = maxWidth * numEntities;
    int batchPadding = inputLength * startTokenPadding;
    int endTokenPadding = numEntities;

    std::vector<std::vector<Span>> spans(batchSize);
    // Process the model output
    for (size_t id = 0; id < modelOutput.size(); ++id) {
        float value = modelOutput[id];
        int batch = id / batchPadding;
        size_t startToken = (id / startTokenPadding) % inputLength;
        size_t endToken = startToken + ((id / endTokenPadding) % maxWidth);
        int entity = id % numEntities; // always one of entities
        float prob = sigmoid(value);
        
        if (prob >= threshold &&
            batch < batchSize &&
            startToken < tokens[batch].size() &&
            endToken < tokens[batch].size()) {

            Span span;
            span.startIdx = tokens[batch][startToken].start;
            span.endIdx = tokens[batch][endToken].end;
            std::string spanText = texts[batch].substr(span.startIdx, span.endIdx - span.startIdx);
            span.text = spanText;
            span.classLabel = entities[entity];
            span.prob = prob;

            spans[batch].push_back(span);
        }
    }    

    // Apply greedy search to each batch
    std::vector<std::vector<Span>> allSelectedSpans(batchSize);

    for (int batch = 0; batch < batchSize; ++batch) {
        allSelectedSpans[batch] = greedySearch(spans[batch], flatNer, multiLabel);
    }

    return allSelectedSpans;
}
