#include <algorithm>
#include <cmath>

#include "GLiNER/decoder.hpp"

using namespace gliner;

float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

static bool adjustSpanToTextBounds(const Span& span, size_t textLength) {
    // Check for negative indices (overflow to large positive values)
    if (span.startIdx < 0 || span.endIdx < 0) {
        return false;
    }

    // Check that indices don't exceed text length
    if (static_cast<size_t>(span.startIdx) >= textLength ||
        static_cast<size_t>(span.endIdx) > textLength) {
        return false;
    }

    // Check that startIdx <= endIdx
    if (span.startIdx > span.endIdx) {
        return false;
    }

    return true;
}

static std::string safeCopySpanText(const Span& span, const std::string& text) {
    // First validate span bounds
    if (!adjustSpanToTextBounds(span, text.length())) {
        return "";
    }

    // Double-check bounds before substr call
    size_t startPos = static_cast<size_t>(span.startIdx);
    size_t endPos = static_cast<size_t>(span.endIdx);

    if (startPos >= text.length() || endPos > text.length() || startPos > endPos) {
        return "";
    }

    return text.substr(startPos, endPos - startPos);
}

bool Decoder::isNested(const Span& s1, const Span& s2) {
    return (s1.startIdx <= s2.startIdx && s2.endIdx <= s1.endIdx) || (s2.startIdx <= s1.startIdx && s1.endIdx <= s2.endIdx);
}

// Check for any overlap between two spans
bool Decoder::hasOverlapping(const Span& s1, const Span& s2, bool multiLabel) {
    if (s1.startIdx == s2.startIdx && s1.endIdx == s2.endIdx) {
        return !multiLabel;
    }
    if (s1.startIdx > s2.endIdx || s2.startIdx > s1.endIdx) {
        return false;
    }
    return true;
}

// Check if spans overlap but are not nested inside each other
bool Decoder::hasOverlappingNested(const Span& s1, const Span& s2, bool multiLabel) {
    return hasOverlapping(s1, s2, multiLabel) || isNested(s1, s2);
}

std::vector<Span> Decoder::greedySearch(
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

    std::vector<Span> sortedSpans = spans;
    std::sort(sortedSpans.begin(), sortedSpans.end(), [](const Span& a, const Span& b) {
        return a.prob > b.prob;
    });

    std::vector<Span> newList;
    newList.reserve(spans.size());

    for (const auto& span : sortedSpans) {
        bool overlapDetected = false;
        for (const auto& selected : newList) {
            if (hasOv(span, selected, multiLabel)) {
                overlapDetected = true;
                break;
            }
        }

        if (!overlapDetected) {
            newList.push_back(span);
        }
    }

    std::sort(newList.begin(), newList.end(), [](const Span& a, const Span& b) {
        return a.startIdx < b.startIdx;
    });

    return newList;
}

std::vector<std::vector<Span>> Decoder::batchGreedySearch(
    const std::vector<std::vector<Span>>& spans_batch, bool flatNer, bool multiLabel
) { // expected sorted spans by start/end position in batches
    // Apply greedy search to each batch
    std::vector<std::vector<Span>> allSelectedSpans;
    allSelectedSpans.reserve(spans_batch.size());
    
    for (const auto& batch : spans_batch) {
        allSelectedSpans.push_back(greedySearch(batch, flatNer, multiLabel));
    }
    return allSelectedSpans;
}

std::vector<std::vector<Span>> SpanDecoder::decode(
    const Batch* batch,
    const std::vector<std::string>& texts,
    const std::vector<std::string>& entities,
    const std::vector<float>& modelOutput,
    bool flatNer,
    float threshold,
    bool multiLabel
) {
    auto tokens = batch->batchTokens;
    int batchSize = batch->batchSize;
    int inputLength = batch->numWords;
    int numEntities = entities.size();

    int startTokenPadding = batch->width() * numEntities;
    int batchPadding = inputLength * startTokenPadding;
    int endTokenPadding = numEntities;

    std::vector<std::vector<Span>> spans(batchSize);
    // Process the model output
    for (size_t id = 0; id < modelOutput.size(); ++id) {
        float value = modelOutput[id];
        int batch_id = id / batchPadding;
        size_t startToken = (id / startTokenPadding) % inputLength;
        size_t endToken = startToken + ((id / endTokenPadding) % batch->width());
        int entity = id % numEntities; // always one of entities
        float prob = sigmoid(value);
        
        if (prob >= threshold &&
            batch_id < batchSize &&
            startToken < tokens[batch_id].size() &&
            endToken < tokens[batch_id].size()) {

            Span span;
            span.startIdx = tokens[batch_id][startToken].start;
            span.endIdx = tokens[batch_id][endToken].end;
            span.classLabel = entities[entity];
            span.prob = prob;

            // Safely extract span text with bounds checking
            span.text = safeCopySpanText(span, texts[batch_id]);

            // Skip spans with invalid indices that couldn't extract text
            if (!span.text.empty()) {
                spans[batch_id].push_back(span);
            }
        }
    }

    return batchGreedySearch(spans, flatNer, multiLabel);
}

std::vector<std::vector<Span>> TokenDecoder::decode(
    const Batch* batch,
    const std::vector<std::string>& texts,
    const std::vector<std::string>& entities,
    const std::vector<float>& modelOutput,
    bool flatNer,
    float threshold,
    bool multiLabel
) {
    auto tokens = batch->batchTokens;
    int batchSize = batch->batchSize;
    int inputLength = batch->numWords;
    int numEntities = entities.size();

    auto outputAt = [&](int batchId, int tokenId, int entityId, int channel) -> float {
        size_t idx = (
            ((static_cast<size_t>(batchId) * inputLength + tokenId) * numEntities + entityId) * 3 + channel
        );
        return modelOutput[idx];
    };

    std::vector<std::vector<Span>> spans(batchSize);

    for (int batchId = 0; batchId < batchSize; ++batchId) {
        for (int entityId = 0; entityId < numEntities; ++entityId) {
            for (int startToken = 0; startToken < inputLength; ++startToken) {
                float startScore = sigmoid(outputAt(batchId, startToken, entityId, 0));
                if (startScore < threshold) {
                    continue;
                }

                for (int endToken = startToken; endToken < inputLength; ++endToken) {
                    float endScore = sigmoid(outputAt(batchId, endToken, entityId, 1));
                    if (endScore < threshold) {
                        continue;
                    }

                    float spanScore = std::min(startScore, endScore);
                    bool validInside = true;
                    for (int tokenId = startToken; tokenId <= endToken; ++tokenId) {
                        float insideScore = sigmoid(outputAt(batchId, tokenId, entityId, 2));
                        if (insideScore < threshold) {
                            validInside = false;
                            break;
                        }
                        spanScore = std::min(spanScore, insideScore);
                    }

                    if (!validInside ||
                        startToken >= static_cast<int>(tokens[batchId].size()) ||
                        endToken >= static_cast<int>(tokens[batchId].size())) {
                        continue;
                    }

                    Span span;
                    span.startIdx = tokens[batchId][startToken].start;
                    span.endIdx = tokens[batchId][endToken].end;
                    span.classLabel = entities[entityId];
                    span.prob = spanScore;
                    span.text = safeCopySpanText(span, texts[batchId]);

                    if (!span.text.empty()) {
                        spans[batchId].push_back(span);
                    }
                }
            }
        }
    }

    return batchGreedySearch(spans, flatNer, multiLabel);
}
