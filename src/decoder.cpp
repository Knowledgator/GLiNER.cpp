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

    int batchPadding = inputLength * numEntities;
    int positionPadding = batchSize * batchPadding;
    int tokenPadding = numEntities;

    std::vector<std::vector<Span>> spans(batchSize);
    for (size_t start_id = 0; start_id < static_cast<size_t>(positionPadding); start_id++) {
        if (
            sigmoid(modelOutput[start_id]) < threshold 
        ) {
            continue;
        }
        
        size_t batch_id = (start_id / batchPadding) % batchSize;
        size_t startToken = (start_id / tokenPadding) % inputLength;
        int entity = start_id % numEntities; // always one of entities
        float score_sum = 0;
        int n = 0;
        for (
            size_t endToken = startToken, end_id = start_id + positionPadding; 
            (((end_id / batchPadding) % batchSize) == batch_id) && (end_id < static_cast<size_t>(2*positionPadding));
            endToken++, end_id += tokenPadding // span should contain same entity class
        ) {
            float score = sigmoid(modelOutput[end_id+positionPadding]);
            if (sigmoid(modelOutput[end_id]) < threshold) {
                continue;
            }
            if (score < threshold) {
                break; // fast exit without skipping entities
            }
            score_sum += score;
            ++n;

            Span span;
            span.startIdx = tokens[batch_id][startToken].start;
            span.endIdx = tokens[batch_id][endToken].end;
            span.classLabel = entities[entity];
            span.prob = score_sum / n;

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
