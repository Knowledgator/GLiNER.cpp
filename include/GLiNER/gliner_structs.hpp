#pragma once

#include <onnxruntime_cxx_api.h>

#include <vector>
#include <string>
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

    struct Batch {
        int64_t batchSize;
        int64_t numTokens;
        int64_t numWords;

        size_t inputsSize;
        int64_t* inputsIds;
        int64_t* attentionMasks;
        int64_t* wordsMasks;
        int64_t* inputsShape;

        int64_t* textLengths;
        int64_t* textLengthsShape;
        
        std::vector<std::vector<Token>> batchTokens;

        virtual ~Batch();
        virtual void tensors(std::vector<Ort::Value>& tensors, const Ort::MemoryInfo& memory_info) = 0;
        virtual int64_t width() const = 0;
    };

    struct TokenBatch : public Batch {
        virtual ~TokenBatch();
        virtual void tensors(std::vector<Ort::Value>& tensors, const Ort::MemoryInfo& memory_info);
        virtual int64_t width() const;
    };

    struct SpanBatch : public Batch {
        int64_t maxWidth;
        int64_t numSpans;
        int64_t spanIdxsSize;
        int64_t spanMasksSize;
        int64_t* spanIdxs;
        int64_t* spanIdxsShape;
        int64_t* spanMasksShape;
        bool* spanMasks;

        virtual void tensors(std::vector<Ort::Value>& tensors, const Ort::MemoryInfo& memory_info);
        virtual int64_t width() const;
        virtual ~SpanBatch();
    };

    struct Span {
        int startIdx;
        int endIdx;
        std::string text;
        std::string classLabel;
        float prob;
    };
}