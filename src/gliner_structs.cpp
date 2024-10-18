#include "GLiNER/gliner_structs.hpp"

using namespace gliner;

Batch::~Batch() {
    delete inputsShape;
    delete inputsIds;
    delete attentionMasks;
    delete wordsMasks;

    delete textLengths;
    delete textLengthsShape;
};

void TokenBatch::tensors(std::vector<Ort::Value>& tensors, const Ort::MemoryInfo& memory_info) {
    tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputsIds, inputsSize, inputsShape, 2));
    tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, attentionMasks, inputsSize, inputsShape, 2));
    tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, wordsMasks, inputsSize, inputsShape, 2));
    tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, textLengths, batchSize, textLengthsShape, 2));
}

TokenBatch::~TokenBatch() {};

int64_t TokenBatch::width() const {
    return numWords;
}

void SpanBatch::tensors(std::vector<Ort::Value>& tensors, const Ort::MemoryInfo& memory_info) {
    tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputsIds, inputsSize, inputsShape, 2));
    tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, attentionMasks, inputsSize, inputsShape, 2));
    tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, wordsMasks, inputsSize, inputsShape, 2));
    tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, textLengths, batchSize, textLengthsShape, 2));
    tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, spanIdxs, spanIdxsSize, spanIdxsShape, 3));
    tensors.push_back(Ort::Value::CreateTensor<bool>(memory_info, spanMasks, spanMasksSize, spanMasksShape, 2));
}

int64_t SpanBatch::width() const {
    return maxWidth;
}

SpanBatch::~SpanBatch() {
    delete spanIdxs;
    delete spanIdxsShape;
    delete spanMasksShape;
    delete spanMasks;
}