#include <tokenizers_cpp.h>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <cassert>
#include <chrono>
#include <fstream>
#include <type_traits>

#include "processor.h"

using tokenizers::Tokenizer;


Processor::Processor(const Config& config, const Tokenizer& tokenizer, const WhitespaceTokenSplitter& wordSplitter)
    : config(config), tokenizer(tokenizer), wordSplitter(wordSplitter), nonConstTokenizer(const_cast<Tokenizer&>(tokenizer)) {}

std::tuple<std::vector<std::string>, std::vector<int>, std::vector<int>> Processor::tokenizeText(const std::string& text) {
    std::vector<std::string> tokens;
    std::vector<int> wordStartIdx;
    std::vector<int> wordsEndIdx;
    for (const auto& [token, start, end] : wordSplitter.call(text)) {
        tokens.push_back(token);
        wordStartIdx.push_back(static_cast<int>(start));
        wordsEndIdx.push_back(static_cast<int>(end));
    }
    return {tokens, wordStartIdx, wordsEndIdx};
}

std::tuple<std::vector<std::vector<std::string>>, std::vector<std::vector<int>>, std::vector<std::vector<int>>>
Processor::batchTokenizeText(const std::vector<std::string>& texts) {
    std::vector<std::vector<std::string>> batchTokens;
    std::vector<std::vector<int>> batchWordsStartIdx;
    std::vector<std::vector<int>> batchWordsEndIdx;

    for (const auto& text : texts) {
        auto [tokens, wordsStartIdx, wordsEndIdx] = tokenizeText(text);
        batchTokens.push_back(tokens);
        batchWordsStartIdx.push_back(wordsStartIdx);
        batchWordsEndIdx.push_back(wordsEndIdx);
    }

    return {batchTokens, batchWordsStartIdx, batchWordsEndIdx};
}

std::unordered_map<int, std::string> Processor::createClassMap(std::vector<std::string> classes) {
    std::unordered_map<int, std::string> idToClass;
    for (size_t i = 0; i < classes.size(); i++) {
        std::string class_ = classes[i];
        idToClass[i + 1] = class_;
    }
    return idToClass;
}

std::tuple<std::vector<std::vector<std::string>>, std::vector<int64_t>, std::vector<int64_t>>
Processor::prepareTextInputs(const std::vector<std::vector<std::string>>& tokens, const std::vector<std::string>& entities) {
    std::vector<std::vector<std::string>> inputTexts;
    std::vector<int64_t> textLengths;
    std::vector<int64_t> promptLengths;

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::vector<std::string> currTokens = tokens[i];
        textLengths.push_back(tokens[i].size());
        std::vector<std::string> inputText;
        for (const auto& ent : entities) {
            inputText.push_back("<<ENT>>");
            inputText.push_back(ent);
        }
        inputText.push_back("<<SEP>>");
        int64_t promptLength = inputText.size();
        promptLengths.push_back(promptLength);
        for (size_t j = 0; j < currTokens.size(); j++) {
            inputText.push_back(currTokens[j]);
        }
        inputTexts.push_back(inputText);
    }

    return {inputTexts, textLengths, promptLengths};
}

std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
Processor::encodeInputs(const std::vector<std::vector<std::string>>& texts, const std::vector<int64_t>* promptLengths ) {
    std::vector<std::vector<int64_t>> wordsMasks;
    std::vector<std::vector<int64_t>> inputsIds;
    std::vector<std::vector<int64_t>> attentionMasks;

    for (size_t id = 0; id < texts.size(); ++id) {
        int64_t promptLength = promptLengths ? (*promptLengths)[id] : 0;
        const auto& tokenizedInputs = texts[id];

        std::vector<int64_t> wordsMask = {0};
        std::vector<int64_t> inputIds = {1};  // Assuming 1 is some initial token id (CLS token in BERT)
        std::vector<int64_t> attentionMask = {1};
        
        size_t wordId = 1;
        for (size_t tokenId = 0; tokenId < tokenizedInputs.size(); ++tokenId) {
            const auto& word = tokenizedInputs[tokenId];
            std::vector<int> wordTokens = nonConstTokenizer.Encode(word);

            for (size_t t = 0; t < wordTokens.size(); ++t) {
                attentionMask.push_back(1);
                if (tokenId < static_cast<size_t>(promptLength)) {
                    wordsMask.push_back(0);
                } else if (t == 0) {
                    wordsMask.push_back(wordId);
                    wordId++;
                } else {
                    wordsMask.push_back(0);
                }
                inputIds.push_back(wordTokens[t]);
            }
        }

        wordsMask.push_back(0);
        inputIds.push_back(2);  // Add separator token
        attentionMask.push_back(1);

        wordsMasks.push_back(wordsMask);
        inputsIds.push_back(inputIds);
        attentionMasks.push_back(attentionMask);
    }

    return {inputsIds, attentionMasks, wordsMasks};
}

template<typename T>
std::vector<std::vector<T>>& Processor::padArray(std::vector<std::vector<T>>& arr, const T& paddingValue) {
    size_t maxLength = 0;

    // Find the maximum length among the vectors
    for (const auto& vec : arr) {
        maxLength = std::max(maxLength, vec.size());
    }

    // Pad each sub-array to the maximum length
    for (auto& subArray : arr) {
        size_t add = maxLength - subArray.size();
        subArray.insert(subArray.end(), add, paddingValue); // Use insert to append padding efficiently
    }

    return arr; // Returning by reference, as the input array is modified in place
}

std::vector<std::vector<std::vector<int64_t>>>& Processor::padSpansArray(std::vector<std::vector<std::vector<int64_t>>>& arr) {
    size_t maxLength = 0;
    std::vector<int64_t> blank{0, 0};  // 2D vector to pad with

    // Find the maximum length among the sub-arrays (3D array's 2D slices)
    for (const auto& vec : arr) {
        maxLength = std::max(maxLength, vec.size());
    }

    // Pad each 2D slice to the maximum length
    for (auto& subArray : arr) {
        size_t add = maxLength - subArray.size();
        subArray.insert(subArray.end(), add, blank);  // Append padding (2D blank array)
    }

    return arr;  // Returning by reference, as the input array is modified in place
}

template<typename T>
std::vector<typename std::remove_reference<typename std::remove_const<T>::type>::type> 
Processor::flatten2DArray(const std::vector<std::vector<T>>& arr) {
    using ValueType = typename std::remove_reference<typename std::remove_const<T>::type>::type;
    std::vector<ValueType> result;
    
    // Calculate the total size of the flattened array
    size_t totalSize = 0;
    for (const auto& subArray : arr) {
        totalSize += subArray.size();
    }
    
    // Reserve space in the result vector to avoid multiple reallocations
    result.reserve(totalSize);
    
    // Flatten the 2D array
    for (const auto& subArray : arr) {
        result.insert(result.end(), subArray.begin(), subArray.end());
    }
    
    return result;
}

template<typename T>
std::vector<typename std::remove_reference<typename std::remove_const<T>::type>::type> 
Processor::flatten3DArray(const std::vector<std::vector<std::vector<T>>>& arr) {
    using ValueType = typename std::remove_reference<typename std::remove_const<T>::type>::type;
    std::vector<ValueType> result;
    
    // Calculate the total size of the flattened array
    size_t totalSize = 0;
    for (const auto& matrix : arr) {
        for (const auto& subArray : matrix) {
            totalSize += subArray.size();
        }
    }
    
    // Reserve space in the result vector to avoid multiple reallocations
    result.reserve(totalSize);
    
    // Flatten the 3D array
    for (const auto& matrix : arr) {
        for (const auto& subArray : matrix) {
            result.insert(result.end(), subArray.begin(), subArray.end());
        }
    }
    
    return result;
}


SpanProcessor::SpanProcessor(const Config& config, const Tokenizer& tokenizer, const WhitespaceTokenSplitter& wordSplitter)
        : Processor(config, tokenizer, wordSplitter) {}
            
std::tuple<std::vector<std::vector<std::vector<int64_t>>>, std::vector<std::vector<uint8_t>>>
SpanProcessor::prepareSpans(const std::vector<int64_t>& textLengths, int64_t MaxWidth) {
    std::vector<std::vector<std::vector<int64_t>>> spanIdxs;
    std::vector<std::vector<uint8_t>> spanMasks;

    for (int64_t textLength : textLengths) {
        std::vector<std::vector<int64_t>> spanIdx;
        std::vector<uint8_t> spanMask;

        for (int64_t i = 0; i < textLength; i++) {
            for (int64_t j = 0; j < MaxWidth; j++) {
                int64_t endIdx = std::min(i + j, textLength - 1);
                spanIdx.push_back({i, endIdx});
                spanMask.push_back(1);
            }
        }
        spanIdxs.push_back(spanIdx);
        spanMasks.push_back(spanMask);
    }
    return {spanIdxs, spanMasks};
}

BatchOutput SpanProcessor::prepareBatch(const std::vector<std::string>& texts, const std::vector<std::string>& entities) {
    auto [batchTokens, batchWordsStartIdx, batchWordsEndIdx] = batchTokenizeText(texts);
    auto idToClass = createClassMap(entities);
    auto [inputTokens, textLengths, promptLengths] = prepareTextInputs(batchTokens, entities);
    auto [inputsIds, attentionMasks, wordsMasks] = encodeInputs(inputTokens, &promptLengths);

    int64_t paddingValue = 0;
    padArray(inputsIds, paddingValue = 0);
    padArray(attentionMasks, paddingValue = 0);
    padArray(wordsMasks, paddingValue = 0);

    int64_t batchSize = inputsIds.size();
    int64_t numTokens = inputsIds[0].size();

    auto max_iterator = std::max_element(textLengths.begin(), textLengths.end());
    int64_t numWords = *max_iterator;;

    int64_t maxWidth = config.max_width;

    auto [spanIdxs, spanMasks] = prepareSpans(textLengths, config.max_width);
    padSpansArray(spanIdxs);
    padArray(spanMasks);

    BatchOutput output;
    output.batchSize = batchSize;
    output.numTokens = numTokens;
    output.numWords = numWords;
    output.maxWidth = maxWidth;
    output.inputsIds = flatten2DArray(inputsIds);
    output.attentionMasks = flatten2DArray(attentionMasks);
    output.wordsMasks = flatten2DArray(wordsMasks);
    output.textLengths = textLengths;
    output.spanIdxs = flatten3DArray(spanIdxs);
    output.spanMasks = flatten2DArray(spanMasks);
    output.idToClass = std::move(idToClass);
    output.batchTokens = std::move(batchTokens);
    output.batchWordsStartIdx = std::move(batchWordsStartIdx);
    output.batchWordsEndIdx = std::move(batchWordsEndIdx);

    return output;
}
