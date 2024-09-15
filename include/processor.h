#pragma once

#include <tokenizers_cpp.h>

#include <vector>
#include <string>
#include <tuple>

#include "gliner_config.h"
#include "gliner_structs.h"
#include "tokenizer_utils.h"

using tokenizers::Tokenizer;

class Processor {
protected:
    Config config;
    WhitespaceTokenSplitter wordSplitter;
    const Tokenizer& tokenizer;
    Tokenizer& nonConstTokenizer;

public:
    Processor(const Config& config, const Tokenizer& tokenizer, const WhitespaceTokenSplitter& wordSplitter);

    std::tuple<std::vector<std::string>, std::vector<int>, std::vector<int>> tokenizeText(const std::string& text);
    std::tuple<std::vector<std::vector<std::string>>, std::vector<std::vector<int>>, std::vector<std::vector<int>>>
    batchTokenizeText(const std::vector<std::string>& texts);
    
    std::unordered_map<int, std::string> createClassMap(std::vector<std::string> classes);

    std::tuple<std::vector<std::vector<std::string>>, std::vector<int64_t>, std::vector<int64_t>>
    prepareTextInputs(const std::vector<std::vector<std::string>>& tokens, const std::vector<std::string>& entities);

    std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
    encodeInputs(const std::vector<std::vector<std::string>>& texts, const std::vector<int64_t>* promptLengths = nullptr);

    template<typename T>
    std::vector<std::vector<T>>& padArray(std::vector<std::vector<T>>& arr, const T& paddingValue = T());

    std::vector<std::vector<std::vector<int64_t>>>& padSpansArray(std::vector<std::vector<std::vector<int64_t>>>& arr);

    template<typename T>
    std::vector<typename std::remove_reference<typename std::remove_const<T>::type>::type> 
    flatten2DArray(const std::vector<std::vector<T>>& arr);

    template<typename T>
    std::vector<typename std::remove_reference<typename std::remove_const<T>::type>::type> 
    flatten3DArray(const std::vector<std::vector<std::vector<T>>>& arr);
};

class SpanProcessor : public Processor {
public:  
    SpanProcessor(const Config& config, const Tokenizer& tokenizer, const WhitespaceTokenSplitter& wordSplitter);
    
    std::tuple<std::vector<std::vector<std::vector<int64_t>>>, std::vector<std::vector<uint8_t>>>
    prepareSpans(const std::vector<int64_t>& textLengths, int64_t MaxWidth);

    BatchOutput prepareBatch(const std::vector<std::string>& texts, const std::vector<std::string>& entities);
};