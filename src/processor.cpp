#include <tokenizers_cpp.h>
#include <onnxruntime_cxx_api.h>

#include <regex>
#include <string>
#include <vector>
#include <cassert>
#include <chrono>
#include <fstream>
#include <type_traits>

#include "GLiNER/processor.hpp"

using tokenizers::Tokenizer;
using namespace gliner;

SpanProcessor::SpanProcessor(const Config& config, Tokenizer& tokenizer, const WhitespaceTokenSplitter& wordSplitter)
    : config(config), wordSplitter(wordSplitter), nonConstTokenizer(tokenizer) {}

std::vector<Token> SpanProcessor::tokenizeText(const std::string& text) {
    return wordSplitter.call(text);
}

std::vector<std::vector<Token>> SpanProcessor::batchTokenizeText(const std::vector<std::string>& texts) {
    std::vector<std::vector<Token>> res;
    res.reserve(texts.size());
    
    for (const auto& text : texts) {
        res.push_back(tokenizeText(text));
    }

    return res;
}

std::vector<Prompt> SpanProcessor::prepareTextInputs(const std::vector<std::vector<Token>>& tokens, const std::vector<std::string>& entities) {
    std::vector<std::string> entities_prompt;
    for (const auto& ent : entities) {
        entities_prompt.push_back("<<ENT>>");
        entities_prompt.push_back(ent);
    }
    entities_prompt.push_back("<<SEP>>");
    auto promptLength = entities_prompt.size();

    std::vector<Prompt> prompts;
    prompts.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::vector<Token>& currTokens = tokens[i];
        std::vector<std::string> inputText;
        inputText.reserve(currTokens.size() + promptLength);
        inputText.insert(inputText.end(), entities_prompt.begin(), entities_prompt.end());
        for (auto t : currTokens) {
            inputText.push_back(t.text);
        }

        prompts.push_back({
            int64_t(currTokens.size()),
            int64_t(promptLength),
            inputText,
        });
    }

    return prompts;
}

void SpanProcessor::encodeInputs(const std::vector<Prompt>& prompts, BatchOutput& output) {    
    std::vector<std::vector<std::vector<int>>> tmp;
    tmp.reserve(prompts.size());

    output.numTokens = 0;
    for (const Prompt& p: prompts) {
        std::vector<std::vector<int>> pt;
        pt.reserve(p.prompt.size());
        
        int64_t s = 2; // padding tokens
        for (const std::string& word : p.prompt) {
            pt.push_back(nonConstTokenizer.Encode(word));
            s += pt.back().size();
        }
        tmp.push_back(pt);
        output.numTokens = std::max(output.numTokens, s);
    }

    size_t l = output.numTokens*prompts.size();

    output.inputsIds = std::vector<int64_t>(l, 0);
    output.attentionMasks = std::vector<int64_t>(l, 0);
    output.wordsMasks = std::vector<int64_t>(l, 0);

    for (size_t p = 0; p < tmp.size(); p++) {
        int64_t promptLength = prompts[p].promptLength;

        size_t idx = p * output.numTokens;
        output.inputsIds[idx] = 1; // initial token id
        output.attentionMasks[idx] = 1;
        idx++;

        for (size_t tokenId = 0, wordId = 1; tokenId < tmp[p].size(); ++tokenId) {
            const auto& word = tmp[p][tokenId];

            if (tokenId >= static_cast<size_t>(promptLength)) {
                output.wordsMasks[idx] = wordId;
                wordId++;
            }

            for (int t : word) {
                output.inputsIds[idx] = t;
                output.attentionMasks[idx] = 1;
                idx++;
            }
        }
        output.attentionMasks[idx] = 1;
        output.inputsIds[idx] = 2;
    }
}

void SpanProcessor::prepareSpans(const std::vector<Prompt>& prompts, int64_t MaxWidth, int64_t numWords, BatchOutput& output) {
    size_t l = numWords * MaxWidth;

    output.spanIdxs = std::vector<int64_t> (l*prompts.size()*2, 0);
    output.spanMasks = std::vector<uint8_t>(l*prompts.size(), 0);

    for (size_t p = 0; p < prompts.size(); p++) {
        for (int64_t i = 0; i < prompts[p].textLength; i++) { 
            int64_t m = std::min(MaxWidth, prompts[p].textLength - i);
            for (int64_t j = 0; j < m; j++) {
                size_t idx = p*l + i*MaxWidth + j;
                output.spanIdxs[2*idx] = i;
                output.spanIdxs[2*idx+1] = i + j;
                output.spanMasks[idx] = 1;
            }
        }
    }
}

BatchOutput SpanProcessor::prepareBatch(const std::vector<std::string>& texts, const std::vector<std::string>& entities) {
    auto batchTokens = batchTokenizeText(texts);
    auto prompts = prepareTextInputs(batchTokens, entities);

    int64_t numWords = 0;
    for (const Prompt& p: prompts) {
        numWords = std::max(p.textLength, numWords);
    }
    int64_t maxWidth = config.max_width;

    BatchOutput output;
    encodeInputs(prompts, output);
    prepareSpans(prompts, maxWidth, numWords, output);

    output.maxWidth = maxWidth;
    output.numWords = numWords;
    output.batchSize = texts.size();
    output.batchTokens = std::move(batchTokens);
    output.batchPrompts = std::move(prompts);
    return output;
}
