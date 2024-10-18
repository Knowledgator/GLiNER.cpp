#include <regex>

#include "GLiNER/processor.hpp"

using namespace gliner;

Processor::Processor(const Config& config, const std::string& tokenizer_path)
    : config(config), wordSplitter(WhitespaceTokenSplitter()) {
    const std::string blob = LoadBytesFromFile(tokenizer_path);
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);
}

std::vector<Token> Processor::tokenizeText(const std::string& text) {
    return wordSplitter.call(text);
}

std::vector<std::vector<Token>> Processor::batchTokenizeText(const std::vector<std::string>& texts) {
    std::vector<std::vector<Token>> res;
    res.reserve(texts.size());
    
    for (const auto& text : texts) {
        res.push_back(tokenizeText(text));
    }

    return res;
}

void Processor::prepareTextInputs(
    const std::vector<std::string>& entities,
    Batch* output,
    std::vector<Prompt>& prompts
) {
    std::vector<std::string> entities_prompt;
    entities_prompt.reserve(entities.size()*2+1);
    for (const auto& ent : entities) {
        entities_prompt.push_back("<<ENT>>");
        entities_prompt.push_back(ent);
    }
    entities_prompt.push_back("<<SEP>>");
    auto promptLength = entities_prompt.size();

    output->textLengths = new int64_t[output->batchSize];
    output->textLengthsShape = new int64_t[2]{output->batchSize, 1};
    output->numWords = 0;
    for (size_t i = 0; i < static_cast<size_t>(output->batchSize); ++i) {
        const std::vector<Token>& currTokens = output->batchTokens[i];
        std::vector<std::string> inputText;
        inputText.reserve(currTokens.size() + promptLength);
        inputText.insert(inputText.end(), entities_prompt.begin(), entities_prompt.end());
        for (auto t : currTokens) {
            inputText.push_back(t.text);
        }

        output->textLengths[i] = int64_t(currTokens.size());
        prompts.push_back({
            int64_t(currTokens.size()),
            int64_t(promptLength),
            inputText,
        });
        output->numWords = std::max(prompts[i].textLength, output->numWords);
    }
}

void Processor::encodeInputs(const std::vector<Prompt>& prompts, Batch* output) {    
    std::vector<std::vector<std::vector<int>>> tmp;
    tmp.reserve(prompts.size());

    output->numTokens = 0;
    for (const Prompt& p: prompts) {
        std::vector<std::vector<int>> pt;
        pt.reserve(p.prompt.size());

        
        int64_t s = 2; // padding tokens
        for (const std::string& word : p.prompt) {
            pt.push_back(tokenizer->Encode(word));
            s += pt.back().size();
        }
        tmp.push_back(pt);
        output->numTokens = std::max(output->numTokens, s);
    }

    output->inputsSize = output->numTokens*output->batchSize;
    output->inputsShape = new int64_t[2]{output->batchSize, output->numTokens};
    output->inputsIds = new int64_t[output->inputsSize]();
    output->attentionMasks = new int64_t[output->inputsSize]();
    output->wordsMasks = new int64_t[output->inputsSize]();

    for (size_t p = 0; p < tmp.size(); p++) {
        int64_t promptLength = prompts[p].promptLength;

        size_t idx = p * output->numTokens;
        output->inputsIds[idx] = 1; // initial token id
        output->attentionMasks[idx] = 1;
        idx++;

        for (size_t tokenId = 0, wordId = 1; tokenId < tmp[p].size(); ++tokenId) {
            const auto& word = tmp[p][tokenId];

            if (tokenId >= static_cast<size_t>(promptLength)) {
                output->wordsMasks[idx] = wordId;
                wordId++;
            }

            for (int t : word) {
                output->inputsIds[idx] = t;
                output->attentionMasks[idx] = 1;
                idx++;
            }
        }
        output->attentionMasks[idx] = 1;
        output->inputsIds[idx] = 2;
    }
}

SpanProcessor::SpanProcessor(const Config& config, const std::string& tokenizer_path)
    : Processor(config, tokenizer_path) {};

// SpanProcessor::SpanProcessor(const Config& config, Tokenizer& tokenizer, const WhitespaceTokenSplitter& wordSplitter)
//     : Processor(config, tokenizer, wordSplitter) {};

void SpanProcessor::prepareSpans(const std::vector<Prompt>& prompts, SpanBatch* output) {
    output->numSpans = output->numWords*output->maxWidth;

    output->spanIdxsSize = output->batchSize*output->numSpans*2;
    output->spanIdxs = new int64_t[output->spanIdxsSize]();
    output->spanIdxsShape = new int64_t[3]{output->batchSize, output->numSpans, 2};

    output->spanMasksSize = output->batchSize*output->numSpans;
    output->spanMasks = new bool[output->spanMasksSize]();
    output->spanMasksShape = new int64_t[2]{output->batchSize, output->numSpans};

    for (size_t p = 0; p < prompts.size(); p++) {
        for (int64_t i = 0; i < prompts[p].textLength; i++) { 
            int64_t m = std::min(output->maxWidth, prompts[p].textLength - i);
            for (int64_t j = 0; j < m; j++) {
                size_t idx = p*output->numSpans + i*output->maxWidth + j;
                output->spanIdxs[2*idx] = i;
                output->spanIdxs[2*idx+1] = i + j;
                output->spanMasks[idx] = 1;
            }
        }
    }
}

Batch* SpanProcessor::prepareBatch(
    const std::vector<std::string>& texts,
    const std::vector<std::string>& entities
) {
    SpanBatch* output = new SpanBatch;
    output->maxWidth = config.maxWidth;
    output->batchSize = texts.size();

    output->batchTokens = batchTokenizeText(texts);

    std::vector<Prompt> prompts;
    prompts.reserve(output->batchSize);
    prepareTextInputs(entities, output, prompts);
    encodeInputs(prompts, output);
    prepareSpans(prompts, output);
    return output;
}

TokenProcessor::TokenProcessor(const Config& config, const std::string& tokenizer_path)
    : Processor(config, tokenizer_path) {};

Batch* TokenProcessor::prepareBatch(
    const std::vector<std::string>& texts,
    const std::vector<std::string>& entities
) {
    TokenBatch* output = new TokenBatch;
    output->batchSize = texts.size();

    output->batchTokens = batchTokenizeText(texts);

    std::vector<Prompt> prompts;
    prompts.reserve(output->batchSize);
    prepareTextInputs(entities, output, prompts);
    encodeInputs(prompts, output);
    return output;
}