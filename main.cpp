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

using tokenizers::Tokenizer;

void printInputTensorSample(const std::vector<Ort::Value>& input_tensors) {
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        auto tensor_info = input_tensors[i].GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        auto type = tensor_info.GetElementType();

        std::cout << "Input tensor " << i << ":" << std::endl;
        std::cout << "  Shape: ";
        for (auto dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;

        std::cout << "  Data sample: ";
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            auto data = input_tensors[i].GetTensorData<int64_t>();
            for (size_t j = 0; j < std::min(static_cast<size_t>(10), tensor_info.GetElementCount()); ++j) {
                std::cout << data[j] << " ";
            }
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            auto data = input_tensors[i].GetTensorData<bool>();
            for (size_t j = 0; j < std::min(static_cast<size_t>(10), tensor_info.GetElementCount()); ++j) {
                std::cout << (data[j] ? "true " : "false ");
            }
        }
        std::cout << std::endl;
    }
}

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

class WhitespaceTokenSplitter {
private:
    std::regex whitespacePattern;

public:
    WhitespaceTokenSplitter() : whitespacePattern(R"(\w+(?:[-_]\w+)*|\S)") {}

    std::vector<std::tuple<std::string, size_t, size_t>> call(const std::string& text) {
        std::vector<std::tuple<std::string, size_t, size_t>> tokens;
        std::smatch match;
        std::string::const_iterator searchStart(text.cbegin());

        while (std::regex_search(searchStart, text.cend(), match, whitespacePattern)) {
            size_t start = match.position() + std::distance(text.cbegin(), searchStart);
            size_t end = start + match.length();
            tokens.push_back(std::make_tuple(match.str(), start, end));
            searchStart = match.suffix().first;
        }

        return tokens;
    }
};

struct Config {
    int max_width;
    int max_length;
};

struct BatchOutput {
    int64_t batchSize;
    int64_t numTokens;
    int64_t numWords;
    int64_t maxWidth;
    std::vector<int64_t> inputsIds;
    std::vector<int64_t> attentionMasks;
    std::vector<int64_t> wordsMasks;
    std::vector<int64_t> textLengths;
    std::vector<int64_t> spanIdxs;
    std::vector<int64_t> spanMasks;
    std::unordered_map<int, std::string> idToClass;
    std::vector<std::vector<std::string>> batchTokens;
    std::vector<std::vector<int>> batchWordsStartIdx;
    std::vector<std::vector<int>> batchWordsEndIdx;
};

struct InputData {
    std::vector<int64_t> inputsIds;
    std::vector<int64_t> attentionMasks;
    std::vector<int64_t> wordsMasks;
    std::vector<int64_t> textLengths;
    std::vector<int64_t> spanIdxs;
    std::vector<int> spanMasksBool;
};

struct Span {
    int startIdx;
    int endIdx;
    std::string classLabel;
    double prob;
};

class Processor {
protected:
    Config config;
    WhitespaceTokenSplitter wordSplitter;
    const Tokenizer& tokenizer;
    Tokenizer& nonConstTokenizer;

public:
    Processor(const Config& config, const Tokenizer& tokenizer, const WhitespaceTokenSplitter& wordSplitter)
        : config(config), tokenizer(tokenizer), wordSplitter(wordSplitter), nonConstTokenizer(const_cast<Tokenizer&>(tokenizer)) {}

    std::tuple<std::vector<std::string>, std::vector<int>, std::vector<int>> tokenizeText(const std::string& text) {
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
    batchTokenizeText(const std::vector<std::string>& texts) {
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

    std::unordered_map<int, std::string> createClassMap(std::vector<std::string> classes) {
        std::unordered_map<int, std::string> idToClass;
        for (size_t i = 0; i < classes.size(); i++) {
            std::string class_ = classes[i];
            idToClass[i + 1] = class_;
        }
        return idToClass;
    }

    std::tuple<std::vector<std::vector<std::string>>, std::vector<int64_t>, std::vector<int64_t>>
    prepareTextInputs(const std::vector<std::vector<std::string>>& tokens, const std::vector<std::string>& entities) {
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
    encodeInputs(const std::vector<std::vector<std::string>>& texts, const std::vector<int64_t>* promptLengths = nullptr) {
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
    std::vector<std::vector<T>>& padArray(std::vector<std::vector<T>>& arr, const T& paddingValue = T()) {
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

    std::vector<std::vector<std::vector<int64_t>>>& padSpansArray(std::vector<std::vector<std::vector<int64_t>>>& arr) {
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
    flatten2DArray(const std::vector<std::vector<T>>& arr) {
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
    flatten3DArray(const std::vector<std::vector<std::vector<T>>>& arr) {
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

};

class SpanProcessor : public Processor {
public:  
    SpanProcessor(const Config& config, const Tokenizer& tokenizer, const WhitespaceTokenSplitter& wordSplitter)
        : Processor(config, tokenizer, wordSplitter) {}
            
    std::tuple<std::vector<std::vector<std::vector<int64_t>>>, std::vector<std::vector<int64_t>>>
    prepareSpans(const std::vector<int64_t>& textLengths, int64_t MaxWidth) {
        std::vector<std::vector<std::vector<int64_t>>> spanIdxs;
        std::vector<std::vector<int64_t>> spanMasks;

        for (int64_t textLength : textLengths) {
            std::vector<std::vector<int64_t>> spanIdx;
            std::vector<int64_t> spanMask;

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

    BatchOutput prepareBatch(const std::vector<std::string>& texts, const std::vector<std::string>& entities) {
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
};

// decoder
bool isNested(const std::vector<int>& idx1, const std::vector<int>& idx2) {
    return (idx1[0] <= idx2[0] && idx1[1] >= idx2[1]) || (idx2[0] <= idx1[0] && idx2[1] >= idx1[1]);
}

// Check for any overlap between two spans
bool hasOverlapping(const std::vector<int>& idx1, const std::vector<int>& idx2, bool multiLabel = false) {
    if (std::vector<int>(idx1.begin(), idx1.begin() + 2) == std::vector<int>(idx2.begin(), idx2.begin() + 2)) {
        return !multiLabel;
    }
    if (idx1[0] > idx2[1] || idx2[0] > idx1[1]) {
        return false;
    }
    return true;
}

// Check if spans overlap but are not nested inside each other
bool hasOverlappingNested(const std::vector<int>& idx1, const std::vector<int>& idx2, bool multiLabel = false) {
    if (std::vector<int>(idx1.begin(), idx1.begin() + 2) == std::vector<int>(idx2.begin(), idx2.begin() + 2)) {
        return !multiLabel;
    }
    if (idx1[0] > idx2[1] || idx2[0] > idx1[1] || isNested(idx1, idx2)) {
        return false;
    }
    return true;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}


class Decoder {
    protected:
        Config config;
    
    public:
        Decoder(const Config& config) :
            config(config) {}
        
    std::vector<std::vector<int>> greedySearch(const std::vector<std::vector<int>>& spans,
                                            const std::vector<double>& scores,
                                            bool flatNer = true, bool multiLabel = false) {
        std::function<bool(const std::vector<int>&, const std::vector<int>&)> hasOv;
        if (flatNer) {
            hasOv = [multiLabel](const std::vector<int>& idx1, const std::vector<int>& idx2) {
                return hasOverlapping(idx1, idx2, multiLabel);;
            };
        } else {
            hasOv = [multiLabel](const std::vector<int>& idx1, const std::vector<int>& idx2) {
                return hasOverlappingNested(idx1, idx2, multiLabel);
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
};


class SpanDecoder : public Decoder {
    public:
        SpanDecoder(const Config& config) : Decoder(config) {}

    std::vector<std::vector<Span>> decode(
        int batchSize,
        int inputLength,
        int maxWidth,
        int numEntities,
        const std::vector<std::vector<int>>& batchWordsStartIdx,
        const std::vector<std::vector<int>>& batchWordsEndIdx,
        const std::unordered_map<int, std::string>& idToClass,
        const std::vector<float>& modelOutput,
        bool flatNer = false,
        double threshold = 0.5,
        bool multiLabel = false
    ) {
        // Initialize spans for each batch
        std::vector<std::vector<Span>> spans(batchSize);

        int batchPadding = inputLength * maxWidth * numEntities;
        int startTokenPadding = maxWidth * numEntities;
        int endTokenPadding = numEntities;

        // Process the model output
        for (size_t id = 0; id < modelOutput.size(); ++id) {
            double value = modelOutput[id];
            int batch = id / batchPadding;
            int startToken = (id / startTokenPadding) % inputLength;
            int endToken = startToken + ((id / endTokenPadding) % maxWidth);
            int entity = id % numEntities;
            double prob = sigmoid(value);

            if (prob >= threshold &&
                batch < batchSize &&
                startToken < batchWordsStartIdx[batch].size() &&
                endToken < batchWordsEndIdx[batch].size()) {

                Span span;
                span.startIdx = batchWordsStartIdx[batch][startToken];
                span.endIdx = batchWordsEndIdx[batch][endToken];
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
            std::vector<double> scoresForGreedySearch;

            for (const Span& s : resI) {
                spansForGreedySearch.push_back({s.startIdx, s.endIdx});
                scoresForGreedySearch.push_back(s.prob);
            }

            // Call greedySearch
            std::vector<std::vector<int>> selectedSpansIndices = greedySearch(spansForGreedySearch, scoresForGreedySearch, flatNer, multiLabel);

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
};

class Model {
protected:
    const char* model_path;
    Config config;
    SpanProcessor processor;
    SpanDecoder decoder;
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;

public:
    Model(const char* path, Config config, SpanProcessor proc, SpanDecoder decoder) :
        model_path(path), config(config), processor(proc), decoder(decoder),
        env(ORT_LOGGING_LEVEL_WARNING, "test"), session_options(),
        session(env, model_path, session_options) {}

    void prepareInput(const BatchOutput& batch, std::vector<Ort::Value>& input_tensors, InputData& inputData) {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> shape = {batch.batchSize, batch.numTokens};

        inputData.inputsIds = batch.inputsIds;
        inputData.attentionMasks = batch.attentionMasks;
        inputData.wordsMasks = batch.wordsMasks;

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.inputsIds.data(), inputData.inputsIds.size(), shape.data(), shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.attentionMasks.data(), inputData.attentionMasks.size(), shape.data(), shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.wordsMasks.data(), inputData.wordsMasks.size(), shape.data(), shape.size()));
        
        std::vector<int64_t> text_lengths_shape = {batch.batchSize, 1};

        inputData.textLengths = batch.textLengths;

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.textLengths.data(), inputData.textLengths.size(), text_lengths_shape.data(), text_lengths_shape.size()));

        int64_t numSpans = batch.numWords*batch.maxWidth;

        std::vector<int64_t> span_shape = {batch.batchSize, numSpans, 2};
        std::vector<int64_t> span_mask_shape = {batch.batchSize, numSpans};

        inputData.spanIdxs = batch.spanIdxs;

        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, inputData.spanIdxs.data(), inputData.spanIdxs.size(), span_shape.data(), span_shape.size()));
        
        inputData.spanMasksBool.reserve(batch.spanMasks.size());
        for (const auto& mask : batch.spanMasks) {
            inputData.spanMasksBool.push_back(mask != 0);
        };

        input_tensors.push_back(Ort::Value::CreateTensor<bool>(memory_info, reinterpret_cast<bool*>(inputData.spanMasksBool.data()), inputData.spanMasksBool.size(), span_mask_shape.data(), span_mask_shape.size()));
    }

    std::vector<std::vector<Span>> inference(const std::vector<std::string>& texts, const std::vector<std::string>& entities,
                                            bool flatNer = false,
                                            double threshold = 0.5,
                                            bool multiLabel = false) {
        BatchOutput batch = processor.prepareBatch(texts, entities);
        std::vector<Ort::Value> input_tensors;
        InputData inputData;

        prepareInput(batch, input_tensors, inputData);

        std::vector<const char*> input_names = {"input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"};
        std::vector<const char*> output_names = {"logits"}; // Adjust based on your model's output
        
        printInputTensorSample(input_tensors);

        std::vector<Ort::Value> modelOutputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

        Ort::Value& output_tensor = modelOutputs[0];
        Ort::TensorTypeAndShapeInfo output_info = output_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = output_info.GetShape();

        // Compute total number of elements
        size_t total_elements = 1;
        for (size_t i = 0; i < output_shape.size(); ++i) {
            total_elements *= output_shape[i];
        }

        const float* output_data = output_tensor.GetTensorData<float>();
        std::vector<float> modelOutput(output_data, output_data + total_elements);

        int batchSize = batch.inputsIds.size();
        int maxWidth = config.max_width;
        auto max_iterator = std::max_element(batch.textLengths.begin(), batch.textLengths.end());
        int inputLength = *max_iterator;
        int numEntities = entities.size();

        std::vector<std::vector<Span>>  outputSpans = decoder.decode(
            batchSize, inputLength, maxWidth, numEntities, 
            batch.batchWordsStartIdx, batch.batchWordsEndIdx, 
            batch.idToClass,
            modelOutput,
            flatNer, threshold, multiLabel
        );
        return outputSpans;
    }

};



void testWithMinimalInput() {
    const char* model_path = "model.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);
    
    std::vector<int64_t> input_ids = {128003, 2, 3, 4, 5};
    std::vector<int64_t> attention_mask = {1, 1, 1, 1, 1};
    std::vector<int64_t> words_mask = {1, 0, 0, 0, 0};
    std::vector<int64_t> text_lengths = {1};
    std::vector<int64_t> span_idx = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<bool> span_mask = {true, true, true, true, true, true, true, true, true, true, true, false};

    std::vector<int64_t> shape1 = {1, 5};
    std::vector<int64_t> shape2 = {1, 1};
    std::vector<int64_t> shape3 = {1, 12, 2};
    std::vector<int64_t> shape4 = {1, 12};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(), input_ids.size(), shape1.data(), shape1.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask.data(), attention_mask.size(), shape1.data(), shape1.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, words_mask.data(), words_mask.size(), shape1.data(), shape1.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, text_lengths.data(), text_lengths.size(), shape2.data(), shape2.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, span_idx.data(), span_idx.size(), shape3.data(), shape3.size()));

    std::vector<int> spanMasksBool;
    spanMasksBool.reserve(span_mask.size());
    for (const auto& mask : span_mask) {
        spanMasksBool.push_back(mask != 0);
    }

    input_tensors.push_back(Ort::Value::CreateTensor<bool>(memory_info, reinterpret_cast<bool*>(spanMasksBool.data()), spanMasksBool.size(), shape4.data(), shape4.size()));

    std::vector<const char*> input_names = {"input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"};
    std::vector<const char*> output_names = {"logits"}; // Adjust based on your model's output

    try {
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());
        std::cout << "Minimal input test successful!" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error running minimal input test: " << e.what() << std::endl;
    }
}



void testModelInference() {
    try {
        // Setup
        Config config;
        config.max_length = 512;
        config.max_width = 12;
        WhitespaceTokenSplitter splitter;

        auto blob = LoadBytesFromFile("tokenizer.json");
        // Create the tokenizer
        auto tokenizer = Tokenizer::FromBlobJSON(blob);

        SpanProcessor processor(config, *tokenizer, splitter);

        SpanDecoder decoder(config);

        Model model("./model.onnx", config, processor, decoder);

        // Test case 1: Normal input
        {
            std::vector<std::string> texts = {"Kyiv is the capital of Ukraine."};
            std::vector<std::string> entities = {"city", "person", "country"};
            
            auto output = model.inference(texts, entities);
            
            for (size_t batch = 0; batch < output.size(); ++batch) {
                std::cout << "Batch " << batch << ":\n";
                for (const auto& span : output[batch]) {
                    std::cout << "  Span: [" << span.startIdx << ", " << span.endIdx << "], "
                            << "Class: " << span.classLabel << ", "
                            << "Prob: " << span.prob << "\n";
                }
            }           

            assert(!output.empty() && "Output should not be empty");
            std::cout << "Test case 1 (Normal input) passed." << std::endl;
        }

        // Test case 2: Large input
        {
            std::vector<std::string> texts(100, "This is a very long text to test the model's capability to handle large inputs");
            std::vector<std::string> entities = {"PERSON", "LOCATION", "ORGANIZATION"};
            
            auto output = model.inference(texts, entities);
            
            assert(!output.empty() && "Output should not be empty for large input");
            std::cout << "Test case 2 (Large input) passed." << std::endl;
        }

        // Test case 3: Invalid entities
        {
            std::vector<std::string> texts = {"Hello world"};
            std::vector<std::string> entities = {"INVALID_ENTITY"};
            
            bool exceptionThrown = false;
            try {
                model.inference(texts, entities);
            } catch (const std::exception& e) {
                exceptionThrown = true;
            }
            assert(exceptionThrown && "Invalid entities should throw an exception");
            std::cout << "Test case 3 (Invalid entities) passed." << std::endl;
        }

        std::cout << "All test cases passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during testing: " << e.what() << std::endl;
    }
}



void testProcessor() {
    WhitespaceTokenSplitter splitter;
    Config config;
    config.max_length = 512;
    config.max_width = 12;
    auto blob = LoadBytesFromFile("tokenizer.json");

    // Create the tokenizer
    auto tokenizer = Tokenizer::FromBlobJSON(blob);

    // Create Processor object directly
    SpanProcessor processor(config, *tokenizer, splitter);

    // Test tokenizeText
    std::string input = "Hello world";
    auto [tokens, startIdx, endIdx] = processor.tokenizeText(input);
    if (!tokens.empty() && !startIdx.empty() && !endIdx.empty()) {
        std::cout << tokens[0] << " " << startIdx[0] << " " << endIdx[0] << std::endl;
    } else {
        std::cout << "Tokenization resulted in empty vectors" << std::endl;
    }

    // Test batch tokenization
    std::vector<std::string> batchInput{"Hello world", "I love you"};
    auto [batchTokens, batchStartIdx, batchEndIdx] = processor.batchTokenizeText(batchInput);
    if (!batchTokens[1].empty() && !batchStartIdx[1].empty() && !batchEndIdx[1].empty()) {
        std::cout << batchTokens[1][0] << " " << batchStartIdx[1][0] << " " << batchEndIdx[1][0] << std::endl;
    }

    // Test createClassMap
    std::vector<std::string> classes{"country", "company", "person"};
    std::unordered_map<int, std::string> idToClass = processor.createClassMap(classes);
    for (const auto& [key, val] : idToClass) {
        std::cout << key << ':' << val << std::endl;
    }

    // Test prepareTextInputs
    auto [inputTexts, textLengths, promptLengths] = processor.prepareTextInputs(batchTokens, classes);
    std::cout << inputTexts[0][0] << std::endl;

    // Test encodeInputs
    auto [inputsIds, attentionMasks, wordsMasks] = processor.encodeInputs(inputTexts, &promptLengths);
    std::cout << inputsIds[0][0] << " " << attentionMasks[0][0] << " " << wordsMasks[0][0] << std::endl;

    std::vector<std::vector<int64_t>> padedinputsIds = processor.padArray(inputsIds);

    auto [spanIdxs, spanMasks] = processor.prepareSpans(textLengths, config.max_width);

    std::cout << "First span: " << spanIdxs[0][0][0] << spanIdxs[0][0][1] << std::endl;

    auto batch = processor.prepareBatch(batchInput, classes);

    std::cout << "Batch is prepared" << std::endl;
}



void testWhitespaceTokenSplitter() {
    WhitespaceTokenSplitter splitter;

    std::string text1 = "Hello world_this-is a_test!";
    auto result1 = splitter.call(text1);

    // Expected output: [("Hello", 0, 5), ("world_this-is", 6, 18), ("a", 19, 20), ("_test", 21, 26), ("!", 26, 27)]
    auto word1 = result1[0];
    std::cout << "Word1: " << std::get<0>(word1) << ", "<< std::get<1>(word1) << ", " << std::get<2>(word1) << std::endl;
    assert(result1.size() == 4);
}

void PrintEncodeResult(const std::vector<int>& ids) {
  std::cout << "tokens=[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;
}

void TestTokenizer(std::unique_ptr<Tokenizer> tok, bool print_vocab = false,
                   bool check_id_back = true) {
  // Check #1. Encode and Decode
  std::string prompt = "What is the capital of Canada?";
  std::vector<int> ids = tok->Encode(prompt);
  std::string decoded_prompt = tok->Decode(ids);
  PrintEncodeResult(ids);
  std::cout << "decode=\"" << decoded_prompt << "\"" << std::endl;
  assert(decoded_prompt == prompt);

  // Check #2. IdToToken and TokenToId
  std::vector<int32_t> ids_to_test = {0, 1, 2, 3, 32, 33, 34, 130, 131, 1000};
  for (auto id : ids_to_test) {
    auto token = tok->IdToToken(id);
    auto id_new = tok->TokenToId(token);
    std::cout << "id=" << id << ", token=\"" << token << "\", id_new=" << id_new << std::endl;
    if (check_id_back) {
      assert(id == id_new);
    }
  }

  // Check #3. GetVocabSize
  auto vocab_size = tok->GetVocabSize();
  std::cout << "vocab_size=" << vocab_size << std::endl;

  std::cout << std::endl;
}


void HuggingFaceTokenizerExample() {
  std::cout << "Tokenizer: Huggingface" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // Read blob from file.
  auto blob = LoadBytesFromFile("tokenizer.json");
  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = Tokenizer::FromBlobJSON(blob);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "Load time: " << duration << " ms" << std::endl;

  TestTokenizer(std::move(tok), false, true);
}

int main() {
    testWhitespaceTokenSplitter();
    HuggingFaceTokenizerExample();
    testWithMinimalInput();
    testModelInference();
    testProcessor();
    return 0;
}
