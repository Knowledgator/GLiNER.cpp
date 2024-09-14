
# 👑 GLiNER.cpp: Generalist and Lightweight Named Entity Recognition for JavaScript

GLiNER.cpp is a C++-based inference engine for running GLiNER (Generalist and Lightweight Named Entity Recognition) models. GLiNER can identify any entity type using a bidirectional transformer encoder, offering a practical alternative to traditional NER models and large language models.

<p align="center">
    <a href="https://arxiv.org/abs/2311.08526">📄 Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://discord.gg/Y2yVxpSQnG">📢 Discord</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/spaces/urchade/gliner_mediumv2.1">🤗 Demo</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/models?library=gliner&sort=trending">🤗 Available models</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://github.com/urchade/GLiNER">🧬 Official Repo</a>
</p>

## 🌟 Key Features

- Flexible entity recognition without predefined categories
- Lightweight and fast inference;
- Ideal for running GLiNER models on CPU;

## 🚀 Getting Started
First of all, clone the repository:
```bash
git clone https://github.com/Knowledgator/GLiNER.cpp.git
```

Then you need initialize and update submodules:
```bash
cd GLiNER.cpp
git submodule update --init --recursive
```

After that you need to download [ONNX runtime](https://github.com/microsoft/onnxruntime/releases) for your system.

Unpack it within the same derictory as GLiNER.cpp code.

For `tar.gz` files you can use the following command:
```bash
tar -xvzf onnxruntime-linux-x64-1.19.2.tgz 
```

Then create build directory and compile the project:
```bash
mkdir build
cd build
cmake ..
make -j8
```

To run main.cpp you need an ONNX format model and tokenizer.json. You can:

1. Search for pre-converted models on [HuggingFace](https://huggingface.co/onnx-community?search_models=gliner)
2. Convert a model yourself using the [official Python script](https://github.com/urchade/GLiNER/blob/main/convert_to_onnx.py)

### Converting to ONNX Format

Use the `convert_to_onnx.py` script with the following arguments:

- `model_path`: Location of the GLiNER model
- `save_path`: Where to save the ONNX file
- `quantize`: Set to True for IntU8 quantization (optional)

Example:

```bash
python convert_to_onnx.py --model_path /path/to/your/model --save_path /path/to/save/onnx --quantize True
```


### Example of usage:
```c++
Config config;
// specify max length parameter which controls maximum amount of words to be processed;
config.max_length = 512;
// specify max_width parameter which controls maximum length os a span;
config.max_width = 12;

// initialize word tokenizer;

WhitespaceTokenSplitter splitter;

// Create the tokenizer
auto blob = LoadBytesFromFile("tokenizer.json");
auto tokenizer = Tokenizer::FromBlobJSON(blob);

// Initialize the processor
SpanProcessor processor(config, *tokenizer, splitter);

// Initialize the model
Model model("./model.onnx", processor);


std::vector<std::string> texts = {"Hello"};
std::vector<std::string> entities = {"PERSON", "LOCATION"};

auto output = model.inference(texts, entities);

assert(!output.empty() && "Output should not be empty");
assert(output[0].IsTensor() && "Output should be a tensor");
std::cout << "Test case 1 (Normal input) passed." << std::endl;
```

## 🌟 Use Cases

GLiNER.cpp offers versatile entity recognition capabilities across various domains:

1. **Enhanced Search Query Understanding**
2. **Real-time PII Detection**
3. **Intelligent Document Parsing**
4. **Content Summarization and Insight Extraction**
5. **Automated Content Tagging and Categorization**
...

## 🔧 Areas for Improvement

- [ ] Further optimize inference speed
- [ ] Add support for token-based GLiNER architecture
- [ ] Implement bi-encoder GLiNER architecture for better scalability
- [ ] Enable model training capabilities
- [ ] Provide more usage examples


## 🙏 Acknowledgements

- [GLiNER original authors](https://github.com/urchade/GLiNER)
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)

## 📞 Support

For questions and support, please join our [Discord community](https://discord.gg/ApZvyNZU) or open an issue on GitHub.
