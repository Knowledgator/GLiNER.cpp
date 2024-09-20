#include <string>
#include <regex>
#include <fstream>
#include <iostream>

#include "GLiNER/tokenizer_utils.hpp"

using namespace gliner;

std::string gliner::LoadBytesFromFile(const std::string& path) {
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

WhitespaceTokenSplitter::WhitespaceTokenSplitter() : whitespacePattern(R"(\w+(?:[-_]\w+)*|\S)") {}

std::vector<Token> WhitespaceTokenSplitter::call(const std::string& text) {
    std::vector<Token> tokens;

    for (
        std::sregex_iterator match = std::sregex_iterator(text.begin(), text.end(), whitespacePattern);
        match != std::sregex_iterator(); ++match
    ) {
        size_t start = match->position();
        size_t end = start + match->length();
        tokens.push_back({start, end, match->str()});
    }

    return tokens;
}

