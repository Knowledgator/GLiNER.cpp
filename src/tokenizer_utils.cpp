#include <string>
#include <regex>
#include <fstream>
#include <iostream>
#include <cassert>

#include "tokenizer_utils.h"

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
};


WhitespaceTokenSplitter::WhitespaceTokenSplitter() : whitespacePattern(R"(\w+(?:[-_]\w+)*|\S)") {}

std::vector<std::tuple<std::string, size_t, size_t>> WhitespaceTokenSplitter::call(const std::string& text) {
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

