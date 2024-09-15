#pragma once

#include <regex>
#include <string>
#include <vector>
#include <tuple>

class WhitespaceTokenSplitter {
private:
    std::regex whitespacePattern;

public:
    WhitespaceTokenSplitter();
    std::vector<std::tuple<std::string, size_t, size_t>> call(const std::string& text);
};

// Utility functions for tokenizer
std::string LoadBytesFromFile(const std::string& path);