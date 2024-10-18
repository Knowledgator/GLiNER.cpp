#pragma once

#include <regex>
#include <string>
#include <vector>

#include "gliner_structs.hpp"

namespace gliner {
    class WhitespaceTokenSplitter {
    private:
        std::regex whitespacePattern;

    public:
        WhitespaceTokenSplitter();
        std::vector<Token> call(const std::string& text);
    };

    // Utility functions for tokenizer
    std::string LoadBytesFromFile(const std::string& path);
}