#pragma once
#include <string>
#include <vector>
#include "gliner_structs.hpp"

namespace gliner {

class WhitespaceTokenSplitter {
private:
    struct Implementation;
    std::unique_ptr<Implementation> pimpl;

public:
    WhitespaceTokenSplitter();
    ~WhitespaceTokenSplitter();
    WhitespaceTokenSplitter(const WhitespaceTokenSplitter&) = delete;
    WhitespaceTokenSplitter& operator=(const WhitespaceTokenSplitter&) = delete;
    std::vector<Token> call(const std::string& text);
};

std::string LoadBytesFromFile(const std::string& path);

}