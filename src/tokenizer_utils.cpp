#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <fstream>
#include <iostream>
#include <memory>
#include "GLiNER/tokenizer_utils.hpp"

namespace gliner
{

    // RAII wrapper for PCRE2 resources
    class PCRE2Resource
    {
    private:
        pcre2_code *pattern_;
        pcre2_match_data *match_data_;

    public:
        PCRE2Resource() : pattern_(nullptr), match_data_(nullptr) {}

        ~PCRE2Resource()
        {
            if (match_data_)
                pcre2_match_data_free(match_data_);
            if (pattern_)
                pcre2_code_free(pattern_);
        }

        // Prevent copying
        PCRE2Resource(const PCRE2Resource &) = delete;
        PCRE2Resource &operator=(const PCRE2Resource &) = delete;

        void compilePattern(const char *pattern)
        {
            int errorcode;
            PCRE2_SIZE erroroffset;

            pattern_ = pcre2_compile(
                reinterpret_cast<PCRE2_SPTR>(pattern),
                PCRE2_ZERO_TERMINATED,
                PCRE2_UTF | PCRE2_UCP,
                &errorcode,
                &erroroffset,
                nullptr);

            if (!pattern_)
            {
                PCRE2_UCHAR buffer[256];
                pcre2_get_error_message(errorcode, buffer, sizeof(buffer));
                throw std::runtime_error("PCRE2 compilation failed at offset " +
                                         std::to_string(erroroffset) + ": " +
                                         reinterpret_cast<char *>(buffer));
            }

            // Enable JIT compilation for better performance
            pcre2_jit_compile(pattern_, PCRE2_JIT_COMPLETE);

            match_data_ = pcre2_match_data_create_from_pattern(pattern_, nullptr);
            if (!match_data_)
            {
                throw std::runtime_error("Failed to create PCRE2 match data");
            }
        }

        pcre2_code *pattern() const { return pattern_; }
        pcre2_match_data *match_data() const { return match_data_; }
        PCRE2_SIZE *getOvectorPointer() const
        {
            return pcre2_get_ovector_pointer(match_data_);
        }
    };

    struct WhitespaceTokenSplitter::Implementation
    {
        PCRE2Resource pcre2;
    };

    std::string LoadBytesFromFile(const std::string &path)
    {
        std::ifstream fs(path, std::ios::in | std::ios::binary);
        if (!fs)
        {
            throw std::runtime_error("Cannot open file: " + path);
        }

        // More efficient file reading using a single allocation
        fs.seekg(0, std::ios::end);
        std::string data;
        data.reserve(fs.tellg());
        fs.seekg(0, std::ios::beg);

        data.assign(
            std::istreambuf_iterator<char>(fs),
            std::istreambuf_iterator<char>());

        return data;
    }

    WhitespaceTokenSplitter::WhitespaceTokenSplitter()
        : pimpl(std::make_unique<Implementation>())
    {
        pimpl->pcre2.compilePattern("\\w+(?:[-_]\\w+)*|\\S");
    }

    WhitespaceTokenSplitter::~WhitespaceTokenSplitter() = default;

    std::vector<Token> WhitespaceTokenSplitter::call(const std::string &text)
    {
        std::vector<Token> tokens;
        tokens.reserve(text.length() / 4); // Estimate initial capacity

        PCRE2_SIZE *ovector = pimpl->pcre2.getOvectorPointer();
        const size_t subject_length = text.length();
        size_t start_offset = 0;

        while (true)
        {
            int rc = pcre2_match(
                pimpl->pcre2.pattern(),
                reinterpret_cast<PCRE2_SPTR>(text.c_str()),
                subject_length,
                start_offset,
                PCRE2_NO_UTF_CHECK,
                pimpl->pcre2.match_data(),
                nullptr);

            if (rc < 0)
            {
                if (rc != PCRE2_ERROR_NOMATCH)
                {
                    throw std::runtime_error("PCRE2 matching error: " + std::to_string(rc));
                }
                break;
            }

            const size_t start = ovector[0];
            const size_t end = ovector[1];

            tokens.push_back({start,
                              end,
                              text.substr(start, end - start)});

            start_offset = end;
        }

        return tokens;
    }

}