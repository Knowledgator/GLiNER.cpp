#pragma once

#include <vector>
#include <string>
#include <regex>

#include "gliner_config.hpp"
#include "gliner_structs.hpp"

namespace gliner {
    class SpanDecoder {
    protected:
        Config config;

    public:
        SpanDecoder(const Config& config);
        virtual std::vector<Span> greedySearch(const std::vector<Span>&  spans, bool flatNer = true, bool multiLabel = false);
        static bool isNested(const Span& s1, const Span& s2);
        static bool hasOverlapping(const Span& s1, const Span& s2, bool multiLabel = false);
        static bool hasOverlappingNested(const Span& s1, const Span& s2, bool multiLabel = false);
        std::vector<std::vector<Span>> decode(
            const BatchOutput& batch,
            int maxWidth,
            const std::vector<std::string>& texts,
            const std::vector<std::string>& entities,
            const std::vector<float>& modelOutput,
            bool flatNer = false,
            float threshold = 0.5,
            bool multiLabel = false
        );
    };
}