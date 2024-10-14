#pragma once

#include <vector>
#include <string>

#include "gliner_config.hpp"
#include "gliner_structs.hpp"

namespace gliner {
    class Decoder {
    protected:
        virtual std::vector<Span> greedySearch(const std::vector<Span>&  spans, bool flatNer = true, bool multiLabel = false);
        virtual std::vector<std::vector<Span>> batchGreedySearch(
            const std::vector<std::vector<Span>>&  spans_batch, bool flatNer = true, bool multiLabel = false
        );
        static bool isNested(const Span& s1, const Span& s2);
        static bool hasOverlapping(const Span& s1, const Span& s2, bool multiLabel = false);
        static bool hasOverlappingNested(const Span& s1, const Span& s2, bool multiLabel = false);
    public:
        virtual ~Decoder() {};
        virtual std::vector<std::vector<Span>> decode(
            const Batch* batch,
            const std::vector<std::string>& texts,
            const std::vector<std::string>& entities,
            const std::vector<float>& modelOutput,
            bool flatNer = false,
            float threshold = 0.5,
            bool multiLabel = false
        ) = 0;
    };

    class SpanDecoder : public Decoder {
    public:
        virtual ~SpanDecoder() {};
        virtual std::vector<std::vector<Span>> decode(
            const Batch* batch,
            const std::vector<std::string>& texts,
            const std::vector<std::string>& entities,
            const std::vector<float>& modelOutput,
            bool flatNer = false,
            float threshold = 0.5,
            bool multiLabel = false
        );
    };

    class TokenDecoder : public Decoder {
    public:
        virtual ~TokenDecoder() {};
        virtual std::vector<std::vector<Span>> decode(
            const Batch* batch,
            const std::vector<std::string>& texts,
            const std::vector<std::string>& entities,
            const std::vector<float>& modelOutput,
            bool flatNer = false,
            float threshold = 0.5,
            bool multiLabel = false
        );
    };
}