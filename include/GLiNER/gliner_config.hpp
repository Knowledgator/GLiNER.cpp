#pragma once

namespace gliner {
    enum ModelType {
        TOKEN_LEVEL,
        SPAN_LEVEL
    };

    struct Config {
        int maxWidth;
        int maxLength;
        ModelType modelType = SPAN_LEVEL;
    };
}