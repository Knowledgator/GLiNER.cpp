#pragma once

#if defined(_WIN32)
    #if defined(GLINER_BUILD_SHARED)
        #if defined(GLINER_EXPORTS)
            #define GLINER_API __declspec(dllexport)
        #else
            #define GLINER_API __declspec(dllimport)
        #endif
    #else
        #define GLINER_API
    #endif
#else
    #if defined(GLINER_BUILD_SHARED)
        #define GLINER_API __attribute__((visibility("default")))
    #else
        #define GLINER_API
    #endif
#endif