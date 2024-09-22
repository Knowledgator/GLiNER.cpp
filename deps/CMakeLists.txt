set(CMAKE_CXX_STANDARD 17)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(googletest)
endif()

if(NOT TOKENIZER_CPP_ROOTDIR)
    add_subdirectory(tokenizers-cpp)
else()
    add_subdirectory(TOKENIZER_CPP_ROOTDIR tokenizers-cpp)
endif()

