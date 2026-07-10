option(GLINER_BUILD_SHARED "Build gliner as a shared library (DLL)" ON)

if(GLINER_BUILD_SHARED)
    add_library(gliner SHARED)
    target_compile_definitions(gliner PRIVATE GLINER_EXPORTS)
    target_compile_definitions(gliner PUBLIC GLINER_BUILD_SHARED)
else()
    add_library(gliner STATIC)
endif()

target_sources(gliner PRIVATE 
    processor.cpp
    model.cpp
    decoder.cpp
    tokenizer_utils.cpp
    gliner_structs.cpp
)

target_include_directories(gliner PUBLIC 
    ${ONNXRUNTIME_INCLUDE}
    ${PCRE2_INCLUDE_DIRS}
)

target_include_directories(gliner PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(gliner 
    ${ONNXRUNTIME_LIB} 
    tokenizers_cpp
    ${PCRE2_LIBRARIES}
)