add_library(gliner SHARED ${PROJECT_SOURCE_DIR}/include/GLiNER)

target_sources(gliner PRIVATE 
    processor.cpp
    model.cpp
    decoder.cpp
    tokenizer_utils.cpp
)

target_include_directories(gliner PUBLIC ${ONNXRUNTIME_INCLUDE})
target_include_directories(gliner PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gliner ${ONNXRUNTIME_LIB} tokenizers_cpp)