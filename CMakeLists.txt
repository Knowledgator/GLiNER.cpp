# Example cmake project
cmake_minimum_required(VERSION 3.18)
project(tokenizers_cpp_example C CXX)

include(CheckCXXCompilerFlag)
if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()

# Path to ONNXRuntime
set(ONNXRUNTIME_ROOTDIR "./onnxruntime-training-linux-x64-1.19.2")

# Include directories
include_directories("${ONNXRUNTIME_ROOTDIR}/include"
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")

# Link directories for ONNXRuntime
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# Path to tokenizers-cpp
set(TOKENIZER_CPP_PATH "./tokenizers-cpp")

# Add tokenizers-cpp subdirectory
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

# Add executable
add_executable(main main.cpp)

# Include directories for tokenizers-cpp
target_include_directories(main PRIVATE ${TOKENIZER_CPP_PATH}/include)

# Link ONNXRuntime and tokenizers-cpp libraries
target_link_libraries(main onnxruntime tokenizers_cpp)
