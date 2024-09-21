cmake_minimum_required(VERSION 3.18)
project(gliner LANGUAGES CXX VERSION 0.0.1)

# Update submodules
option(UPDATE_SUBMODULES "Update submodules" ON)
if(UPDATE_SUBMODULES)
    include(cmake/UpdateSubmodules.cmake)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(BUILD_EXAMPLES "Build example programs" OFF)

# Find ONNXRuntime library
option(ONNXRUNTIME_ROOTDIR "Onnxruntime root dir")
if(NOT ONNXRUNTIME_ROOTDIR)
    message(FATAL_ERROR "The ONNXRuntime path must be specified. Use the -D ONNXRUNTIME_ROOTDIR='absolute_path' option to set it.")
endif()

set(ONNXRUNTIME_INCLUDE
${ONNXRUNTIME_ROOTDIR}/include
${ONNXRUNTIME_ROOTDIR}/include/onnxruntime
${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session
)

include_directories(
    ${ONNXRUNTIME_INCLUDE}
)

link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime libonnxruntime
    PATHS ${ONNXRUNTIME_ROOTDIR}/lib
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNXRuntime library not found. Please check the library path.")
endif()

option(TOKENIZER_CPP_ROOTDIR "Tokenizer-cpp root dir")

add_subdirectory(deps)
add_subdirectory(src)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

if(CMAKE_BUILD_TYPE EQUAL Debug)
    include(CTest)
    add_subdirectory(tests)
endif()