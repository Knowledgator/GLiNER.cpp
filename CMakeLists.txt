cmake_minimum_required(VERSION 3.18)
project(gliner LANGUAGES CXX VERSION 0.0.2)

# Update submodules
option(UPDATE_SUBMODULES "Update submodules" ON)
if(UPDATE_SUBMODULES)
    include(cmake/UpdateSubmodules.cmake)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(PCRE2_VERSION "10.44")

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(FetchContent)

FetchContent_Declare(
    pcre2
    GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
    GIT_TAG pcre2-${PCRE2_VERSION}
)

set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "Build 8bit PCRE2 library")
set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "Build 16bit PCRE2 library")
set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "Build 32bit PCRE2 library")
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "Build PCRE2 tests")
set(PCRE2_BUILD_PROGRAMS OFF CACHE BOOL "Build PCRE2 programs")

# Fetch and make PCRE2 available
FetchContent_MakeAvailable(pcre2)

set(PCRE2_INCLUDE_DIRS ${pcre2_SOURCE_DIR}/src ${pcre2_BINARY_DIR}/src)
set(PCRE2_LIBRARIES pcre2-8)

option(BUILD_EXAMPLES "Build example programs" OFF)

# Find ONNXRuntime library
option(ONNXRUNTIME_ROOTDIR "root dir containing include/ and lib/")
option(ONNXRUNTIME_INCLUDE_DIR "Path to ONNXRuntime include directory (containing onnxruntime_cxx_api.h etc.)")
option(ONNXRUNTIME_LIB_DIR "Path to ONNXRuntime lib directory (containing onnxruntime.lib / .dll / .so)")

if(ONNXRUNTIME_ROOTDIR)
    if(NOT ONNXRUNTIME_INCLUDE_DIR)
        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include")
    endif()
    if(NOT ONNXRUNTIME_LIB_DIR)
        set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOTDIR}/lib")
    endif()
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "The ONNXRuntime include path must be specified. Use -D ONNXRUNTIME_INCLUDE_DIR='path' to set it, or set path to root dir via ONNXRUNTIME_ROOTDIR")
endif()

if(NOT ONNXRUNTIME_LIB_DIR)
    message(FATAL_ERROR "The ONNXRuntime lib path must be specified. Use -D ONNXRUNTIME_LIB_DIR='path' to set it, or set path to root dir via ONNXRUNTIME_ROOTDIR")
endif()

set(ONNXRUNTIME_INCLUDE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime
    ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session
)

include_directories(
    ${ONNXRUNTIME_INCLUDE}
)

link_directories(${ONNXRUNTIME_LIB_DIR})

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime libonnxruntime
    PATHS ${ONNXRUNTIME_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNXRuntime library not found. Please check the library path.")
endif()

option(GPU_CHECK "Check GPU dependencies." OFF)
if (GPU_CHECK)
    execute_process(
        COMMAND nvidia-smi
        RESULT_VARIABLE NVIDIA_SMI_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(NOT NVIDIA_SMI_RESULT EQUAL 0)
        message(WARNING "NVIDIA GPU driver was not detected or nvidia-smi command not available")
        set(GPU_CHECK OFF)
    endif()

    execute_process(
        COMMAND nvcc --version
        RESULT_VARIABLE NVCC_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(NOT NVCC_RESULT EQUAL 0)
        message(WARNING "NVIDIA GPU compiler was not detected or nvcc --version command not available.")
        set(GPU_CHECK OFF)
    endif()

    find_library(CUDNN_LIB
        NAMES cudnn
        PATHS CMAKE_CXX_STANDARD_LIBRARIES CMAKE_C_STANDARD_LIBRARIES
        DOC "Path to cuDNN library"
    )

    if(NOT CUDNN_LIB)
        message(WARNING "cuDNN not found. Ensure that cuDNN is installed and available in the system paths.")
        set(GPU_CHECK OFF)
    endif()

    if (NOT GPU_CHECK)
        message(FATAL_ERROR "CUDA GPU dependencies were not resolved. Please check the logs and try again.")
    endif()
endif()

execute_process(
    COMMAND cargo --version
    RESULT_VARIABLE CARGO_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT CARGO_RESULT EQUAL 0)
    message(FATAL_ERROR "Cargo is not available. You can install it by following the instructions here: https://www.rust-lang.org/tools/install")
    set(GPU_CHECK OFF)
endif()

add_subdirectory(deps)
add_subdirectory(src)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(CTest)
    add_subdirectory(tests)
endif()