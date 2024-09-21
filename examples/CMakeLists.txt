cmake_minimum_required(VERSION 3.18)
project(gliner_example LANGUAGES CXX VERSION 0.0.1)

add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(GLINER_ROOTDIR ../)

add_subdirectory(${GLINER_ROOTDIR} GLiNER.cpp)

add_executable(inference inference.cpp)

target_include_directories(inference PRIVATE ${GLINER_ROOTDIR}/include)
target_link_libraries(inference gliner)