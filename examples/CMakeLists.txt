cmake_minimum_required(VERSION 3.18)
project(gliner_example LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(GLINER_ROOTDIR ../)

add_subdirectory(${GLINER_ROOTDIR} GLiNER.cpp)

add_executable(inference inference.cpp)

target_include_directories(inference PRIVATE ${GLINER_ROOTDIR}/include)
target_link_libraries(inference gliner)

add_executable(inference_gpu inference_gpu.cpp)

target_include_directories(inference_gpu PRIVATE ${GLINER_ROOTDIR}/include)
target_link_libraries(inference_gpu gliner)