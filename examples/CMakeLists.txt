add_executable(inference inference.cpp)

target_include_directories(inference PRIVATE ../ ../include)
target_link_libraries(inference gliner)