if (BUILD_TESTING)
    add_executable(test_basic test.cpp)
    
    target_include_directories(test_basic PRIVATE ${ONNXRUNTIME_INCLUDE})

    target_include_directories(test_basic PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(test_basic PRIVATE ${ONNXRUNTIME_LIB} GTest::gtest_main gliner)

    include(GoogleTest)
    gtest_discover_tests(test_basic)
endif()