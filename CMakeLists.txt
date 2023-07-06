cmake_minimum_required(VERSION 3.20.0)
project(cformer VERSION 0.1 LANGUAGES CXX)

set(FILES
    main.cpp
    cformer.h
    tensor.cpp
)

option(CF_DEBUG "Enable debug" off)
if(CF_DEBUG)
    add_compile_definitions(CF_DEBUG)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

find_package(ArrayFire)
add_executable(cformer ${FILES})
target_link_libraries(cformer ArrayFire::afcpu)

option(CF_TEST "Enable tests" off)
if(CF_TEST)
    enable_testing()
    find_package(GTest)
    add_executable(test_tensor cformer.h tests/tensor.cpp tensor.cpp)
    target_link_libraries(test_tensor ArrayFire::afcpu GTest::gtest_main)
    gtest_discover_tests(test_tensor)
endif()
