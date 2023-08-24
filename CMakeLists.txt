cmake_minimum_required(VERSION 3.20.0)
project(cformer VERSION 0.1 LANGUAGES CXX)

set(FILES
    cformer.h
    tensor.cpp
    net.cpp
    data.cpp
    backtrace.cpp
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CF_DEBUG "Enable debug" off)
if(CF_DEBUG)
    add_compile_definitions(CF_DEBUG)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -std=c++11 -g")

find_package(ArrayFire)

option(CF_MNIST "Enable MNIST" off)
if(CF_MNIST)
    add_executable(cformer_mnist ${FILES} examples/mnist.cpp)
    target_link_libraries(cformer_mnist ArrayFire::af -ldw)
else()
    add_executable(cformer ${FILES} main.cpp)
    target_link_libraries(cformer ArrayFire::afcuda -ldw)
endif()

option(CF_TEST "Enable tests" off)
if(CF_TEST)
    enable_testing()
    find_package(GTest)
    add_executable(test_cformer tensor.cpp net.cpp data.cpp backtrace.cpp
        tests/tensor.cpp tests/net.cpp tests/data.cpp
    )
    target_link_libraries(test_cformer ArrayFire::afcpu GTest::gtest_main -ldw)
    gtest_discover_tests(test_cformer)
    #gtest_discover_tests(test_cformer WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
