cmake_minimum_required(VERSION 3.16)
project(llm_on_riscv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/compiler.cmake)

option(BUILD_TESTS "Build unit tests" ON)

add_subdirectory(src)
add_subdirectory(third_party/eigen)
add_subdirectory(third_party/gtest)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
