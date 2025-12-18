add_library(core
    hello.cpp
)

target_include_directories(core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)