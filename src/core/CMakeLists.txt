add_library(core
    hello.cpp
    tensor/tensor.cpp
    memory/memory_pool.cpp
)

target_include_directories(core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(core
    PUBLIC
        eigen
)