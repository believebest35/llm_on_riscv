add_library(qwen3_runtime STATIC
    weights.cpp
    qwen3_model.cpp
)

target_include_directories(qwen3_runtime
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(qwen3_runtime
    PUBLIC
        eigen
)

add_executable(qwen3_run
    main.cpp
)

target_link_libraries(qwen3_run
    PRIVATE
        qwen3_runtime
)
