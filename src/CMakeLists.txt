add_subdirectory(core)

add_executable(llm_on_riscv
    main.cpp
)

target_link_libraries(llm_on_riscv
    PRIVATE
        core
)