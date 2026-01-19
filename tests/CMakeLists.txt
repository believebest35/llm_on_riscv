add_executable(test_hello
    test_hello.cpp
)
add_executable(test_matmul
    test_matmul.cpp
)
add_executable(test_mul
    test_mul.cpp
)
add_executable(test_simplified_layernorm
    test_simplified_layernorm.cpp
)

target_link_libraries(test_hello
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_matmul
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_mul
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_simplified_layernorm
    PRIVATE
        core
        gtest
        gtest_main
)

add_test(NAME hello_test COMMAND test_hello)
add_test(NAME matmul_test COMMAND test_matmul)
add_test(NAME mul_test COMMAND test_mul)
add_test(NAME simplified_layernorm_test COMMAND test_simplified_layernorm)