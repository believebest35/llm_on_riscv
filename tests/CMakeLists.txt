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
add_executable(test_skip_simplified_layernorm
    test_skip_simplified_layernorm.cpp
)

add_executable(test_add
    test_add.cpp
)

add_executable(test_shape
    test_shape.cpp
)

add_executable(test_constant
    test_constant.cpp
)

add_executable(test_gather
    test_gather.cpp
)

add_executable(test_transpose
    test_transpose.cpp
)

add_executable(test_reduce_sum
    test_reduce_sum.cpp
)

add_executable(test_unsqueeze
    test_unsqueeze.cpp
)

add_executable(test_sub
    test_sub.cpp
)

add_executable(test_cast
    test_cast.cpp
)

add_executable(test_concat
    test_concat.cpp
)

add_executable(test_reshape
    test_reshape.cpp
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

target_link_libraries(test_skip_simplified_layernorm
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_add
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_shape
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_constant
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_gather
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_transpose
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_reduce_sum
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_unsqueeze
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_sub
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_cast
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_concat
    PRIVATE
        core
        gtest
        gtest_main
)

target_link_libraries(test_reshape
    PRIVATE
        core
        gtest
        gtest_main
)

add_test(NAME hello_test COMMAND test_hello)
add_test(NAME matmul_test COMMAND test_matmul)
add_test(NAME mul_test COMMAND test_mul)
add_test(NAME simplified_layernorm_test COMMAND test_simplified_layernorm)
add_test(NAME skip_simplified_layernorm_test COMMAND test_skip_simplified_layernorm)
add_test(NAME add_test COMMAND test_add)
add_test(NAME shape_test COMMAND test_shape)
add_test(NAME constant_test COMMAND test_constant)
add_test(NAME gather_test COMMAND test_gather)
add_test(NAME transpose_test COMMAND test_transpose)
add_test(NAME reduce_sum_test COMMAND test_reduce_sum)
add_test(NAME unsqueeze_test COMMAND test_unsqueeze)
add_test(NAME sub_test COMMAND test_sub)
add_test(NAME cast_test COMMAND test_cast)
add_test(NAME concat_test COMMAND test_concat)
add_test(NAME reshape_test COMMAND test_reshape)
