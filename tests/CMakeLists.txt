include(GoogleTest)

set(TEST_TARGETS
    test_hello
    test_matmul
    test_mul
    test_simplified_layernorm
    test_skip_simplified_layernorm
    test_add
    test_shape
    test_constant
    test_gather
    test_transpose
    test_reduce_sum
    test_unsqueeze
    test_sub
    test_cast
    test_concat
    test_reshape
    test_rotary_embedding
    test_sigmoid
    test_group_query_attention
)

foreach(test_target IN LISTS TEST_TARGETS)
    add_executable(${test_target} ${test_target}.cpp)

    target_link_libraries(${test_target}
        PRIVATE
            core
            gtest
            gtest_main
    )

    gtest_discover_tests(${test_target}
        EXTRA_ARGS --gtest_print_time=0
    )
endforeach()
