add_executable(test_hello
    test_hello.cpp
)

target_link_libraries(test_hello
    PRIVATE
        core
        gtest
        gtest_main
)

add_test(NAME hello_test COMMAND test_hello)