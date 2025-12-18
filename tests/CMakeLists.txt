add_executable(test_hello
    test_hello.cpp
)

target_link_libraries(test_hello
    PRIVATE
        core
)

add_test(NAME hello_test COMMAND test_hello)