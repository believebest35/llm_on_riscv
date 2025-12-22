#include <gtest/gtest.h>
#include "core/hello.h"

TEST(HelloTest, Basic) {
  say_hello();
  SUCCEED();
}