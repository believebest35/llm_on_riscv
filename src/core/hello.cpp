#include "hello.h"

#include <iostream>

#include "Eigen/Dense"

void say_hello() {
  std::cout << "Hello from core library" << std::endl;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m;
}
