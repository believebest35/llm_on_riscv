#include "hello.h"

#include <Eigen/Dense>
#include <iostream>

#include "Eigen/Core"

void say_hello() {
  std::cout << "Hello from core library" << std::endl;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m;
}
