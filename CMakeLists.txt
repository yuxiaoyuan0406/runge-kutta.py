cmake_minimum_required(VERSION 3.18)
project(runge_kutta_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE
  module/_native/core.cpp
)

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/module/_native/include
)

# 把扩展安装进 Python 包目录：module/_native/_core*.so (或 .pyd)
install(TARGETS _core
  LIBRARY DESTINATION module/_native
  RUNTIME DESTINATION module/_native
  ARCHIVE DESTINATION module/_native
)
