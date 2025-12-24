cmake_minimum_required(VERSION 3.20)
project(module_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 依赖管理：把 pybind11 “收口”在 module/_native 内部 ----
include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# ---- 生成 Python 扩展：名字建议就是 _core（对应你 module/_core.py 的导入习惯）----
pybind11_add_module(_core core.cpp)

# 头文件路径（你已有 include/）
target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- 让生成的 .so/.pyd 直接落到 module/ 目录，便于 Python import ----
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..   # => module/
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..   # Windows
)
