from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

extra_compile_args = []
if sys.platform.startswith("win"):
    extra_compile_args += ["/O2"]
else:
    extra_compile_args += ["-O3"]

ext_modules = [
    Pybind11Extension(
        "SpringDamping",
        ["src/spring_damping.cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
    Pybind11Extension(
        "Test",
        ["src/test.cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="backend",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
