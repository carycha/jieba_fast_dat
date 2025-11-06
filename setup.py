# -*- coding: utf-8 -*-
from setuptools import setup, Extension
import platform
import os  # Import os for environment variable checking
import pybind11  # Import pybind11

# Define the pybind11 extension
jieba_fast_dat_functions_py3 = Extension(
    "_jieba_fast_dat_functions_py3",
    sources=[
        "jieba_fast_dat/source/pybind_bindings.cpp"
    ],  # Point to our new pybind11 source
    include_dirs=[pybind11.get_include()],  # Include pybind11 headers
    language="c++",  # Specify C++ language
    extra_compile_args=["-std=c++17"]
    + (
        ["-fsanitize=address", "-fno-omit-frame-pointer", "-g"]
        if os.environ.get("ENABLE_ASAN") == "1"
        else []
    ),  # Ensure C++11 or later standard
)


if platform.python_version().startswith("3"):
    setup(ext_modules=[jieba_fast_dat_functions_py3])
