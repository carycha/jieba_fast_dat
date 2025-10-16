# -*- coding: utf-8 -*-
import setuptools
from setuptools import setup, Extension
import pybind11
import platform


# Define the pybind11 extension
jieba_fast_functions_pybind = Extension(
    '_jieba_fast_functions_pybind',
    sources=['jieba_fast_dat/source/jieba_fast_functions_pybind.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++', 
    extra_compile_args=['-std=c++11'],
)


if __name__ == '__main__':
    setup(
        ext_modules = [jieba_fast_functions_pybind],
    )