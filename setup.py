# -*- coding: utf-8 -*-
import setuptools
from setuptools import setup, Extension
import pybind11
import platform

LONGDOC = '使用 dat 與 mmap to Speed up jieba<Chinese Words Segementation Utilities>'


# Define the pybind11 extension
jieba_fast_functions_pybind = Extension(
    '_jieba_fast_functions_pybind',
    sources=['jieba_fast_dat/source/jieba_fast_functions_pybind.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++', 
    extra_compile_args=['-std=c++11'],
)


if __name__ == '__main__':
    setup(name='jieba_fast_dat',
              version='0.54',
            description='使用 dat 與 mmap to Speed up jieba<Chinese Words Segementation Utilities>',
            long_description=LONGDOC,
            author='carycha, Sun, Junyi, deepcs233',
            author_email='carycha@gmail.com',
            url='https://github.com/deepcs233/jieba_fast',
            license="MIT",
            classifiers=[
                'Intended Audience :: Developers',
                'License :: OSI Approved :: MIT License',
                'Operating System :: OS Independent',
                'Natural Language :: Chinese (Simplified)',
                'Natural Language :: Chinese (Traditional)',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.8',
                'Programming Language :: Python :: 3.9',
                'Programming Language :: Python :: 3.10',
                'Programming Language :: Python :: 3.11',
                'Programming Language :: Python :: 3.12',
                'Topic :: Text Processing',
                'Topic :: Text Processing :: Indexing',
                'Topic :: Text Processing :: Linguistic',
            ],
            keywords='NLP,tokenizing,Chinese word segementation',
            packages=['jieba_fast_dat'],
            package_dir={'jieba_fast_dat':'jieba_fast_dat'},
            package_data={'jieba_fast_dat':['*.*','finalseg/*','analyse/*','posseg/*','source/*']},
            ext_modules = [jieba_fast_functions_pybind],
            install_requires=[
                'pybind11>=2.6.0', # Specify a minimum version for pybind11
            ],
        )
