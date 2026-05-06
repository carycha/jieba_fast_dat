import os
import platform
from setuptools import Extension, setup

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path.
    The purpose of this class is to postpone importing pybind11 until it is actually
    installed, so that the ``get_include()`` method can be invoked. """
    def __str__(self):
        import pybind11
        return pybind11.get_include()

# Define the pybind11 extension
jieba_fast_dat_functions_py3 = Extension(
    "jieba_fast_dat._jieba_fast_dat_functions_py3",
    sources=[
        "jieba_fast_dat/source/pybind_bindings.cpp",
        "jieba_fast_dat/source/core/hmm_model.cpp",
        "jieba_fast_dat/source/core/viterbi_engine.cpp",
        "jieba_fast_dat/source/core/segmenter.cpp",
    ],
    include_dirs=[
        get_pybind_include(),
        "jieba_fast_dat/source",
        "jieba_fast_dat/source/core",
    ],
    language="c++",
    extra_compile_args=(
        ["-std=c++17", "-O3"]
        + (["-mmacosx-version-min=11.0"] if platform.system() == "Darwin" else [])
        + (
            ["-fsanitize=address", "-fno-omit-frame-pointer", "-g"]
            if os.environ.get("ENABLE_ASAN") == "1"
            else []
        )
    ),
    extra_link_args=(
        ["-fsanitize=address"] if os.environ.get("ENABLE_ASAN") == "1" else []
    ),
)

setup(
    ext_modules=[jieba_fast_dat_functions_py3],
)
