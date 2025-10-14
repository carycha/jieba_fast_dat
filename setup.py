# -*- coding: utf-8 -*-
from distutils.core import setup, Extension
import platform

LONGDOC = 'Use C and Swig to Speed up jieba<Chinese Words Segementation Utilities>'



jieba_fast_functions_py3 = Extension('_jieba_fast_functions_py3',
                         sources=['jieba_fast_dat/source/jieba_fast_functions_wrap_py3_wrap.c'],
                           )





setup(name='jieba_fast_dat',
          version='0.53',
        description='Use C and Swig to Speed up jieba<Chinese Words Segementation Utilities>',
        long_description=LONGDOC,
        author='Sun, Junyi, deepcs233',
        author_email='shaohao97@gmail.com',
        url='https://github.com/deepcs233/jieba_fast',
        license="MIT",
        classifiers=[
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
            'Natural Language :: Chinese (Simplified)',
            'Natural Language :: Chinese (Traditional)',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.7',
            'Topic :: Text Processing',
            'Topic :: Text Processing :: Indexing',
            'Topic :: Text Processing :: Linguistic',
        ],
        keywords='NLP,tokenizing,Chinese word segementation',
        packages=['jieba_fast_dat'],
        package_dir={'jieba_fast_dat':'jieba_fast_dat'},
          package_data={'jieba_fast_dat':['*.*','finalseg/*','analyse/*','posseg/*','source/*']},
        ext_modules = [jieba_fast_functions_py3],
    )
