#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import jieba_fast_dat as jieba
import threading
import logging

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_abs_path_dict_small = os.path.join(_cur_dir, '..', 'extra_dict', 'dict.txt.small')

def inittokenizer(tokenizer, group):
	logging.info(f'===> Thread {group}:{threading.current_thread().ident} started')
	tokenizer.initialize()
	logging.info(f'<=== Thread {group}:{threading.current_thread().ident} finished')

tokrs1 = [jieba.Tokenizer() for n in range(5)]
tokrs2 = [jieba.Tokenizer(_abs_path_dict_small) for n in range(5)]

thr1 = [threading.Thread(target=inittokenizer, args=(tokr, 1)) for tokr in tokrs1]
thr2 = [threading.Thread(target=inittokenizer, args=(tokr, 2)) for tokr in tokrs2]
for thr in thr1:
	thr.start()
for thr in thr2:
	thr.start()
for thr in thr1:
	thr.join()
for thr in thr2:
	thr.join()

del tokrs1, tokrs2

logging.info('='*40)

tokr1 = jieba.Tokenizer()
tokr2 = jieba.Tokenizer(_abs_path_dict_small)

thr1 = [threading.Thread(target=inittokenizer, args=(tokr1, 1)) for n in range(5)]
thr2 = [threading.Thread(target=inittokenizer, args=(tokr2, 2)) for n in range(5)]
for thr in thr1:
	thr.start()
for thr in thr2:
	thr.start()
for thr in thr1:
	thr.join()
for thr in thr2:
	thr.join()
