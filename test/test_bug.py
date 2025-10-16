#encoding=utf-8

import logging

import sys
sys.path.append("../")
import jieba_fast_dat as jieba
import jieba_fast_dat.posseg as pseg
words=pseg.cut("又跛又啞")
for w in words:
	logging.info(w.word,w.flag)

