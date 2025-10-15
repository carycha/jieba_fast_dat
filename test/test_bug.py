#encoding=utf-8
from __future__ import print_function
import sys
sys.path.append("../")
import jieba_fast_dat as jieba
import jieba_fast_dat.posseg as pseg
words=pseg.cut("又跛又啞")
for w in words:
	print(w.word,w.flag)

