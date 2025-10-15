import os
import sys
import time
sys.path.append("../")
import jieba_fast_dat as jieba
jieba.initialize()
import jieba_fast_dat.posseg as pseg

def test_pos_file_processing():
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    url = os.path.join(_cur_dir, 'test.txt') # Assuming test.txt is in the same directory
    content = open(url,"rb").read()
    t1 = time.time()
    words = list(pseg.cut(content))

    t2 = time.time()
    tm_cost = t2-t1

    assert len(words) > 0
    assert tm_cost > 0

    print('speed' , len(content)/tm_cost, " bytes/second")
