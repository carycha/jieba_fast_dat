#encoding=utf-8
import sys
import threading
import logging
sys.path.append("../")

import jieba_fast_dat as jieba

class Worker(threading.Thread):
    def run(self):
        seg_list = jieba.cut("我来到北京清华大学",cut_all=True)
        logging.info("Full Mode:" + "/ ".join(seg_list)) #全模式

        seg_list = jieba.cut("我来到北京清华大学",cut_all=False)
        logging.info("Default Mode:" + "/ ".join(seg_list)) #默认模式

        seg_list = jieba.cut("他来到了网易杭研大厦")
        logging.info(", ".join(seg_list))

        seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造") #搜索引擎模式
        logging.info(", ".join(seg_list))
workers = []
for i in range(10):
    worker = Worker()
    workers.append(worker)
    worker.start()

for worker in workers:
    worker.join()

