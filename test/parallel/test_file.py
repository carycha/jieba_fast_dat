import sys
import time

sys.path.append("../../")
import jieba_fast_dat

jieba_fast_dat.enable_parallel()

url = sys.argv[1]
content = open(url, "rb").read()
t1 = time.time()
words = "/ ".join(jieba_fast_dat.cut(content))

t2 = time.time()
tm_cost = t2 - t1

log_f = open("1.log", "wb")
log_f.write(words.encode("utf-8"))

print("speed %s bytes/second" % (len(content) / tm_cost))
