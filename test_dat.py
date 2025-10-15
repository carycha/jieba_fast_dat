import jieba_fast_dat
import jieba_fast_dat.posseg

# The initialize() method should be called automatically on the first cut
jieba_fast_dat.set_dictionary("extra_dict/dict.txt.big")
jieba_fast_dat.load_userdict("extra_dict/dict.txt.big.tw_nerd.txt")
text = "看過華燈初上簡單訂沉浸式劇場改編平行時空,因為所以想你幾乎完全真的覺得門票入場雖然由於"
print("测试一下DAT的实现cut=====")
words = jieba_fast_dat.cut(text,HMM=True)
print(list(words))
print('測試pos========')
words = jieba_fast_dat.posseg.cut(text,HMM=True)
print(list(words))


print("Test script finished.")