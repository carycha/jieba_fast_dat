import jieba_fast_dat

# The initialize() method should be called automatically on the first cut
words = jieba_fast_dat.cut("测试一下DAT的实现")

for word in words:
    print(word)

print("Test script finished.")