import jieba
import jieba.analyse

import jieba_fast_dat
import jieba_fast_dat.analyse

# 測試文本
main_test_text = (
    "台灣的台北是一個充滿活力的城市，這裡有許多電腦和手機的程式設計師。"
    "他們正在開發區塊鏈和人工智慧的應用。賴清德和柯文哲是台灣的政治人物。"
    "館長和Joeman是知名的YouTuber。這些都是世界上的新趨勢。"
    "我喜歡學習深度學習的知識。這是一個很棒的時代。"
)
stop_words_path = "test/test_dicts/test_stop_words.txt"
idf_path = "test/test_dicts/text_idf_base.txt"
dict_path = "test/test_dicts/test_dict_base.txt"

# 初始化 jieba 和 jieba_fast_dat 的字典
jieba.dt.initialize(dict_path)
jieba_fast_dat.dt.initialize(dict_path)

# 初始化原始 jieba 的 TF-IDF 提取器
orig_tfidf_extractor = jieba.analyse.TFIDF(idf_path=idf_path)
# 原始 jieba 的 set_stop_words 是全局的，它會修改 TFIDF 類的 stop_words 屬性
jieba.analyse.set_stop_words(stop_words_path)
# 重新初始化 orig_tfidf_extractor，使其使用更新後的 TFIDF 類的 stop_words 屬性
orig_tfidf_extractor = jieba.analyse.TFIDF(idf_path=idf_path)

# 初始化 jieba_fast_dat 的 TF-IDF 提取器
fast_tfidf_extractor = jieba_fast_dat.analyse.TFIDF(idf_path=idf_path)
# jieba_fast_dat 的 set_stop_words 是全局的，它會修改 TFIDF 類的 stop_words 屬性
jieba_fast_dat.analyse.set_stop_words(stop_words_path)
# 重新初始化 fast_tfidf_extractor，使其使用更新後的 TFIDF 類的 stop_words 屬性
fast_tfidf_extractor = jieba_fast_dat.analyse.TFIDF(idf_path=idf_path)


# --- 比較分詞結果 ---
print("--- Comparing tokenization results ---")
orig_words = list(orig_tfidf_extractor.tokenizer.cut(main_test_text))
fast_words = list(fast_tfidf_extractor.tokenizer.cut(main_test_text))

print(f"Original jieba tokenization (first 20): {orig_words[:20]}")
print(f"jieba_fast_dat tokenization (first 20): {fast_words[:20]}")

if orig_words == fast_words:
    print("Tokenization results are identical.")
else:
    print("Tokenization results are DIFFERENT.")
    from collections import Counter

    orig_counts = Counter(orig_words)
    fast_counts = Counter(fast_words)
    all_words = sorted(set(orig_words) | set(fast_words))
    diff_found = False
    for word in all_words:
        if orig_counts[word] != fast_counts[word]:
            print(
                f"  Word '{word}': original count {orig_counts[word]}, "
                f"fast count {fast_counts[word]}"
            )
            diff_found = True
    if not diff_found:
        print(
            "  No differences in word counts, but order or other "
            "properties might differ."
        )


# --- 比較 IDF dictionaries ---
print("\n--- Comparing IDF dictionaries ---")
orig_idf_freq = orig_tfidf_extractor.idf_freq
fast_idf_freq = fast_tfidf_extractor.idf_freq

if orig_idf_freq == fast_idf_freq:
    print("IDF dictionaries are identical.")
else:
    print("IDF dictionaries are DIFFERENT.")
    all_keys = sorted(set(orig_idf_freq.keys()) | set(fast_idf_freq.keys()))
    diff_count = 0
    for key in all_keys:
        orig_val = orig_idf_freq.get(key)
        fast_val = fast_idf_freq.get(key)
        if orig_val != fast_val:
            diff_count += 1
            if diff_count < 20:  # 只顯示前 20 個差異
                print(
                    f"  Key '{key}': original value {orig_val}, fast value {fast_val}"
                )
    if diff_count >= 20:
        print(f"  ... and {diff_count - 19} more differences.")

# --- 比較停用詞 ---
print("\n--- Comparing stop words ---")
# 原始 jieba 的 stop_words 是 KeywordExtractor 的類屬性，
# 但通過 set_stop_words 修改的是 TFIDF 實例的 stop_words
# 這裡需要確保獲取的是正確的實例屬性
orig_stop_words = orig_tfidf_extractor.stop_words
fast_stop_words = fast_tfidf_extractor.stop_words

if orig_stop_words == fast_stop_words:
    print("Stop words are identical.")
else:
    print("Stop words are DIFFERENT.")
    diff = orig_stop_words ^ fast_stop_words
    print(f"  Differences: {diff}")


# --- 比較 TF-IDF results (reproducing test_with_stop_words) ---
print("\n--- Comparing TF-IDF results (reproducing test_with_stop_words) ---")
orig_tags = orig_tfidf_extractor.extract_tags(main_test_text, topK=5)
fast_tags = fast_tfidf_extractor.extract_tags(main_test_text, topK=5)

print(f"Original jieba TF-IDF tags: {orig_tags}")
print(f"jieba_fast_dat TF-IDF tags: {fast_tags}")

if orig_tags == fast_tags:
    print("TF-IDF results are identical.")
else:
    print("TF-IDF results are DIFFERENT.")
    print(f"  Original: {orig_tags}")
    print(f"  Fast:     {fast_tags}")
