from collections import Counter

import jieba
import jieba.analyse

import jieba_fast_dat
import jieba_fast_dat.analyse

# 測試文本
main_test_text = (
    "台灣的台北是一個充滿活力的城市，這裡有許多電腦和手機的程式設計師。"
    "他們正在開發區塊鏈和人工智慧的應用。賴清德和柯文哲是台灣的政治人物。"
    "館長和Joeman是知名的YouTuber。這些都是世界上的新趨勢。我喜歡學習深度學習的知識。"
    "這是一個很棒的時代。"
)
stop_words_path = "test/test_dicts/test_stop_words.txt"
idf_path = "test/test_dicts/text_idf_base.txt"
dict_path = "test/test_dicts/test_dict_base.txt"

jieba.set_dictionary(dict_path)
jieba_fast_dat.set_dictionary(dict_path)


jieba.analyse.set_stop_words(stop_words_path)
jieba_fast_dat.analyse.set_stop_words(stop_words_path)

jieba.analyse.set_idf_path(idf_path)
jieba_fast_dat.analyse.set_idf_path(idf_path)


jieba.initialize()
jieba_fast_dat.initialize()


# --- 比較分詞結果 ---
print("--- Comparing tokenization results ---")
orig_words = list(jieba.cut(main_test_text))
fast_words = list(jieba_fast_dat.cut(main_test_text))

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
            "  No differences in word counts, but order or other properties "
            "might differ."
        )


# --- 比較 IDF dictionaries ---
print("\n--- Comparing IDF dictionaries ---")
orig_idf_freq = jieba.analyse.default_tfidf.idf_freq
fast_idf_freq = jieba_fast_dat.analyse.default_tfidf.idf_freq

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
orig_stop_words = jieba.analyse.default_tfidf.stop_words
fast_stop_words = jieba_fast_dat.analyse.default_tfidf.stop_words

if orig_stop_words == fast_stop_words:
    print("Stop words are identical.")
else:
    print("Stop words are DIFFERENT.")
    diff = orig_stop_words ^ fast_stop_words
    print(f"  Differences: {diff}")


# --- 比較 TF-IDF results (reproducing test_with_stop_words) ---
print("\n--- Comparing TF-IDF results (reproducing test_with_stop_words) ---")
orig_tags = jieba.analyse.extract_tags(main_test_text, topK=5)
fast_tags = jieba_fast_dat.analyse.extract_tags(main_test_text, topK=5)

print(f"Original jieba TF-IDF tags: {orig_tags}")
print(f"jieba_fast_dat TF-IDF tags: {fast_tags}")

if orig_tags == fast_tags:
    print("TF-IDF results are identical.")
else:
    print("TF-IDF results are DIFFERENT.")
    print(f"  Original: {orig_tags}")
    print(f"  Fast:     {fast_tags}")

# --- 比較 TF-IDF withWeight results ---
print("\n--- Comparing TF-IDF withWeight results ---")
orig_tags_weight = jieba.analyse.extract_tags(main_test_text, withWeight=True, topK=5)
fast_tags_weight = jieba_fast_dat.analyse.extract_tags(
    main_test_text, withWeight=True, topK=5
)

print(f"Original jieba TF-IDF tags with weight: {orig_tags_weight}")
print(f"jieba_fast_dat TF-IDF tags with weight: {fast_tags_weight}")


# 比較浮點數時，需要考慮精度問題
def compare_float_lists(
    list1: list[tuple[str, float]], list2: list[tuple[str, float]], tol: float = 1e-9
) -> bool:
    if len(list1) != len(list2):
        return False
    for (word1, weight1), (word2, weight2) in zip(list1, list2, strict=True):
        if word1 != word2 or abs(weight1 - weight2) > tol:
            return False
    return True


if compare_float_lists(orig_tags_weight, fast_tags_weight):
    print("TF-IDF withWeight results are identical.")
else:
    print("TF-IDF withWeight results are DIFFERENT.")
    print(f"  Original: {orig_tags_weight}")
    print(f"  Fast:     {fast_tags_weight}")
