import os
from typing import Any

import jieba
import jieba.analyse  # 引入關鍵詞提取模組

import jieba_fast_dat
import jieba_fast_dat.analyse  # 引入關鍵詞提取模組

# --- 1. 定義路徑和測試資料 ---
# Define paths to our test resources
try:
    TEST_DICTS_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "test", "test_dicts"
    )
except NameError:
    # 處理在交互式環境中 __file__ 未定義的情況
    TEST_DICTS_DIR = os.path.join(os.getcwd(), "..", "test", "test_dicts")

# 系統/用戶字典
DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_dict_base.txt")
USER_DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_user_dict_base.txt")
# 關鍵詞提取所需字典
TEST_IDF_PATH = os.path.join(TEST_DICTS_DIR, "text_idf_base.txt")  # 🚨 必須存在！
TEST_STOPWORDS_PATH = os.path.join(
    TEST_DICTS_DIR, "test_soptword_base.txt"
)  # 🚨 必須存在！

# 檢查路徑是否存在 (關鍵詞提取字典也必須檢查)
paths_to_check = {
    "System Dict": DICT_BASE_PATH,
    "User Dict": USER_DICT_BASE_PATH,
    "IDF Dict": TEST_IDF_PATH,
    "Stopwords Dict": TEST_STOPWORDS_PATH,
}

for name, path in paths_to_check.items():
    if not os.path.exists(path):
        print(f"⚠️ WARNING: Required file not found for {name}. Tests may fail.")
        print(f"Path: {path}")

# 待測試的句子
SENTENCE = (
    "這是一個關於討論生成式AI的公司測試,郭台明明也喜歡蘋果20iphone。"
    "柳丁20orange自定義dict才有的字iphone16 covid-19 $89.64*&)$!$"
)

# 期望的關鍵詞彙 (Word, 期望的最低權重, 是否為UserDict詞)
# 🚨 注意: 這裡的期望結果需要根據您的 IDF/Stopword 檔案和字典內容手動調整！
EXPECTED_KEYWORDS: dict[str, tuple[float, bool]] = {
    "生成式AI": (1.0, False),  # 系統詞彙
    "公司": (0.8, False),  # 系統詞彙
    "iphone16": (0.6, False),  # 不在字典但本來就應該切出來
    "郭台明明也": (0.4, True),  # 自定義詞彙
    "柳丁20orange": (0.2, True),  # 自定義詞彙
}

# --- 2. 輔助函數：測試結果檢查 ---


def check_tfidf_results(
    keywords_pairs: list[tuple[str, float]],
    expected_keywords: dict[str, tuple[float, bool]],
    is_userdict_loaded: bool,
) -> bool:
    """檢查 tfidf 結果是否包含期望的關鍵詞彙和權重，並打印結構化的結果訊息。"""

    test_passed = True
    # 將 (word, weight) 對列表轉換為字典，方便查找
    result_map = dict(keywords_pairs)

    print("\n--- [ 🔍 tfidf 關鍵詞檢查結果 ] ---")

    for word, (expected_min_weight, is_user_dict_word) in expected_keywords.items():
        key_status = "系統詞彙" if not is_user_dict_word else "自定義詞彙"

        if is_user_dict_word and not is_userdict_loaded:
            # 在未載入自定義字典的情況下，期望找不到自定義詞彙
            if word not in result_map:
                print(f"✅ PASS: **'{word}'** 成功未找到 (非自定義模式下期望)。")
            else:
                print(f"❌ FAIL: **'{word}'** 被找到 (非預期/HMM 可能導致)。")
                test_passed = False
            continue

        # 檢查關鍵詞
        if word in result_map:
            actual_weight = result_map[word]
            if actual_weight >= expected_min_weight:
                print(
                    f"✅ PASS: **'{word}'** ({key_status}) 成功找到，"
                    f"權重 **{actual_weight:.4f}** >= "
                    f"預期最小值 **{expected_min_weight:.4f}**。"
                )
            else:
                print(
                    f"❌ FAIL: **'{word}'** ({key_status}) 詞彙找到，但權重太低 "
                    f"({actual_weight:.4f} < 預期最小值 {expected_min_weight:.4f})。"
                )
                test_passed = False
        else:
            if not is_user_dict_word or is_userdict_loaded:
                # 期望找到 (系統詞彙或已載入自定義字典)
                print(f"❌ FAIL: **'{word}'** ({key_status}) 找不到！")
                test_passed = False

    print("\n------------------------------")
    return test_passed


# --- 3. 輔助函數：測試單一場景 ---


def run_tfidf_test(
    library: Any,  # noqa: ANN401
    lib_name: str,
    dict_path: str,
    userdict_path: str,
    idf_path: str,
    stopword_path: str,
    load_userdict: bool,
):
    """運行單一組配置的 tfidf 關鍵詞提取測試邏輯。"""

    # 由於 tfidf 不需要 HMM，這裡只測試載入字典的差異
    test_scenario = f"{lib_name} | Dict: {'Sys+User' if load_userdict else 'System'}"
    print(f"\n{'=' * 60}")
    print(f"🚀 開始測試: tfidf 關鍵詞提取 - {test_scenario}")
    print(f"{'=' * 60}")

    # 1. 初始化字典和 tfidf 相關字典
    try:
        # 載入系統字典
        library.set_dictionary(dict_path)

        # 載入自定義字典
        if load_userdict:
            library.load_userdict(userdict_path)

        # 初始化 tfidf 字典
        print(f"📝 載入 IDF 字典: {idf_path}")
        library.analyse.set_idf_path(idf_path)

        print(f"📝 載入 Stopword 字典: {stopword_path}")
        library.analyse.set_stop_words(stopword_path)

        # 初始化（這會確保字典被載入到核心結構中）
        library.initialize()

    except Exception as e:
        print(f"❌ 初始化字典失敗: {e}")
        return False

    # 2. 執行 tfidf 關鍵詞提取
    # tfidf 預設是不使用 HMM 的，且它會進行詞性篩選。
    # 這裡使用 withWeight=True 來獲取權重
    keywords_pairs = library.analyse.extract_tags(
        SENTENCE,
        topK=5,
        withWeight=True,
        # allowPOS=('ns', 'n', 'vn', 'v', 'nz') # tfidf 預設的詞性
    )

    # tfidf 產出的是 (Word, Weight) 的列表
    print(f"\n[ Sentence ]:\n> {SENTENCE}")
    print(f"\n[ tfidf 關鍵詞結果 (Top 5) ]:\n> {keywords_pairs}")

    # 3. 檢查結果
    test_result = check_tfidf_results(keywords_pairs, EXPECTED_KEYWORDS, load_userdict)

    print(f"\n{'=' * 60}")
    print(
        f"🏁 測試完成: {test_scenario} -> **{'✅ 成功' if test_result else '❌ 失敗'}**"
    )
    print(f"{'=' * 60}")
    return test_result


# --- 4. 主測試流程 ---


def main():
    """執行所有 tfidf 測試場景。"""

    print("=" * 60)
    print("🚀 中文分詞 tfidf 關鍵詞提取測試腳本開始")
    print("=" * 60)

    print(f"📚 系統字典路徑: **{DICT_BASE_PATH}**")
    print(f"👤 自定義字典路徑: **{USER_DICT_BASE_PATH}**")
    print(f"🔢 IDF 字典路徑: **{TEST_IDF_PATH}**")
    print(f"🛑 Stopwords 字典路徑: **{TEST_STOPWORDS_PATH}**")
    print("-" * 60)

    print(f"待測試句子:\n> {SENTENCE}")
    print(f"期望的關鍵詞彙: {list(EXPECTED_KEYWORDS.keys())}")
    print("-" * 60)

    # 定義所有測試場景
    # tfidf 測試只需要關注是否載入 User Dict
    TEST_CASES = [
        # (library, lib_name, load_userdict)
        (jieba_fast_dat, "jieba_fast_dat", False),
        (jieba, "jieba (Origin)", False),
        (jieba_fast_dat, "jieba_fast_dat", True),
        (jieba, "jieba (Origin)", True),
    ]

    all_passed = True
    for library, lib_name, load_userdict in TEST_CASES:
        if not run_tfidf_test(
            library,
            lib_name,
            DICT_BASE_PATH,
            USER_DICT_BASE_PATH,
            TEST_IDF_PATH,
            TEST_STOPWORDS_PATH,
            load_userdict,
        ):
            all_passed = False

        print("\n" + "~" * 60 + "\n")  # 增加間隔，更易讀

    print("\n" + "=" * 60)
    print(
        f"✅ 所有 tfidf 測試運行完畢，總體結果: "
        f"**{'✅ ALL PASSED' if all_passed else '❌ FAILED'}**"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
