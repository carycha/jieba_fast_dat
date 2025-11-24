import os
from typing import Any

import jieba
import jieba.posseg  # 引入標準 jieba 的詞性標註模組

import jieba_fast_dat
import jieba_fast_dat.posseg  # 引入標準 jieba 的詞性標註模組

# --- 1. 定義路徑和測試資料 ---
# Define paths to our test resources
try:
    TEST_DICTS_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "test", "test_dicts"
    )
except NameError:
    # 處理在交互式環境中 __file__ 未定義的情況
    TEST_DICTS_DIR = os.path.join(os.getcwd(), "..", "test", "test_dicts")

DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_dict_base.txt")
USER_DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_user_dict_base.txt")

# 檢查路徑是否存在 (可根據實際情況調整)
if not os.path.exists(DICT_BASE_PATH) or not os.path.exists(USER_DICT_BASE_PATH):
    print("⚠️ WARNING: Required dictionary files not found. Tests may fail.")
    print(f"System Dict Path: {DICT_BASE_PATH}")
    print(f"User Dict Path: {USER_DICT_BASE_PATH}")

# 待測試的句子
SENTENCE = (
    "這是一個關於討論生成式AI的公司測試,郭台明明也喜歡蘋果20iphone。"
    "柳丁20orange自定義dict才有的字iphone16 covid-19 $89.64*&)$!$"
)

# --- 2. 輔助函數：獲取詞性標註結果 ---


def get_posseg_result(
    library: Any,  # noqa: ANN401
    dict_path: str,
    userdict_path: str,
    hmm: bool,
    load_userdict: bool,
    sentence: str,
) -> list[tuple[str, str]]:
    """
    獲取指定庫在特定配置下的詞性標註結果。
    """
    # 1. 初始化字典
    try:
        library.set_dictionary(dict_path)
        if load_userdict:
            library.load_userdict(userdict_path)
        library.initialize()
        library.posseg.initialize()  # 確保 posseg 模塊使用最新的詞典
    except Exception as e:
        print(f"❌ 初始化字典失敗: {e}")
        return []

    # 2. 執行詞性標註
    segmented_pairs = list(library.posseg.cut(sentence, HMM=hmm))

    # 轉換結果為 (word, flag) 對列表
    result_list = [(w.word, w.flag) for w in segmented_pairs]
    return result_list


# --- 3. 輔助函數：測試單一場景 ---


def run_posseg_test(
    fast_dat_library: Any,  # noqa: ANN401
    lib_name: str,
    dict_path: str,
    userdict_path: str,
    hmm: bool,
    load_userdict: bool,
    sentence: str,
) -> bool:
    """
    運行單一組配置的詞性標註測試邏輯，並與標準 jieba 進行比較。
    """

    test_scenario = (
        f"{lib_name} | Dict: {'Sys+User' if load_userdict else 'System'} | HMM: {hmm}"
    )
    print(f"\n{'=' * 60}")
    print(f"🚀 開始測試: 詞性標註 (POSseg) - {test_scenario}")
    print(f"{'=' * 60}")

    # 1. 獲取標準 jieba 的結果作為黃金標準
    jieba_golden_standard = get_posseg_result(
        jieba, dict_path, userdict_path, hmm, load_userdict, sentence
    )
    print(
        f"\n[ 標準 Jieba 結果 ({'HMM' if hmm else 'No HMM'}) ]:\n"
        f"> {jieba_golden_standard}"
    )

    # 2. 獲取 jieba_fast_dat 的結果
    fast_dat_result = get_posseg_result(
        fast_dat_library, dict_path, userdict_path, hmm, load_userdict, sentence
    )
    print(
        f"\n[ Jieba_fast_dat 結果 ({'HMM' if hmm else 'No HMM'}) ]:\n"
        f"> {fast_dat_result}"
    )

    # 3. 比較兩個結果
    test_passed = jieba_golden_standard == fast_dat_result

    print("\n--- [ 🔍 結果比較 ] ---")
    if test_passed:
        print(f"✅ PASS: {lib_name} 的結果與標準 Jieba 完全一致。")
    else:
        print(f"❌ FAIL: {lib_name} 的結果與標準 Jieba 不一致！")
        print("\n--- 差異詳情 ---")
        print(f"標準 Jieba: {jieba_golden_standard}")
        print(f"Jieba_fast_dat: {fast_dat_result}")
        # 更詳細的差異比較 (可選)
        # for i, (j_word, j_flag) in enumerate(jieba_golden_standard):
        #     if i < len(fast_dat_result):
        #         f_word, f_flag = fast_dat_result[i]
        #         if j_word != f_word or j_flag != f_flag:
        #             print(f"位置 {i}: 標準 Jieba: {j_word}/{j_flag}, "
        #                   f"Jieba_fast_dat: {f_word}/{f_flag}")
        #     else:
        #         print(f"位置 {i}: 標準 Jieba: {j_word}/{j_flag}, "
        #               f"Jieba_fast_dat: (無)")
        # if len(fast_dat_result) > len(jieba_golden_standard):
        #     for i in range(len(jieba_golden_standard), len(fast_dat_result)):
        #         f_word, f_flag = fast_dat_result[i]
        #         print(f"位置 {i}: 標準 Jieba: (無), "
        #               f"Jieba_fast_dat: {f_word}/{f_flag}")
        print("--------------------")

    print(f"\n{'=' * 60}")
    print(
        f"🏁 測試完成: {test_scenario} -> **{'✅ 成功' if test_passed else '❌ 失敗'}**"
    )
    print(f"{'=' * 60}")
    return test_passed


# --- 4. 主測試流程 ---


def main():
    """執行所有詞性標註測試場景。"""

    print("=" * 60)
    print("🚀 中文分詞詞性標註 (POSseg) 兼容性測試腳本開始")
    print("=" * 60)

    print(f"📚 系統字典路徑: **{DICT_BASE_PATH}**")
    print(f"👤 自定義字典路徑: **{USER_DICT_BASE_PATH}**")
    print("-" * 60)

    print(f"待測試句子:\n> {SENTENCE}")
    print("-" * 60)

    # 定義所有測試場景 (只針對 jieba_fast_dat 進行測試，並與標準 jieba 比較)
    TEST_CASES = [
        # (library, lib_name, HMM, load_userdict)
        (jieba_fast_dat, "jieba_fast_dat", False, False),
        (jieba_fast_dat, "jieba_fast_dat", True, False),
        (jieba_fast_dat, "jieba_fast_dat", False, True),
        (jieba_fast_dat, "jieba_fast_dat", True, True),
    ]

    all_passed = True
    for library, lib_name, hmm, load_userdict in TEST_CASES:
        if not run_posseg_test(
            library,
            lib_name,
            DICT_BASE_PATH,
            USER_DICT_BASE_PATH,
            hmm,
            load_userdict,
            SENTENCE,
        ):
            all_passed = False

        print("\n" + "~" * 60 + "\n")  # 增加間隔，更易讀

    print("\n" + "=" * 60)
    print(
        f"✅ 所有詞性標註兼容性測試運行完畢，總體結果: "
        f"**{'✅ ALL PASSED' if all_passed else '❌ FAILED'}**"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
