import os
from types import ModuleType
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

# 期望的詞彙和詞性 (Word, Flag) - 需根據您的字典內容來確定準確的詞性和詞彙
# 注意: 由於我們使用的是自定義的測試字典，這裡的詞性標註（Flag）可能與標準 jieba
# 運行結果不同，請根據實際的預期結果調整。
EXPECTED_IN_SYS_DICT: dict[str, tuple[str, str]] = {
    "公司": ("公司", "n"),
    "生成式AI": ("生成式AI", "n"),
    "蘋果20iphone": ("蘋果20iphone", "n"),
}

# 期望在載入自定義字典後找到的詞彙和詞性
EXPECTED_IN_USER_DICT: dict[str, tuple[str, str]] = {
    "郭台明明也": ("郭台明明也", "n"),
    "自定義dict才有的字": ("自定義dict才有的字", "n"),
    "柳丁20orange": ("柳丁20orange", "n"),
}

# --- 2. 輔助函數：測試結果檢查 (更新為 POSseg 格式) ---


def check_segmentation_results(
    segmented_pairs: list[tuple[str, str]],
    expected_sys: dict[str, tuple[str, str]],
    expected_user: dict[str, tuple[str, str]],
    is_userdict_loaded: bool,
) -> bool:
    """檢查詞性標註結果是否包含期望的詞彙和詞性對，並打印結構化的結果訊息。"""

    test_passed = True
    # 將 (word, flag) 對列表轉換為字典，方便查找
    result_map = dict(segmented_pairs)

    print("\n--- [ 🔍 詞彙與詞性檢查結果 ] ---")

    # 檢查系統字典詞彙
    print("\n--- 系統字典詞彙檢查 ---")
    for word, expected_flag in expected_sys.values():
        if word in result_map:
            actual_flag = result_map[word]
            if actual_flag == expected_flag:
                print(f"✅ PASS: '{word}'/{actual_flag} 成功找到，詞性匹配。")
            else:
                print(
                    f"❌ FAIL: '{word}'/{actual_flag} "
                    f"詞彙找到，但詞性不符 (預期: {expected_flag})。"
                )
                test_passed = False
        else:
            print(f"❌ FAIL: '{word}' 找不到！")
            test_passed = False

    # 檢查自定義字典詞彙
    print("\n--- 自定義字典詞彙檢查 ---")
    for word, expected_flag in expected_user.values():
        if is_userdict_loaded:
            # 期望找到 (UserDict 模式)
            if word in result_map:
                actual_flag = result_map[word]
                if actual_flag == expected_flag:
                    print(
                        f"✅ PASS: '{word}'/{actual_flag} 成功找到 (自定義)，詞性匹配。"
                    )
                else:
                    print(
                        f"❌ FAIL: '{word}'/{actual_flag} "
                        "詞彙找到 (自定義)，但詞性不符 "
                        f"(預期: {expected_flag})。"
                    )
            else:
                print(f"❌ FAIL: '{word}' 找不到 (自定義)！")
                test_passed = False
        else:
            # 期望找不到 (僅 SystemDict 模式)
            if word not in result_map:
                print(f"✅ PASS: '{word}' 成功未找到 (非自定義模式下期望)。")
            else:
                # 在非 UserDict 模式下找到了 UserDict 詞彙，這通常是 HMM/新詞發現的結果
                print(f"❌ FAIL: '{word}' 被找到 (非預期/HMM 可能導致)。")
                test_passed = False

    print("\n------------------------------")
    return test_passed


# --- 3. 輔助函數：測試單一場景 ---


def run_posseg_test(
    library: ModuleType,
    lib_name: str,
    dict_path: str,
    userdict_path: str,
    hmm: bool,
    load_userdict: bool,
) -> bool:
    """運行單一組配置的詞性標註測試邏輯。"""

    test_scenario = (
        f"{lib_name} | Dict: {'Sys+User' if load_userdict else 'System'} | HMM: {hmm}"
    )
    print(f"\n{'=' * 60}")
    print(f"🚀 開始測試: 詞性標註 (POSseg) - {test_scenario}")
    print(f"{'=' * 60}")

    # 1. 初始化字典
    try:
        library.set_dictionary(dict_path)
        if load_userdict:
            library.load_userdict(userdict_path)
        library.initialize()
        library.posseg.initialize()
    except Exception as e:
        print(f"❌ 初始化字典失敗: {e}")
        return False

    # 2. 輸出 Debug 訊息 (僅針對 jieba_fast_dat)
    # if lib_name == "jieba_fast_dat" and hasattr(library, 'dt'):
    #     print("\n--- [ 🔍 DEBUG: 詞頻檢查 ] ---")
    #     # 檢查系統詞頻
    #     for key, (word, _) in EXPECTED_IN_SYS_DICT.items():
    #         freq = library.dt.get_freq(word)
    #         print(f"DEBUG: Sys_Dict {key} ('{word}'): Freq={freq}")

    #     # 檢查自定義詞頻 (只有載入時才有意義)
    #     if load_userdict:
    #         for key, (word, _) in EXPECTED_IN_USER_DICT.items():
    #             freq = library.dt.get_freq(word)
    #             print(f"DEBUG: User_Dict {key} ('{word}'): Freq={freq}")
    #     print("--------------------------------")

    # 3. 執行詞性標註
    # 使用 posseg.cut 替換 cut
    segmented_pairs = list(library.posseg.cut(SENTENCE, HMM=hmm))

    # 轉換結果為 (word, flag) 對列表
    result_list = [(w.word, w.flag) for w in segmented_pairs]

    print(f"\n[ Sentence ]:\n> {SENTENCE}")
    print(f"\n[ 詞性標註結果 ({'HMM' if hmm else 'No HMM'}) ]:\n> {result_list}")

    # 4. 檢查結果
    test_result = check_segmentation_results(
        result_list, EXPECTED_IN_SYS_DICT, EXPECTED_IN_USER_DICT, load_userdict
    )

    print(f"\n{'=' * 60}")
    print(
        f"🏁 測試完成: {test_scenario} -> **{'✅ 成功' if test_result else '❌ 失敗'}**"
    )
    print(f"{'=' * 60}")
    return test_result


# --- 4. 主測試流程 ---


def main() -> None:
    """執行所有詞性標註測試場景。"""

    print("=" * 60)
    print("🚀 中文分詞詞性標註 (POSseg) 測試腳本開始")
    print("=" * 60)

    print(f"📚 系統字典路徑: **{DICT_BASE_PATH}**")
    print(f"👤 自定義字典路徑: **{USER_DICT_BASE_PATH}**")
    print("-" * 60)

    print(f"待測試句子:\n> {SENTENCE}")
    print(f"期望的系統詞彙: {[w for w, _ in EXPECTED_IN_SYS_DICT.values()]}")
    print(f"期望的自定義詞彙: {[w for w, _ in EXPECTED_IN_USER_DICT.values()]}")
    print("-" * 60)

    # 定義所有測試場景
    TEST_CASES: list[tuple[Any, str, bool, bool]] = [
        # (library, lib_name, HMM, load_userdict)
        (jieba_fast_dat, "jieba_fast_dat", False, False),
        (jieba, "jieba (Origin)", False, False),
        (jieba_fast_dat, "jieba_fast_dat", True, False),
        (jieba, "jieba (Origin)", True, False),
        (jieba_fast_dat, "jieba_fast_dat", False, True),
        (jieba, "jieba (Origin)", False, True),
        (jieba_fast_dat, "jieba_fast_dat", True, True),
        (jieba, "jieba (Origin)", True, True),
    ]

    all_passed = True
    for library, lib_name, hmm, load_userdict in TEST_CASES:
        # 注意: 這裡我們直接傳遞 jieba/jieba_fast_dat 模組本身，
        # 因為 posseg 是它們的一個屬性
        if not run_posseg_test(
            library, lib_name, DICT_BASE_PATH, USER_DICT_BASE_PATH, hmm, load_userdict
        ):
            all_passed = False

        print("\n" + "~" * 60 + "\n")  # 增加間隔，更易讀

    print("\n" + "=" * 60)
    print(
        f"✅ 所有詞性標註測試運行完畢，總體結果: "
        f"**{'✅ ALL PASSED' if all_passed else '❌ FAILED'}**"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
