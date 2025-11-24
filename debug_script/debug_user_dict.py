import os
from types import ModuleType

import jieba

import jieba_fast_dat

# --- 1. 定義路徑和測試資料 ---
# Define paths to our test resources
# 更好的方式是使用 pathlib，但為了保持與原始碼一致性，這裡繼續使用 os.path
try:
    TEST_DICTS_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "test", "test_dicts"
    )
except NameError:
    # 處理在交互式環境中 __file__ 未定義的情況
    TEST_DICTS_DIR = os.path.join(os.getcwd(), "..", "test", "test_dicts")

DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_dict_base.txt")
USER_DICT_BASE_PATH = os.path.join(TEST_DICTS_DIR, "test_user_dict_base.txt")

# 檢查路徑是否存在
if not os.path.exists(DICT_BASE_PATH) or not os.path.exists(USER_DICT_BASE_PATH):
    print("FATAL ERROR: Required dictionary files not found!")
    print(f"System Dict Path: {DICT_BASE_PATH}")
    print(f"User Dict Path: {USER_DICT_BASE_PATH}")
    # 這裡可以選擇退出程式碼
    # sys.exit(1)
    pass  # 為了讓程式碼可執行，這裡暫時pass

# 待測試的句子
SENTENCE = (
    "這是一個關於討論生成式AI的公司測試,郭台明明也喜歡蘋果20iphone。"
    "柳丁20orange自定義dict才有的字iphone16 covid-19 $89.64*&)$!$"
)

# 期望找到的詞彙
EXPECTED_IN_SYS_DICT = {
    "Word 1 (System)": "公司",
    "Word 2 (System)": "生成式AI",
    "Word 3 (System)": "蘋果20iphone",
}

# 期望在載入自定義字典後找到的詞彙
EXPECTED_IN_USER_DICT = {
    "Word 1 (User)": "郭台明明也",
    "Word 2 (User)": "自定義dict才有的字",
    "Word 3 (User)": "柳丁20orange",
}

# --- 2. 輔助函數：測試結果檢查 ---


def check_segmentation_results(
    segmented_words: list[str],
    expected_sys: dict[str, str],
    expected_user: dict[str, str],
    is_userdict_loaded: bool,
) -> bool:
    """檢查分詞結果是否包含期望的詞彙，並打印結構化的結果訊息。"""

    test_passed = True
    print("\n--- [ 🔍 詞彙檢查結果 ] ---")

    # 檢查系統字典詞彙
    print("\n--- 系統字典詞彙檢查 ---")
    for key, word in expected_sys.items():
        if word in segmented_words:
            print(f"✅ PASS: {key}: **'{word}'** 成功找到。")
        else:
            print(f"❌ FAIL: {key}: **'{word}'** 找不到！")
            test_passed = False

    # 檢查自定義字典詞彙
    print("\n--- 自定義字典詞彙檢查 ---")
    for key, word in expected_user.items():
        if is_userdict_loaded:
            # 期望找到 (UserDict 模式)
            if word in segmented_words:
                print(f"✅ PASS: {key}: **'{word}'** 成功找到 (自定義)。")
            else:
                print(f"❌ FAIL: {key}: **'{word}'** 找不到 (自定義)！")
                test_passed = False
        else:
            # 期望找不到 (僅 SystemDict 模式)
            if word not in segmented_words:
                print(f"✅ PASS: {key}: **'{word}'** 成功未找到 (非自定義模式下期望)。")
            else:
                # 在非 UserDict 模式下找到了 UserDict 詞彙，
                # 這可能是一個問題 (或 HMM 造成)
                # 這裡將其標記為一個 "注意" 或 "非預期"
                print(f"❌ FAIL: {key}: **'{word}'** 被找到 (非預期/HMM 可能導致)。")
                test_passed = False

    print("\n------------------------------")
    return test_passed


# --- 3. 輔助函數：測試單一場景 ---


def run_segmentation_test(
    library: ModuleType,
    lib_name: str,
    dict_path: str,
    userdict_path: str,
    hmm: bool,
    load_userdict: bool,
) -> bool:
    """運行單一組配置的測試邏輯。"""

    test_scenario = (
        f"{lib_name} | Dict: {'Sys+User' if load_userdict else 'System'} | HMM: {hmm}"
    )
    print(f"\n{'=' * 50}")
    print(f"🚀 開始測試: {test_scenario}")
    print(f"{'=' * 50}")

    # 1. 初始化字典
    try:
        library.set_dictionary(dict_path)
        if load_userdict:
            library.load_userdict(userdict_path)
        library.initialize()
    except Exception as e:
        print(f"❌ 初始化字典失敗: {e}")
        return False

    # # 2. 輸出 Debug 訊息 (僅針對 jieba_fast_dat)
    # if lib_name == "jieba_fast_dat" and hasattr(library, 'dt'):
    #     print("\n--- [ 🔍 DEBUG: 詞頻檢查 ] ---")
    #     # 檢查系統詞頻
    #     for key, word in EXPECTED_IN_SYS_DICT.items():
    #         freq = library.dt.get_freq(word)
    #         print(
    #             f"DEBUG: Sys_Dict {key} ('{word}'): Freq={freq}, "
    #             f"Prefix_Freq={library.dt.get_freq(word[:3])}"
    #         )

    #     # 檢查自定義詞頻 (只有載入時才有意義)
    #     if load_userdict:
    #         for key, word in EXPECTED_IN_USER_DICT.items():
    #             freq = library.dt.get_freq(word)
    #             print(
    #                 f"DEBUG: User_Dict {key} ('{word}'): Freq={freq}, "
    #                 f"Prefix_Freq={library.dt.get_freq(word[:3])}"
    #             )
    #     print("--------------------------------")

    # 3. 執行分詞
    result = list(library.cut(SENTENCE, HMM=hmm))
    print(f"\n[ Sentence ]:\n> {SENTENCE}")
    print(f"\n[ 分詞結果 ({'HMM' if hmm else 'No HMM'}) ]:\n> {result}")

    # 4. 檢查結果
    test_result = check_segmentation_results(
        result, EXPECTED_IN_SYS_DICT, EXPECTED_IN_USER_DICT, load_userdict
    )

    print(f"\n{'=' * 50}")
    print(
        f"🏁 測試完成: {test_scenario} -> **{'✅ 成功' if test_result else '❌ 失敗'}**"
    )
    print(f"{'=' * 50}")
    return test_result


# --- 4. 主測試流程 ---


def main() -> None:
    """執行所有測試場景。"""

    print("=" * 60)
    print("🚀 中文分詞庫測試腳本開始")
    print("=" * 60)

    print(f"📚 系統字典路徑: **{DICT_BASE_PATH}**")
    print(f"👤 自定義字典路徑: **{USER_DICT_BASE_PATH}**")
    print("-" * 60)

    print(f"待測試句子:\n> {SENTENCE}")
    print(f"期望的系統詞彙: {list(EXPECTED_IN_SYS_DICT.values())}")
    print(f"期望的自定義詞彙: {list(EXPECTED_IN_USER_DICT.values())}")
    print("-" * 60)

    # 定義所有測試場景
    TEST_CASES: list[tuple[ModuleType, str, bool, bool]] = [
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
        if not run_segmentation_test(
            library, lib_name, DICT_BASE_PATH, USER_DICT_BASE_PATH, hmm, load_userdict
        ):
            all_passed = False

    print("\n" + "=" * 60)
    print(
        "✅ 所有測試運行完畢，總體結果: "
        f"{'✅ ALL PASSED' if all_passed else '❌ FAILED'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
