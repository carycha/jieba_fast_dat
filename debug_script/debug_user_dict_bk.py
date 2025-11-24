import os

import jieba

import jieba_fast_dat

# Define paths to our test resources
TEST_DICTS_DIR: str = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "test", "test_dicts"
)
dict_base_path: str = os.path.join(TEST_DICTS_DIR, "test_dict_base.txt")
user_dict_base_path: str = os.path.join(TEST_DICTS_DIR, "test_user_dict_base.txt")

print(f"DEBUG: Using dict_base_path: {dict_base_path}")
print(f"DEBUG: Using user_dict_base_path: {user_dict_base_path}")

SENTENCE: str = (
    "這是一個關於討論生成式AI的公司測試,郭台明明也喜歡蘋果20iphone。"
    "柳丁20orange自定義dict才有的字iphone16 covid-19 $89.64*&)$!$"
)
text_in_dict_1: str = "公司"
text_in_dict_2: str = "生成式AI"
text_in_dict_3: str = "蘋果20iphone"
text_in_user_dict_1: str = "郭台明明也"
text_in_user_dict_2: str = "自定義dict才有的字"
text_in_user_dict_3: str = "柳丁20orange"


print(f"測試字詞: {SENTENCE}")
print(f"只在系統字典裡面的字詞1: {text_in_dict_1}")
print(f"只在系統字典裡面的字詞2: {text_in_dict_2}")
print(f"只在系統字典裡面的字詞3: {text_in_dict_3}")
print(f"只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
print(f"只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
print(f"只在自定義字典裡面的字詞3: {text_in_user_dict_3}")


print("======先測試載入系統字典測試分詞效果 hmm False")

jieba_fast_dat.set_dictionary(dict_base_path)
jieba_fast_dat.initialize()

print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_2}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_2)}"
)
print(
    "DEBUG jieba_fast_dat.dt.get_freq("
    f"'{text_in_dict_2[:3]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_2[:3])}"
)  # Prefix of 生成式AI
print(
    "DEBUG jieba_fast_dat.dt.get_freq("
    f"'{text_in_dict_3}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_3)}"
)
print(
    "DEBUG jieba_fast_dat.dt.get_freq("
    f"'{text_in_dict_3[:4]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_3[:4])}"
)  # Prefix of 蘋果20iphone

result_chinese: list[str] = list(jieba_fast_dat.cut(SENTENCE, HMM=False))
print(f"jieba_fast_dat Segmented words(no HMM): {result_chinese}")
if text_in_dict_1 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞1: {text_in_dict_1}")
else:
    print(f"錯誤！找不到只在系統字典裡面的字詞1: {text_in_dict_1}")
if text_in_dict_2 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞2: {text_in_dict_2}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞2: {text_in_dict_2}")
if text_in_dict_3 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞3: {text_in_dict_3}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞3: {text_in_dict_3}")
if text_in_user_dict_1 in result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
if text_in_user_dict_2 in result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
if text_in_user_dict_3 in result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
print("-----------")

jieba.set_dictionary(dict_base_path)
jieba.initialize()
origin_result_chinese: list[str] = list(jieba.cut(SENTENCE, HMM=False))

print(f"origin_jieba Segmented words(no HMM): {origin_result_chinese}")
if text_in_dict_1 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞1: {text_in_dict_1}")
else:
    print(f"錯誤！找不到只在系統字典裡面的字詞1: {text_in_dict_1}")
if text_in_dict_2 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞2: {text_in_dict_2}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞2: {text_in_dict_2}")
if text_in_dict_3 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞3: {text_in_dict_3}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞3: {text_in_dict_3}")
if text_in_user_dict_1 in origin_result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
if text_in_user_dict_2 in origin_result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
if text_in_user_dict_3 in origin_result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
print("-----------")

print("======先測試載入系統字典測試分詞效果 hmm True")


jieba_fast_dat.set_dictionary(dict_base_path)
jieba_fast_dat.initialize()

print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_2}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_2)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_2[:3]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_2[:3])}"
)  # Prefix of 生成式AI
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_3}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_3)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_3[:4]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_3[:4])}"
)  # Prefix of 蘋果20iphone

result_chinese = list(jieba_fast_dat.cut(SENTENCE, HMM=True))
print(f"jieba_fast_dat Segmented words(HMM): {result_chinese}")
if text_in_dict_1 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞1: {text_in_dict_1}")
else:
    print(f"錯誤！找不到只在系統字典裡面的字詞1: {text_in_dict_1}")
if text_in_dict_2 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞2: {text_in_dict_2}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞2: {text_in_dict_2}")
if text_in_dict_3 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞3: {text_in_dict_3}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞3: {text_in_dict_3}")
if text_in_user_dict_1 in result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
if text_in_user_dict_2 in result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
if text_in_user_dict_3 in result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
print("-----------")

jieba.set_dictionary(dict_base_path)
jieba.initialize()
origin_result_chinese = list(jieba.cut(SENTENCE, HMM=True))

print(f"origin_jieba Segmented words(HMM): {origin_result_chinese}")
if text_in_dict_1 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞1: {text_in_dict_1}")
else:
    print(f"錯誤！找不到只在系統字典裡面的字詞1: {text_in_dict_1}")
if text_in_dict_2 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞2: {text_in_dict_2}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞2: {text_in_dict_2}")
if text_in_dict_3 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞3: {text_in_dict_3}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞3: {text_in_dict_3}")
if text_in_user_dict_1 in origin_result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
if text_in_user_dict_2 in origin_result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
if text_in_user_dict_3 in origin_result_chinese:
    print(f"錯誤!找到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
else:
    print(f"正常!找不到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
print("-----------")

print("======測試載入系統字典+自定義字典測試分詞效果 HMM False")
jieba_fast_dat.set_dictionary(dict_base_path)
jieba_fast_dat.load_userdict(user_dict_base_path)
jieba_fast_dat.initialize()

print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_2}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_2)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_2[:3]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_2[:3])}"
)  # Prefix of 生成式AI
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_3}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_3)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_3[:4]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_3[:4])}"
)  # Prefix of 蘋果20iphone
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_1}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_1)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_1[:2]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_1[:2])}"
)  # Prefix of 郭台銘
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_2}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_2)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_2[:5]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_2[:5])}"
)  # Prefix of 自定義dict才有的字
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_3}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_3)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_3[:4]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_3[:4])}"
)  # Prefix of 柳丁20orange

result_chinese = list(jieba_fast_dat.cut(SENTENCE, HMM=False))
print(f"jieba_fast_dat Segmented words(no HMM): {result_chinese}")

if text_in_dict_1 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞1: {text_in_dict_1}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞1: {text_in_dict_1}")
if text_in_dict_2 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞2: {text_in_dict_2}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞2: {text_in_dict_2}")
if text_in_dict_3 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞3: {text_in_dict_3}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞3: {text_in_dict_3}")
if text_in_user_dict_1 in result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
if text_in_user_dict_2 in result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
if text_in_user_dict_3 in result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
print("-----------")

jieba.set_dictionary(dict_base_path)
jieba.load_userdict(user_dict_base_path)
jieba.initialize()
origin_result_chinese = list(jieba.cut(SENTENCE, HMM=False))
print(f"origin jieba Segmented words(no HMM): {origin_result_chinese}")

if text_in_dict_1 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞1: {text_in_dict_1}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞1: {text_in_dict_1}")
if text_in_dict_2 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞2: {text_in_dict_2}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞2: {text_in_dict_2}")
if text_in_dict_3 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞3: {text_in_dict_3}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞3: {text_in_dict_3}")
if text_in_user_dict_1 in origin_result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
if text_in_user_dict_2 in origin_result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
if text_in_user_dict_3 in origin_result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
print("-----------")

print("======測試載入系統字典+自定義字典測試分詞效果 HMM True")
jieba_fast_dat.set_dictionary(dict_base_path)
jieba_fast_dat.load_userdict(user_dict_base_path)
jieba_fast_dat.initialize()

print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_2}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_2)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_2[:3]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_2[:3])}"
)  # Prefix of 生成式AI
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_3}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_3)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_dict_3[:4]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_dict_3[:4])}"
)  # Prefix of 蘋果20iphone
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_1}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_1)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_1[:2]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_1[:2])}"
)  # Prefix of 郭台銘
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_2}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_2)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_2[:5]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_2[:5])}"
)  # Prefix of 自定義dict才有的字
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_3}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_3)}"
)
print(
    f"DEBUG jieba_fast_dat.dt.get_freq('{text_in_user_dict_3[:4]}'): "
    f"{jieba_fast_dat.dt.get_freq(text_in_user_dict_3[:4])}"
)  # Prefix of 柳丁20orange

result_chinese = list(jieba_fast_dat.cut(SENTENCE, HMM=True))
print(f"jieba_fast_dat Segmented words(HMM): {result_chinese}")

if text_in_dict_1 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞1: {text_in_dict_1}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞1: {text_in_dict_1}")
if text_in_dict_2 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞2: {text_in_dict_2}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞2: {text_in_dict_2}")
if text_in_dict_3 in result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞3: {text_in_dict_3}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞3: {text_in_dict_3}")
if text_in_user_dict_1 in result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
if text_in_user_dict_2 in result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
if text_in_user_dict_3 in result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
print("-----------")

jieba.set_dictionary(dict_base_path)
jieba.load_userdict(user_dict_base_path)
jieba.initialize()
origin_result_chinese = list(jieba.cut(SENTENCE, HMM=True))
print(f"origin jieba Segmented words(HMM): {origin_result_chinese}")

if text_in_dict_1 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞1: {text_in_dict_1}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞1: {text_in_dict_1}")
if text_in_dict_2 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞2: {text_in_dict_2}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞2: {text_in_dict_2}")
if text_in_dict_3 in origin_result_chinese:
    print(f"正常!找到只在系統字典裡面的字詞3: {text_in_dict_3}")
else:
    print(f"錯誤!找不到只在系統字典裡面的字詞3: {text_in_dict_3}")
if text_in_user_dict_1 in origin_result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞1: {text_in_user_dict_1}")
if text_in_user_dict_2 in origin_result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞2: {text_in_user_dict_2}")
if text_in_user_dict_3 in origin_result_chinese:
    print(f"正常!找到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
else:
    print(f"錯誤!找不到只在自定義字典裡面的字詞3: {text_in_user_dict_3}")
print("-----------")
