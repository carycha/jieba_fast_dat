# jieba_fast_dat: 高效能中文分詞與詞性標註工具

由於自己在使用時發現隨著字典的增加, 字典載入速度越來越久(甚至超過 10 秒),
且原始 [`jieba`](https://github.com/fxsjy/jieba) 與 [`jieba_fast`](https://github.com/deepcs233/jieba_fast) 由於久未維護, 有些依賴已經與現在主流 python 版本已經有警告訊息出現(看著不舒服)

所以在支援原有功能的狀態下(大部分), 進行更新與開發, 主要優化內容如下:

## 技術優化內容

*   **DAT 詞典結構**: 詞典採用均 Double-Array Trie (DAT) 結構，實現低記憶體佔用和極速查詢。
*   **C++ 核心算法**: 關鍵算法（如 Viterbi）在 C++ 中實現，並透過 `pybind11` 無縫暴露給 Python，結合了 Python 的靈活性和 C++ 的高效能。
*   **CPU 優先原則**: 所有算法和庫的選擇都符合 CPU 執行效率，不依賴 GPU。
*   **繁體強化**: 將預設的系統字典與 idf 均直接改用 `jieba` 原廠提供的繁體優化字典, 無須額外修改設定

## 重大差異：為了極速，我們做出一個取捨

* **不支援動態增加字典**：為了實現 DAT 結構的極速查詢和持久化快取，我們移除了運行時動態增加字典的功能。
    > ** 替代方案：** 您只需將新字典加入字典檔，重新啟動程式，**快取將自動更新**，依然享受極速！
* **Python 版本限制**：我們擁抱現代開發！僅支持 **Python >= 3.10**。

## changelog
- 20251204 強化cedar, 增加自定義字典cache機制, upgrade version to 0.56
- 20251124 整體大幅重構, 確保結果與原生jieba相同, 修復字典錯誤, upgrade version to 0.55
- 20251106 add pypi install version 0.54
- 20251106 [0.54] 核心分詞引擎重構，將 Viterbi 完整遷移至 C++ 實現，執行效能大幅提升，並升級至 C++17 標準。
- 20251102 增加 memory-leak 測試以避免 python, c++ memory leak
- 20251102 重翻 c++程式, 移除無效程式, 優化 dat 效能

## 數字會說話：字典載入速度 **94.83%** 的巨大提升！

我們用一個包含 **130 萬筆資料**的超大型字典進行了對比。結果顯示：不論是第一次init還是在第二次使用快取時，我們的提升幅度是**巨大**！

|| 初次 init 花費時間 | cached 花費時間|cache 提昇速度%|
|---|---|---|---|
|jieba_fast| 6.00 s| 4.76 s| 20.69% |
|**jieba_fast_dat**| **1.58 s**| **0.25 s**| **84.48%** |
|dat 提昇速度% | **73.59%** | **94.83%** | |

## 介紹影片
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/nmaEbAAgpno/0.jpg)](https://www.youtube.com/watch?v=nmaEbAAgpno)


## 🚀 安裝
pypi 安裝最新
```bash
pip install jieba_fast_dat
```

github 安裝最新
```bash
pip install git+https://github.com/carycha/jieba_fast_dat
```
github 安裝指定版號
```bash
pip install git+https://github.com/carycha/jieba_fast_dat@0.55
```

## 🛠️ 使用方式

### 基本分詞

```python
import jieba_fast_dat as jieba

text = "雨要下到什麼時候？氣象署：今雨勢最猛　週日長榮馬拉松要穿雨衣"
print("精確模式:", "/".join(jieba.cut(text)))
print("全模式:", "/".join(jieba.cut(text, cut_all=True)))
print("搜尋引擎模式:", "/".join(jieba.cut_for_search(text)))
```

### 詞性標註

```python
import jieba_fast_dat.posseg as pseg

text = "雨要下到什麼時候？氣象署：今雨勢最猛　週日長榮馬拉松要穿雨衣"
words = pseg.cut(text)
for word, flag in words:
    print(f"{word}/{flag}")
```

### 載入使用者詞典

```python
import jieba_fast_dat as jieba

# userdict.txt 範例內容:
# 創新模式 3
# 程式設計 5 n
jieba.load_userdict("userdict.txt")
print("載入使用者詞典後:", "/".join(jieba.cut("雨要下到什麼時候？氣象署：今雨勢最猛　週日長榮馬拉松要穿雨衣")))
```

## 分詞與詞性標註結果比較
統一用以下文字測試

```
東北季風發威！4縣市豪大雨特報「雨下整夜」　一路濕到這天
```
### 分詞差異
|模式 | 原始 jieba_fast | **jieba_fast_dat** |
|---|---|---|
|HMM OFF|東/北/季/風/發/威/！/4/縣/市/豪/大雨/特/報/「/雨/下/整夜/」/　/一路/濕/到/這/天|**東北/季風/發威/！/4/縣市/豪/大雨/特報/「/雨/下/整夜/」/　/一路/濕/到/這天**|
|HMM ON|東北/季風/發威/！/4/縣市/豪/大雨/特報/「/雨下/整夜/」/　/一路/濕到/這天|**東北/季風/發威/！/4/縣市/豪/大雨/特報/「/雨下/整夜/」/　/一路/濕到/這天**|
### 詞性標注差異
|模式 | 原始 jieba_fast | **jieba_fast_dat** |
|---|---|---|
|HMM OFF| 東/zg 北/ns 季/n 風/zg 發/zg 威/ns ！/x 4/eng 縣/x 市/n 豪/n 大雨/n 特/d 報/zg 「/x 雨/n 下/f 整夜/b 」/x  /x 一路/m 濕/x 到/v 這/zg 天/q | **東北/ns 季風/n 發威/v ！/x 4/eng 縣市/n 豪/n 大雨/n 特報/n 「/x 雨/n 下/f 整夜/b 」/x 　/x 一路/m 濕/x 到/v 這天/r**|
|HMM ON| 東北/ns 季風/n 發威/v ！/x 4/m 縣/n 市豪/n 大雨/n 特報/n 「/x 雨/n 下/f 整夜/b 」/x 　/x 一路/m 濕到/v 這天/r| **東北/ns 季風/n 發威/v ！/x 4/x 縣市/n 豪/n 大雨/n 特報/n 「/x 雨/n 下/f 整夜/b 」/x 　/x 一路/m 濕到/x 這天/r**|

## 支持與鼓勵
如果您重視效率、速度、穩定性，並認同我們為中文 NLP 提昇的小小貢獻：

⭐ 點擊 Star！ 您的肯定是我們持續開發的最大動力！

📢 轉發擴散！ 讓所有還在飽受載入慢之苦的開發者知道這個工具！

🤝 提出 Issue/PR！ 歡迎加入我們，讓這個神器更加完美！

## 📄 許可證

`jieba_fast_dat` 採用 MIT 許可證。詳情請參閱 `LICENSE` 文件。

## 🤝 貢獻

歡迎任何形式的貢獻！如果您有任何建議、功能請求或錯誤報告，請隨時提出 Issue 或提交 Pull Request。

## 🌟 鳴謝

本專案基於 [jieba](https://github.com/fxsjy/jieba) 與 [jieba_fast](https://github.com/deepcs233/jieba_fast) 庫進行優化和增強。感謝原作者及所有貢獻者。
