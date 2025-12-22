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
* **Python 版本限制**：我們擁抱現代開發！僅支持 **Python >= 3.10**。
* **linux only**: 不再支援 windows 降低維護複雜度

## changelog
- **pypi 累積安裝次數: 1.43k(20251222)**
- 20251222 優化快取結構, refactor c++, 大幅提昇效能, upgrade version to 0.58
- 20251221 優化使用者字典載入, 調整整體結構更多轉入c++ , 修復 IO 邏輯, 再次提昇效能, upgrade version to 0.57
- 20251204 強化cedar, 增加自定義字典cache機制, upgrade version to 0.56
- 20251124 整體大幅重構, 確保結果與原生jieba相同, 修復字典錯誤, upgrade version to 0.55
- 20251106 [0.54] 核心分詞引擎重構，將 Viterbi 完整遷移至 C++ 實現，執行效能大幅提升，並升級至 C++17 標準。

## 數字會說話：最高 **62 倍速** 的極致效能！

我們使用大型繁體字典（包含 **130 萬筆資料**）進行了深度效能對比。結果顯示，`jieba_fast_dat` 在各項指標上均徹底超越了原始 `jieba`。

### 效能對比數據 (Final Summary: Performance Comparison)

| 評測項目 (Metric) | 原生 Jieba | **jieba_fast_dat** | **加速倍率 (Speedup)** |
| :--- | :---: | :---: | :---: |
| **主字典載入 (Cold Init)** | 2.579 s | **1.035 s** | **2.49x** |
| **主字典載入 (With Cache)** | 1.847 s | **0.021 s** | **86.07x** |
| **HMM 模型載入 (Import Load)** | 0.110 s | **0.062 s** | **1.78x** |
| **自定義字典載入 (No Cache)** | 4.508 s | **1.591 s** | **2.83x** |
| **自定義字典載入 (With Cache)** | 5.592 s | **0.011 s** | **515.88x** |
| **分詞速度 (HMM=False)** | 0.843 s | **0.014 s** | **61.27x** |
| **詞性標注 (HMM=False)** | 0.909 s | **0.036 s** | **24.99x** |
| **分詞速度 (HMM=True)** | 0.962 s | **0.015 s** | **62.93x** |
| **詞性標注 (HMM=True)** | 1.013 s | **0.040 s** | **25.33x** |

> *測試環境：Linux, Python 3.12, 採用大型繁體字典進行測試，分詞/標註數據為多次執行之總和時間。*


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
pip install git+https://github.com/carycha/jieba_fast_dat@0.58
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
