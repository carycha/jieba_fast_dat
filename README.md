# jieba_fast_dat: 高效能中文分詞與詞性標註工具

`jieba_fast_dat` 是一個專注於中文文本處理的高效能分詞與詞性標註工具。它基於 `jieba` 庫，並透過引入 Double-Array Trie (DAT) 結構、持久化快取機制以及 C++ 實現的核心算法，顯著提升了分詞速度和啟動效能。

## ✨ 主要功能

*   **高效中文分詞**: 提供精確模式、全模式和搜尋引擎模式，快速準確地切分中文詞語。
*   **詞性標註 (POS Tagging)**: 能夠對分詞結果進行詞性標註，提供更豐富的語言分析能力。
*   **DAT 詞典結構**: 核心詞典採用 Double-Array Trie (DAT) 結構，實現低記憶體佔用和極速查詢。
*   **持久化快取機制**: 引入基於 MD5 雜湊的動態、持久化且自動失效的 DAT 快取，顯著加速後續詞典載入，實現近乎瞬時的啟動速度。
*   **C++ 核心算法**: 關鍵算法（如 Viterbi）在 C++ 中實現，並透過 `pybind11` 無縫暴露給 Python，結合了 Python 的靈活性和 C++ 的高效能。
*   **關鍵詞提取**: 提供基於 TF-IDF 和 TextRank 兩種算法的關鍵詞提取功能。
*   **使用者詞典支援**: 支援載入使用者自訂詞典。
*   **多執行緒安全**: 詞典初始化過程採用 `threading.RLock` 保護，確保多執行緒環境下的穩定性。
*   **CPU 優先原則**: 所有算法和庫的選擇都符合 CPU 執行效率，不依賴 GPU。

## 🚀 安裝

確保您已安裝 `uv` (高性能的 Python 套件管理工具)。

```bash
uv pip install jieba_fast_dat
```

## 🛠️ 使用方式

### 基本分詞

```python
import jieba_fast_dat as jieba

text = "我愛北京天安門"
print("精確模式:", "/".join(jieba.cut(text)))
print("全模式:", "/".join(jieba.cut(text, cut_all=True)))
print("搜尋引擎模式:", "/".join(jieba.cut_for_search(text)))
```

### 詞性標註

```python
import jieba_fast_dat.posseg as pseg

text = "我愛北京天安門"
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
print("載入使用者詞典後:", "/".join(jieba.cut("我喜歡創新模式的程式設計")))
```

### 效能優化與快取

`jieba_fast_dat` 會自動管理詞典的持久化快取。首次載入詞典時會稍慢，但後續載入將利用快取實現極速啟動。

## 👨‍💻 開發設定 (Development Setup)

如果您希望參與 `jieba_fast_dat` 的開發，請遵循以下步驟設定您的環境：

1.  **複製專案**
    ```bash
    git clone https://github.com/cary-cha/jieba_fast_dat.git
    cd jieba_fast_dat
    ```

2.  **建立並啟動虛擬環境**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **安裝開發依賴**
    使用以下指令安裝所有用於開發、測試和建置的套件。`-e` 參數會以可編輯模式安裝，讓您的程式碼變更能立即生效。
    ```bash
    uv pip install -e .[dev]
    ```

## 🧪 測試 (Testing)

在設定好開發環境後，您可以使用以下指令執行完整的測試套件：

```bash
uv run pytest
```

## 📦 建置 (Building)

在設定好開發環境後，您可以使用以下指令建置原始碼和 Wheel 發行版：

```bash
python -m build
```
建置產物將會出現在 `dist/` 目錄下。

## 📄 許可證

`jieba_fast_dat` 採用 MIT 許可證。詳情請參閱 `LICENSE` 文件。

## 🤝 貢獻

歡迎任何形式的貢獻！如果您有任何建議、功能請求或錯誤報告，請隨時提出 Issue 或提交 Pull Request。

## 🌟 鳴謝

本專案基於 `jieba` 庫進行優化和增強。感謝原作者及所有貢獻者。
