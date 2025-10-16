# `jieba_fast_dat` 效能優化與功能增強專案知識庫

最後更新時間: 2025-10-16

### E. 進度管理與待辦事項 (Progress & To-Do)

*   **TODO:**
    *   **POS 優化 - 階段 1: C++ HMM 參數管理與 Viterbi 實現**
        *   定義 C++ 數據結構以高效儲存 POS HMM 參數 (start_P, trans_P, emit_P, char_state_tab_P)。
        *   在 C++ 中實現從檔案載入這些 HMM 參數的邏輯。
        *   修改 `jieba_fast_dat/source/jieba_fast_functions_pybind.cpp` 中的 `_viterbi` 函數，使其正確處理 (狀態, 標籤) 對，並使用 C++ 內部 HMM 參數。
        *   將 C++ POS Viterbi 函數暴露給 Python。
    *   **POS 優化 - 階段 2: Python `POSTokenizer` 整合 C++ 實現**
        *   修改 `jieba_fast_dat/posseg/__init__.py` 中的 `POSTokenizer.__cut` 方法，調用新的 C++ POS Viterbi 函數。
    *   研究與評估 CPU 優化的中文依存句法分析器 算法
    *   研究與評估 CPU 優化的中文命名實體識別 (NER) 算法
    *   研究與實現 CPU 優化的中文三元組萃取 算法
    *   **分詞優化 - 階段 1: 檢查 Python 層的文本預處理和後處理是否有優化空間**
        *   評估將頻繁使用的正規表達式操作移至 C++ 的可行性。
    *   **分詞優化 - 階段 2: HMM 模式下未登錄詞處理的優化**
        *   評估將 HMM 模式下未登錄詞處理邏輯遷移到 C++ 的可行性。

*   **DONE:**
    *   **重新建立 README.md 文件**
    *   **處理 pyproject.toml 與 setup.py 同時存在且職權重疊的問題**
    *   **測試修復 - 階段 1: 調整 `test_dict_speed.py` 中的 `test_dictionary_loading_speed`**
    *   **完善建置與測試流程**

### A. 專案核心目標 (Goal)

1.  **專注於中文文本處理:** 專案的核心目標是高效處理中文文本。
2.  **專案基礎**: `jieba_fast_dat` 基於 `jieba_fast` 進行開發，而 `jieba_fast` 則源自於 `jieba` 專案，旨在提供更高效能的中文分詞解決方案。
3.  **提供高效、準確的中文分詞功能:** 作為 `jieba_fast_dat` 的核心，提供快速且精確的中文詞語切分。
3.  **支援詞性標注功能:** 能夠對分詞結果進行詞性標注，提供更豐富的語言分析能力。
4.  **核心效能優化:** 解決 `jieba_fast_dat` 首次啟動時的效能瓶頸，實現近乎瞬時的啟動速度與最低的記憶體佔用。
5.  **功能增強:** 仿效 `cppjieba-py-dat` 的設計，實現了一個動態、持久化且能自動失效的 DAT 快取機制，以優雅地支援使用者自訂詞典。
6.  **長期目標 - 知識圖譜萃取 (KGE):** 最終目標是能夠從中文文本中自動萃取實體、關係和事件，構建知識圖譜。

### B. 環境與執行者 (Context)

*   **執行者:** 所有程式碼與指令由使用者在本地端執行。
*   **production環境:** Docker Swarm (Container 內)。
*   **執行環境約束:** 所有計算必須在 CPU 上執行，不依賴 GPU。
*   **核心技術棧:** Python 3.x, C++, `darts-clone`, `uv` (虛擬環境管理與套件安裝), `pybind11`, `pytest` (測試框架)。
*   **命名規範:**
    *   **安裝名稱 (Package Name):** `jieba_fast_dat`
    *   **導入名稱 (Import Name):** `jieba_fast_dat`

### C. 最新狀態與規範 (Current Spec)

*   **現行版本**: `0.54` (來自 `pyproject.toml`)
*   **關鍵參數總結**:
    *   **預設詞典**: `dict.txt`
    *   **分詞模式**: 精確模式、全模式、搜尋引擎模式。
    *   **HMM**: 可選。
    *   **快取機制**: 基於 MD5 雜湊的持久化 DAT 快取。

### F. 知識與設計規範 (KNOWLEDGE & DESIGN)

#### F.1 核心算法與模型

*   **Double-Array Trie (DAT)**:
    *   **實現**: 核心詞典結構，透過 `darts-clone` 庫在 C++ 中實現 (`jieba_fast_dat/source/jieba_fast_functions_pybind.cpp` 的 `JiebaDict` 類別)。
    *   **功能**: 提供高效的詞語儲存、查詢 (`contains_word`) 和前綴匹配 (`common_prefix_search`)，是分詞詞圖構建的基礎。
    *   **優勢**: 記憶體佔用低，查詢速度極快，適用於大型詞典。
*   **Viterbi 算法**:
    *   **分詞**: 在 C++ 層實現 (`_calc`, `_get_DAG_and_calc` 函數)，用於在詞圖 (DAG) 上尋找最佳詞語路徑，實現精確分詞。
    *   **詞性標注**: 在 C++ 層實現 (`_viterbi` 函數)，基於隱馬爾可夫模型 (HMM) 進行詞性標注，程式碼中暗示使用 BEMS (Begin, Middle, End, Single) 狀態。
*   **TF-IDF (Term Frequency-Inverse Document Frequency)**:
    *   **實現**: Python 層 (`jieba_fast_dat/analyse/tfidf.py` 的 `TFIDF` 類別)。
    *   **功能**: 根據詞語在文檔中的頻率和在語料庫中的稀有程度，評估詞語的重要性，用於關鍵詞提取。
*   **TextRank**:
    *   **實現**: Python 層 (`jieba_fast_dat/analyse/textrank.py` 的 `TextRank` 類別)。
    *   **功能**: 一種基於圖的排序算法，透過詞語共現關係構建圖，提取關鍵詞。

#### F.2 核心功能實現

*   **中文分詞 (Chinese Word Segmentation)**:
    *   **入口**: `jieba_fast_dat/__init__.py` 中的 `Tokenizer.cut` 方法。
    *   **模式**: 支援精確模式、全模式和搜尋引擎模式。
    *   **HMM**: 可選，用於提高分詞準確性，尤其是在處理未登錄詞時。
    *   **並行處理**: 支援使用 `multiprocessing.Pool` 進行並行分詞，提高處理大量文本的效率。
*   **詞性標注 (Part-of-Speech Tagging)**:
    *   **實現**: 透過 `jieba_fast_dat/posseg` 模組，並在 `jieba_fast_dat/__init__.py` 中整合，底層由 C++ 加速的 Viterbi 算法驅動。
*   **關鍵詞提取 (Keyword Extraction)**:
    *   提供基於 TF-IDF 和 TextRank 兩種算法的關鍵詞提取功能。
*   **使用者詞典管理**:
    *   `load_userdict`: 支援載入使用者自訂詞典，會觸發主詞典的重新初始化以適應 DAT 的不可變特性。
    *   `add_word`, `del_word`, `suggest_freq`: 目前在 Python 層為空操作 (no-op)，反映 DAT 不支援執行時動態詞語增刪的特性。
*   **持久化快取 (Persistent Caching)**:
    *   **機制**: 在 `Tokenizer.initialize` 中實現，使用詞典內容的 MD5 雜湊生成唯一的快取檔案路徑。
    *   **流程**: 優先從 `.trie` 和 `.freq` 快取檔案載入 DAT 詞典。若快取不存在或失效，則從原始詞典檔案載入，並在成功後將 DAT 序列化並儲存到快取中。
    *   **效益**: 顯著加速後續詞典載入，降低啟動延遲，並透過 MD5 確保快取一致性。

#### F.3 技術棧與最佳實踐

*   **C++ 與 `pybind11` 綁定**:
    *   **目的**: 將 C++ 實現的高效能算法 (DAT 構建、Viterbi 算法) 無縫暴露給 Python，結合兩者優勢。
    *   **遷移**: 已從 SWIG 遷移至 `pybind11`，提供更現代、Pythonic 的接口。
*   **Memory-Mapped Files (mmap)**:
    *   **應用**: C++ 層的 `Darts::DoubleArray::save` 和 `Darts::DoubleArray::open` 方法可能利用 `mmap` 或類似機制，實現高效的 DAT 檔案 I/O，確保快速載入和低記憶體消耗。
*   **MD5 雜湊**:
    *   **應用**: `get_cache_file_path` 函數使用詞典檔案內容的 MD5 雜湊值來生成唯一的快取檔案名，確保快取失效機制。
*   **多執行緒安全**:
    *   `jieba_fast_dat/__init__.py` 中的 `Tokenizer` 類別使用 `threading.RLock` 保護詞典初始化過程，確保多執行緒環境下的穩定性。
*   **正規表達式 (Regular Expressions)**:
    *   廣泛用於文本預處理和分塊，例如在 `Tokenizer.cut` 中分割不同類型的文本區塊。
*   **CPU 優先原則**: 所有算法和庫的選擇都必須符合 CPU 執行效率，避免引入對 GPU 的依賴，以確保專案在 CPU 環境下的高效運行。

#### F.4 統一測試流程與知識 (Unified Testing Protocol)

*   **開發依賴管理:** 所有用於測試、建置和開發的工具 (如 `pytest`, `build`, `psutil`, `whoosh`) 統一在 `pyproject.toml` 的 `[project.optional-dependencies].dev` 中管理。
*   **統一測試框架 - pytest:** 專案統一使用 `pytest` 作為測試框架，以利用其豐富功能、易用斷言和插件生態系統。
*   **測試執行流程:**
    *   **前置作業:** 參考 `F.6` 確保開發環境與依賴已正確安裝。
    *   **執行指令:** 使用 `uv run pytest` 執行所有測試。使用 `-s` 參數可查看 `print` 輸出。
*   **測試覆蓋率:** 擴展測試案例以覆蓋更多輸入類型（如混合語言和數字）是確保 NLP 工具魯棒性的關鍵。
*   **效能測試:** 透過精確的計時測試（如 `test/test_dict_speed.py`）來驗證和基準化效能優化，是確保專案目標達成的重要手段。
*   **處理檔案路徑:** 在測試腳本中應避免使用相對路徑。使用 `os.path` 動態建構絕對路徑可以提高測試在不同執行環境下的魯棒性。
*   **腳本轉測試:** 獨立運行的腳本若要納入 `pytest`，需將其邏輯包裝在 `test_` 函數中，並移除對 `sys.argv` 等外部輸入的依賴。
*   **避免副作用:** 測試文件中的頂層可執行代碼（如 `sys.exit()`, `sys.stdin` 讀取）或 `setup()` 調用，都應包裹在 `if __name__ == '__main__':` 塊中，以防止在 `pytest` 導入時意外執行。

#### F.5 快取機制實作與驗證

*   **Python 快取邏輯整合**:
    *   在 `jieba_fast_dat/__init__.py` 中實作 `get_cache_file_path` 函數，利用詞典內容的 MD5 雜湊生成唯一的快取檔案路徑，並負責快取目錄的建立。
    *   修改 `Tokenizer.initialize` 方法，實現 DAT 詞典的快取載入與儲存機制。優先從快取載入，若快取不存在或失效，則從原始詞典檔案載入並儲存至快取。
    *   引入 `hashlib` 模組以支援 MD5 雜湊計算。
    *   在關鍵流程點（詞典載入、快取操作）增加偵錯日誌，記錄詞彙量與時間，以便追蹤與效能分析。
*   **C++ 擴展與套件更新**:
    *   確保 C++ 擴展在 Python 更改後重新編譯並重新安裝套件 (`uv pip install . --force-reinstall`)，以應用最新的功能。
*   **快取功能測試**:
    *   建立 `test/test_mmap_cache.py` 測試檔案，涵蓋快取檔案建立、載入速度、失效機制及分詞正確性等方面的驗證。
    *   透過 `pytest -s` 執行測試，確保快取機制的穩定性與正確性。

#### F.6 開發與建置流程 (Development & Build Workflow)

*   **環境初始化 (Environment Setup):**
    1.  **建立虛擬環境:** 使用 `uv venv` 建立獨立的 Python 虛擬環境。
    2.  **啟動虛擬環境:** `source .venv/bin/activate`
    3.  **安裝開發依賴:** 使用 `uv pip install -e .[dev]` 安裝所有開發與測試所需的套件。`-e` 參數以可編輯模式安裝，方便開發。

*   **統一測試流程 (Testing Workflow):**
    *   **執行指令:** 在虛擬環境中，使用 `uv run pytest` 執行完整的測試套件。

*   **套件建置流程 (Build Workflow):**
    *   **執行指令:** 使用 `python -m build` 來建置原始碼發行版 (sdist) 和 Wheel 發行版 (wheel)。產物會存放在 `dist/` 目錄下。

*   **套件打包設定 (Packaging Configuration):**
    *   **自動探索套件:** `pyproject.toml` 已設定 `[tool.setuptools.packages.find]`，可自動探索所有 `jieba_fast_dat` 下的子套件，無需手動增列。`test` 目錄已被排除。
    *   **包含額外檔案:** `MANIFEST.in` 檔案負責將 `README.md`, `LICENSE` 等非程式碼檔案打包至最終的發行版中。

### H. 主要技術債與風險 (Tech Debt & Risks)

*   **潛在的 `mmap` 實現細節**: 雖然 `Darts::DoubleArray::save` 和 `open` 可能利用 `mmap`，但具體實現細節仍需進一步確認，以確保最佳效能和跨平台兼容性。
*   **使用者詞典動態增刪的限制**: DAT 結構的不可變特性導致 `add_word` 和 `del_word` 成為空操作，這可能限制了某些需要頻繁動態更新詞典的應用場景。
*   **首次載入大型詞典的初始化時間**: 儘管快取機制旨在緩解此問題，但首次載入大型詞典時仍存在初始化時間瓶頸。
*   **Python 2 兼容性移除的影響**: 雖然決策是專注於 Python 3，但對於仍在使用 Python 2 的潛在用戶，這可能是一個遷移成本。
*   **Windows 平台支援的移除**: 同樣，這可能限制了在 Windows 環境下使用的便利性。

### D. 協作日誌與歷史決策總結 (Key Decisions & Logs)

*   **問題:** 專案同時支援 Python 2 和 Windows，增加了維護複雜性。
*   **決策:** 移除對 Python 2 和 Windows 的支援，專注於 Python 3 (Linux/macOS)。
*   **問題:** 套件名稱和導入名稱與專案目錄結構不一致。
*   **決策:** 將套件名稱和導入名稱統一改為 `jieba_fast_dat`。
*   **問題:** 需要一個高效的工具來管理 Python 套件和虛擬環境。
*   **決策:** 採用 `uv` 作為標準工具。
*   **問題:** 專案使用 SWIG 進行 C++ 綁定，但 `pybind11` 更現代。
*   **決策:** 將 C++ 綁定技術從 SWIG 遷移到 `pybind11`。
*   **問題:** 專案的測試分散且沒有統一的測試框架。
*   **決策:** 統一使用 `pytest` 作為專案的測試框架。
*   **問題:** `pytest` 執行時，多個測試因檔案路徑、命令列參數依賴、非測試檔案收集等問題導致錯誤。
*   **決策:** 全面重構測試並配置 `pyproject.toml`，以符合 `pytest` 規範。
*   **問題:** C++ 擴展遷移到 `pybind11` 後，出現多個 `AttributeError`。
*   **決策:** 修正 C++ 擴展的函數公開、調整 Python 層的 API 調用，並更新測試。
*   **問題:** `load_userdict` 等方法拋出 `NotImplementedError`。
*   **決策:** 重新實現 `load_userdict` 以觸發字典重新初始化，並將不支援的操作改為無操作。
*   **問題:** 需要驗證字典初始化和快取載入的速度。
*   **決策:** 新增 `test/test_dict_speed.py` 效能測試腳本。
*   **決策:** 成功執行 `uv pip install . --force-reinstall` 和 `uv run pytest`，所有測試通過。
*   **問題:** `pyproject.toml` 與 `setup.py` 同時存在且職權重疊，導致建置失敗。用戶希望以 `pyproject.toml` 為主要配置。
*   **決策:** 刪除 `setup.py`，將所有專案元數據和 `pybind11` 擴展配置移至 `pyproject.toml`。解決 `long_description` 和 `classifiers` 的配置錯誤。最終，由於 `setuptools` 對 `ext_modules` 在 `pyproject.toml` 中的限制，重新引入一個極簡的 `setup.py` 僅用於定義 `ext_modules`，而 `pyproject.toml` 負責所有其他元數據。成功解決衝突並通過所有測試。
*   **問題:** 測試與建置流程繁瑣，依賴未被統一管理。
*   **決策:** 在 `pyproject.toml` 中建立 `[project.optional-dependencies].dev` 群組，統一管理 `pytest`, `build`, `psutil`, `whoosh` 等開發依賴。並修正 `MANIFEST.in` 與 `setuptools` 的套件探索設定，確保建置流程乾淨無警告。
