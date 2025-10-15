# `jieba_fast_dat` 效能優化與功能增強專案知識庫

最後更新時間: 2025-10-15

### A. 專案核心目標 (Goal)

1.  **核心效能優化:** 解決 `jieba_fast_dat` 首次啟動時的效能瓶頸，實現近乎瞬時的啟動速度與最低的記憶體佔用。
2.  **功能增強:** 仿效 `cppjieba-py-dat` 的設計，實現了一個動態、持久化且能自動失效的 DAT 快取機制，以優雅地支援使用者自訂詞典。

### B. 環境與執行者 (Context)

*   **執行者:** 所有程式碼與指令由使用者在本地端執行。
*   **開發環境:** Docker Swarm (Container 內)。
*   **核心技術棧:** Python 3.x, C++, `darts-clone`, `uv` (虛擬環境管理與套件安裝), `pybind11`, `pytest` (測試框架)。
*   **命名規範:**
    *   **安裝名稱 (Package Name):** `jieba_fast_dat`
    *   **導入名稱 (Import Name):** `jieba_fast_dat`

### F. 專案核心知識與最佳實踐 (Core Knowledge & Best Practices)

#### F.1 核心架構與技術選型

*   **Double-Array Trie (DAT) 優勢:** DAT 使用雙陣列結構 (BASE 和 CHECK) 大幅降低記憶體佔用，結構緊湊，尤其適用於大型詞典。
*   **Memory-Mapped Files (mmap) 優勢:** `mmap` 允許將磁碟檔案直接映射到記憶體，實現預建 DAT 快取檔案的近乎瞬時初始化。它透過按需載入來降低物理 RAM 消耗，並支援多行程共享相同的詞典資料。
*   **DAT 與 mmap 結合效益:** `cppjieba-py-dat` 的成功實踐證明，結合 DAT 的記憶體效率和 `mmap` 的快速載入能力，是提供高效能、低記憶體中文分詞解決方案的關鍵。
*   **持久化快取策略:** `cppjieba-py-dat` 的 DAT 快取檔案預設存放在使用者快取目錄下（例如 Linux 的 `~/.cache/cppjieba_py_dat/`），檔案名包含詞典內容的 MD5 值。這套機制是 `jieba_fast_dat` 優化的重要參考。

#### F.2 C++ 擴展與綁定

*   **C++ 綁定技術 - pybind11:** 專案已從 SWIG 遷移到 `pybind11`，以實現更現代、更 Pythonic 的 C++ 與 Python 綁定。`pybind11` 是一個輕量級的僅標頭庫，API 簡潔，能生成更慣用的 Python 程式碼。
*   **函數可見性:** C++ 擴展中的函數必須明確地透過 `pybind11` 的 API 公開，才能在 Python 中被調用。

#### F.3 Jieba_fast_dat 實作細節

*   **DAT 模型的限制:** 採用 DAT 快取機制不支援執行時動態詞語新增 (`add_word`, `del_word`)。這些操作應被視為無操作 (no-op)。
*   **使用者詞典載入:** `load_userdict` 應透過觸發主詞典的重新初始化來載入用戶詞典，以符合 DAT 的不可變特性。
*   **複合詞處理:** 對於應被視為單一詞語的複合詞彙（如 `easy_install`），需要將其明確添加到使用者詞典中，分詞器才能正確識別。
*   **內部 API 變更:** 當底層 API (如 `get_word_frequency`) 發生變化時，所有依賴於舊實現的模塊都需要同步更新。
*   **處理棄用警告:** 應積極處理 `pkg_resources` 等棄用模組的警告，並遷移到如 `importlib.resources` 的現代替代方案，以確保專案的長期穩定性。

#### F.4 測試與品質保證 (Testing & QA)

*   **統一測試框架 - pytest:** 專案統一使用 `pytest` 作為測試框架，以利用其豐富功能、易用斷言和插件生態系統。
*   **測試執行流程:**
    *   **強制重新編譯:** 每次執行測試前，強制執行 `uv pip install . --force-reinstall`，確保測試基於最新的 C++ 擴展模組，避免因編譯產物過時導致的錯誤。
    *   **執行指令:** 使用 `uv run pytest` 執行所有測試。使用 `-s` 參數可查看 `print` 輸出。
*   **測試覆蓋率:** 擴展測試案例以覆蓋更多輸入類型（如混合語言和數字）是確保 NLP 工具魯棒性的關鍵。
*   **效能測試:** 透過精確的計時測試（如 `test/test_dict_speed.py`）來驗證和基準化效能優化，是確保專案目標達成的重要手段。
*   **處理檔案路徑:** 在測試腳本中應避免使用相對路徑。使用 `os.path` 動態建構絕對路徑可以提高測試在不同執行環境下的魯棒性。
*   **腳本轉測試:** 獨立運行的腳本若要納入 `pytest`，需將其邏輯包裝在 `test_` 函數中，並移除對 `sys.argv` 等外部輸入的依賴。
*   **避免副作用:** 測試文件中的頂層可執行代碼（如 `sys.exit()`, `sys.stdin` 讀取）或 `setup()` 調用，都應包裹在 `if __name__ == '__main__':` 塊中，以防止在 `pytest` 導入時意外執行。

#### F.5 專案配置與管理

*   **依賴管理 - uv:** 專案採用 `uv` 進行虛擬環境管理與套件安裝，以利用其高性能和簡潔性。所有依賴由 `requirements.txt` 集中管理。
*   **命名一致性:** 保持套件名稱 (`jieba_fast_dat`)、導入名稱和目錄結構的一致性，對於專案的長期維護至關重要。
*   **精確配置 pytest 收集範圍:**
    *   `testpaths`: 將此選項設置為 `["test"]`，將測試收集限制在指定目錄。
    *   `norecursedirs`: 用於排除 `build`, `.venv`, `jieba_fast_dat` 等非測試目錄，避免模組導入衝突。
    *   `--ignore`: 用於精確排除特定檔案（如 `test/test.txt` 或有未滿足依賴的腳本），避免 `UnicodeDecodeError` 或 `ModuleNotFoundError`。
*   **處理正則表達式警告:** 在 Python 中定義正則表達式時，應使用原始字串 (raw string, e.g., `r'\s'`)，以避免 `SyntaxWarning: invalid escape sequence` 警告和潛在的轉義錯誤。

### D. 協作日誌與決策總結 (Key Decisions & Logs)

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
*   **問題:** `jieba_fast_dat` 的 DAT 實現是記憶體內的，效能有待提升。
*   **決策:** 分析 `cppjieba-py-dat` 的 `mmap` 和持久化快取實現，作為後續優化的藍圖。

### E. 進度管理與待辦事項 (Progress & To-Do)

*   **DONE:**
    *   **專案現代化:** 成功移除 Python 2 和 Windows 支援，並將 C++ 綁定從 SWIG 遷移至 `pybind11`。
    *   **工具鏈統一:** 全面採用 `uv` 進行環境管理，並使用 `pytest` 作為唯一的測試框架。
    *   **測試重構:** 完成了對 `test/` 目錄的全面重構，解決了 `pytest` 的收集與執行問題，並確保所有測試通過。
    *   **功能對齊:** 調整了用戶詞典相關 API (`load_userdict`) 以適應 DAT 模型，並新增了效能測試。

    *   **階段 1: C++ 層 - 實作 DAT 序列化/反序列化**
        *   **步驟 1.1: 在 C++ 中實作 `JiebaDict::save_dat`:**
            *   修改 `jieba_fast_functions_pybind.cpp`，向 `JiebaDict` 添加 `save_dat` 方法。
            *   此方法將接收一個 `cache_path` 字串。
            *   呼叫 `trie.save(cache_path + ".trie")` 儲存 Double-Array Trie。
            *   將詞頻 (`freq_map`) 序列化到 `cache_path + ".freq"`。
            *   返回一個布林值表示成功/失敗。
        *   **步驟 1.2: 在 C++ 中實作 `JiebaDict::open_dat`:**
            *   修改 `jieba_fast_functions_pybind.cpp`，向 `JiebaDict` 添加 `open_dat` 方法。
            *   此方法將接收一個 `cache_path` 字串。
            *   呼叫 `trie.open(cache_path + ".trie")` 載入 Double-Array Trie。
            *   從 `cache_path + ".freq"` 反序列化詞頻 (`freq_map`)。
            *   返回一個布林值表示成功/失敗。
        *   **步驟 1.3: 透過 `pybind11` 將 `save_dat` 和 `open_dat` 公開給 Python。**
            *   修改 `jieba_fast_functions_pybind.cpp`，添加 `save_dat` 和 `open_dat` 的綁定。
        *   **步驟 1.4: 重新編譯 C++ 擴展並重新安裝套件。**
            *   執行 `uv pip install . --force-reinstall` 以應用 C++ 更改。

    *   **階段 2: Python 整合快取邏輯**
        *   **步驟 2.1: 在 Python 中實作 `get_cache_file_path`:**
            *   修改 `jieba_fast_dat/__init__.py`，添加 `get_cache_file_path(dict_path)`。
            *   此函數將根據 `dict_path` 內容的 MD5 雜湊生成唯一的快取檔案路徑。
            *   如果快取目錄不存在，它將建立快取目錄。
        *   **步驟 2.2: 修改 `Tokenizer.initialize` 以進行快取載入/儲存 (並增加偵錯日誌):**
            *   修改 `jieba_fast_dat/__init__.py` 中的 `initialize` 方法。
            *   在呼叫 `_jieba_fast_functions.load_dict` 之前，使用 `get_cache_file_path` 檢查快取是否存在。
            *   如果快取存在，嘗試使用 `_jieba_fast_functions.open_dat` 載入。如果成功，設定 `self.initialized = True` 並返回。
            *   如果快取不存在或載入失敗，則繼續執行 `_jieba_fast_functions.load_dict`。
            *   在成功執行 `_jieba_fast_functions.load_dict` 之後，呼叫 `_jieba_fast_functions.save_dat` 以將新建立的 DAT 儲存到快取。
            *   將 `import hashlib` 添加到 `jieba_fast_dat/__init__.py`。
            *   **增加偵錯日誌:** 在關鍵點（例如：開始載入詞典、嘗試從快取載入、快取載入成功/失敗、建立快取、快取檔案路徑等）添加 `default_logger.debug` 訊息，以追蹤流程和檔案路徑。**特別是，記錄詞彙量和花費時間。**
        *   **步驟 2.3: 重新編譯 C++ 擴展並重新安裝套件。**
            *   執行 `uv pip install . --force-reinstall` 以應用 Python 更改（並確保 C++ 是最新的）。

    *   **階段 3: 測試**
        *   **步驟 3.1: 建立/修改 `test_mmap_cache.py`:**
            *   建立一個新的測試檔案 `test/test_mmap_cache.py` (或修改如果它存在於之前的嘗試中)。
            *   添加測試以：
                *   驗證快取檔案建立。
                *   驗證快取載入速度。
                *   驗證快取失效 (如果詞典內容更改)。
                *   驗證從快取載入後分詞的正確性。
        *   **步驟 3.2: 執行測試。**
            *   執行 `pytest -s` 以驗證實作。
