# `jieba_fast_dat` 效能優化與功能增強專案知識庫

最後更新時間: 2025-10-14

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

*   **Double-Array Trie (DAT) 優勢:**
    *   **記憶體效率:** 相較於傳統 Trie，DAT 使用雙陣列結構 (BASE 和 CHECK) 大幅降低記憶體佔用，尤其適用於大型詞典，可將數百 MB 的詞典壓縮至數 MB。
    *   **緊湊性:** DAT 結構緊湊，適合記憶體受限的環境。
*   **Memory-Mapped Files (mmap) 優勢:**
    *   **快速載入:** 允許將磁碟檔案直接映射到記憶體，實現預建 DAT 快取檔案的近乎瞬時初始化，因為資料是按需載入而非一次性載入。
    *   **降低 RAM 使用:** 將檔案直接映射到虛擬位址空間，減少實際物理 RAM 消耗。
        *   **多行程共享:** 支援多個行程共享相同的詞典資料，進一步優化多行程環境下的記憶體使用。
    
    *   **DAT 與 mmap 結合效益:** `cppjieba-py-dat` 透過結合 DAT 的記憶體效率和 `mmap` 的快速載入及記憶體共享能力，提供了高效能、低記憶體的中文分詞解決方案。
    *   **DAT 快取機制:** 為了實現快速載入，本庫會在首次運行時根據當前詞典（主詞典 + 用戶詞典）內容生成一個 `.dat` 快取文件。快取文件默認存放在用戶快取目錄下（例如 Linux 的 `~/.cache/cppjieba_py_dat/`）。可以通過 Jieba 構造函數的 `dat_cache_dir` 參數指定位置。文件名包含詞典內容的 MD5 值，當詞典文件發生變化時，程序會自動檢測並重新生成快取。生成快取可能需要幾秒鐘時間。
    *   **限制:** 採用 DAT 快取機制通常不支援執行時動態詞語新增；詞典更新通常需要重建 DAT 快取。
*   **依賴管理與虛擬環境:** 專案已採用 `uv` 進行虛擬環境管理與套件安裝，並透過 `requirements.txt` 集中管理依賴。`requirements.txt` 中包含了 `setuptools`, `wheel`, `swig` (暫時保留，待後續遷移至 `pybind11` 後移除) 及 `pytest`。開發環境指南 (`README.md`) 已更新以反映 `uv` 的使用方式和最新的環境要求。
*   **C++ 綁定技術:** 目前專案使用 SWIG 進行 C++ 與 Python 的綁定，透過生成 `_wrap.c` 檔案來實現 C++ 程式碼的 Python 接口。
*   **測試流程:** 已建立自動化測試檔案 (`test/test_segmentation_pos.py`)，用於驗證切詞與詞性標注功能。該測試可透過 `uv run pytest -s test/test_segmentation_pos.py` 指令執行，並透過 `-s` 參數顯示詳細輸出，以便肉眼確認切詞與詞性標注結果的正確性，確保每次修改後功能正常。**為確保測試始終反映最新的 C++ 擴展模組修改，每次執行測試前，會強制執行 `uv pip install . --force-reinstall` 命令，以確保所有 C++ 擴展模組都已重新編譯並安裝。**

### D. 協作日誌與決策總結 (Key Decisions & Logs)

*   **問題:** 專案目前支援 Python 2 和 Windows，增加了維護複雜性並限制了對現代 Python 特性和工具的利用。
*   **決策:** 移除對 Python 2 和 Windows 的支援，專注於 Python 3 (Linux/macOS) 環境，以簡化開發、提高效率並利用最新技術。
*   **知識/教訓:** 移除舊版支援有助於專案的現代化和長期可維護性。
*   **問題:** 專案的套件名稱和導入名稱與實際的專案目錄結構 `jieba_fast_dat` 不一致，可能導致混淆。
*   **決策:** 將套件名稱和導入名稱統一改為 `jieba_fast_dat`，以提高專案的一致性和可維護性。
*   **知識/教訓:** 保持專案命名的一致性對於長期維護和新開發者的上手至關重要。
*   **問題:** 需要一個高效且現代化的工具來管理 Python 套件安裝和虛擬環境。
*   **決策:** 採用 `uv` 作為套件管理和虛擬環境建立工具，以其高性能和簡潔性提升開發體驗。
*   **知識/教訓:** `uv` 能夠顯著加速依賴解析和安裝過程，並提供輕量級的虛擬環境管理。
*   **問題:** 專案目前使用 SWIG 進行 C++ 與 Python 的綁定，但 `pybind11` 提供了更現代、更 Pythonic 的綁定方式，且通常具有更好的性能和更低的學習曲線。
*   **決策:** 將 C++ 綁定技術從 SWIG 遷移到 `pybind11`，以提升開發效率、程式碼品質和維護性。
*   **知識/教訓:** `pybind11` 是一個輕量級的僅標頭庫，用於在 C++11 和 Python 之間創建互操作性，它比 SWIG 更容易使用，並生成更慣用的 Python 程式碼。
*   **問題:** 專案的測試分散且可能沒有統一的測試框架，導致測試管理和執行效率低下。
*   **決策:** 統一使用 `pytest` 作為專案的測試框架，並確保 `test/` 目錄下的所有測試都能被 `pytest` 發現和執行。
*   **知識/教訓:** `pytest` 提供了豐富的功能、易於使用的斷言和插件生態系統，有助於提高測試效率和程式碼品質。
*   **問題:** 執行 `test/test_segmentation_pos.py` 測試時，出現 `pkg_resources` 模組已棄用的警告。
*   **決策:** 測試成功通過，並已將 `pkg_resources` 替換為 `importlib.resources`，成功解決棄用警告。
*   **知識/教訓:** 即使測試通過，也應關注並解決警告，特別是關於棄用模組的警告，以避免未來潛在的問題。

*   **問題:** 為了確保測試始終反映最新的 C++ 擴展模組修改，避免因忘記重新編譯而導致的錯誤。
*   **決策:** 每次執行測試前，強制執行 `uv pip install . --force-reinstall` 命令，以確保所有 C++ 擴展模組都已重新編譯並安裝。
*   **知識/教訓:** 雖然這會增加測試時間，但能有效避免因編譯產物過時導致的潛在問題，確保測試結果的準確性。

*   **問題:** 執行測試時，出現多個 `SyntaxWarning: invalid escape sequence` 警告，主要集中在正則表達式中。
*   **決策:** 這些警告表明正則表達式字串中的反斜杠可能被 Python 解釋器錯誤處理，應將其轉換為原始字串 (raw string)。
*   **知識/教訓:** 在 Python 中定義正則表達式時，使用原始字串 (例如 `r'\s'`) 可以避免 Python 解釋器對反斜杠進行二次處理，確保正則表達式引擎接收到正確的模式。

*   **問題:** 執行測試時，出現多個 `SyntaxWarning: invalid escape sequence` 警告，主要集中在正則表達式中。
*   **決策:** 已將所有相關的正則表達式字串轉換為原始字串 (raw string)，成功解決所有 `SyntaxWarning` 警告。
*   **知識/教訓:** 解決 `SyntaxWarning` 有助於提升程式碼品質，避免潛在的運行時錯誤，並確保程式碼在不同 Python 版本間的兼容性。

*   **問題:** 為了確保 `jieba_fast_dat` 的切詞與詞性標注功能能夠正確處理包含中文、英文和數字的混合文本，需要擴展測試覆蓋率。
*   **決策:** 已在 `test/test_segmentation_pos.py` 中增加新的測試案例，包含中文、英文單字和數字，並細化了詞性標注的斷言，以確保這些混合文本能夠被正確處理。
*   **知識/教訓:** 擴展測試案例以覆蓋更多輸入類型（如混合語言和數字）是確保軟體魯棒性和正確性的關鍵步驟，特別是在處理多語言文本的自然語言處理工具中。

### E. 進度管理與待辦事項 (Progress & To-Do)

*   **DONE:**
    *   移除所有 Python 2 相關的程式碼和配置。
    *   移除所有 Windows 相關的程式碼和配置。
    *   更新 `setup.py` 以反映移除 Python 2 和 Windows 支援。
    *   更新 `MANIFEST.in` 以移除任何 Python 2 或 Windows 相關的檔案包含規則。
    *   檢查並修改 `jieba_fast/_compat.py` 中任何 Python 2 相容性程式碼。
    *   修改 `setup.py` 中的套件名稱、`packages` 和 `package_dir` 為 `jieba_fast_dat`。
    *   將專案根目錄下的 `jieba_fast/` 目錄重新命名為 `jieba_fast_dat/`。
    *   創建 `requirements.txt` 並包含 `setuptools`, `wheel`, `swig`。
    *   創建 `uv` 虛擬環境並安裝依賴。
    *   建制uv虛擬環境後，就先測試是否能在虛擬環境中正確運作。
    *   建立測試檔案，測試切詞與詞性標注是否有正確運作(是否有切詞, 是否能標注詞性(不能全部是x)), 以此每次修改後就執行測試一下是否功能壞掉了。
    *   更新開發和測試環境的設置指南，以包含 `uv` 的使用說明。
    *   檢查 `MANIFEST.in` 是否需要更新 `jieba_fast/` 為 `jieba_fast_dat/`。
    *   處理 `pkg_resources` 模組已棄用的警告，並替換為 `importlib.resources`。
    *   處理 `SyntaxWarning: invalid escape sequence` 警告，將正則表達式字串轉換為原始字串。
    *   將專案的套件安裝和虛擬環境管理流程遷移到 `uv`。
    *   更新 `setup.py` 中的 Python 分類器，以明確支援 Python 3.8+。
    *   移除 `jieba_fast_dat/source/jieba_fast_functions_wrap_py2_wrap.c` 檔案。
# `jieba_fast_dat` 效能優化與功能增強專案知識庫

最後更新時間: 2025-10-14

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

*   **Double-Array Trie (DAT) 優勢:**
    *   **記憶體效率:** 相較於傳統 Trie，DAT 使用雙陣列結構 (BASE 和 CHECK) 大幅降低記憶體佔用，尤其適用於大型詞典，可將數百 MB 的詞典壓縮至數 MB。
    *   **緊湊性:** DAT 結構緊湊，適合記憶體受限的環境。
*   **Memory-Mapped Files (mmap) 優勢:**
    *   **快速載入:** 允許將磁碟檔案直接映射到記憶體，實現預建 DAT 快取檔案的近乎瞬時初始化，因為資料是按需載入而非一次性載入。
    *   **降低 RAM 使用:** 將檔案直接映射到虛擬位址空間，減少實際物理 RAM 消耗。
    *   **多行程共享:** 支援多個行程共享相同的詞典資料，進一步優化多行程環境下的記憶體使用。
*   **DAT 與 mmap 結合效益:** `cppjieba-py-dat` 透過結合 DAT 的記憶體效率和 `mmap` 的快速載入及記憶體共享能力，提供了高效能、低記憶體的中文分詞解決方案。
*   **限制:** 採用 DAT 快取機制通常不支援執行時動態詞語新增；詞典更新通常需要重建 DAT 快取。
*   **依賴管理與虛擬環境:** 專案已採用 `uv` 進行虛擬環境管理與套件安裝，並透過 `requirements.txt` 集中管理依賴。`requirements.txt` 中包含了 `setuptools`, `wheel`, `pybind11` 及 `pytest`。開發環境指南 (`README.md`) 已更新以反映 `uv` 的使用方式和最新的環境要求。
*   **C++ 綁定技術:** 已成功從 SWIG 遷移到 `pybind11`，實現了更現代、更 Pythonic 的 C++ 與 Python 綁定。`pybind11` 提供了更簡潔的 API 和更好的類型轉換機制，提升了開發效率和程式碼品質。
*   **測試流程:** 已統一使用 `pytest` 作為專案的測試框架，並確保 `test/` 目錄下的所有測試都能被 `pytest` 發現和正確執行。測試執行指令和相關文件已更新。**為確保測試始終反映最新的 C++ 擴展模組修改，每次執行測試前，會強制執行 `uv pip install . --force-reinstall` 命令，以確保所有 C++ 擴展模組都已重新編譯並安裝。**

### D. 協作日誌與決策總結 (Key Decisions & Logs)

*   **問題:** 專案目前支援 Python 2 和 Windows，增加了維護複雜性並限制了對現代 Python 特性和工具的利用。
*   **決策:** 移除對 Python 2 和 Windows 的支援，專注於 Python 3 (Linux/macOS) 環境，以簡化開發、提高效率並利用最新技術。
*   **知識/教訓:** 移除舊版支援有助於專案的現代化和長期可維護性。
*   **問題:** 專案的套件名稱和導入名稱與實際的專案目錄結構 `jieba_fast_dat` 不一致，可能導致混淆。
*   **決策:** 將套件名稱和導入名稱統一改為 `jieba_fast_dat`，以提高專案的一致性和可維護性。
*   **知識/教訓:** 保持專案命名的一致性對於長期維護和新開發者的上手至關重要。
*   **問題:** 需要一個高效且現代化的工具來管理 Python 套件安裝和虛擬環境。
*   **決策:** 採用 `uv` 作為套件管理和虛擬環境建立工具，以其高性能和簡潔性提升開發體驗。
*   **知識/教訓:** `uv` 能夠顯著加速依賴解析和安裝過程，並提供輕量級的虛擬環境管理。
*   **問題:** 專案目前使用 SWIG 進行 C++ 與 Python 的綁定，但 `pybind11` 提供了更現代、更 Pythonic 的綁定方式，且通常具有更好的性能和更低的學習曲線。
*   **決策:** 將 C++ 綁定技術從 SWIG 遷移到 `pybind11`，以提升開發效率、程式碼品質和維護性。
*   **知識/教訓:** `pybind11` 是一個輕量級的僅標頭庫，用於在 C++11 和 Python 之間創建互操作性，它比 SWIG 更容易使用，並生成更慣用的 Python 程式碼。
*   **問題:** 專案的測試分散且可能沒有統一的測試框架，導致測試管理和執行效率低下。
*   **決策:** 統一使用 `pytest` 作為專案的測試框架，並確保 `test/` 目錄下的所有測試都能被 `pytest` 發現和執行。
*   **知識/教訓:** `pytest` 提供了豐富的功能、易於使用的斷言和插件生態系統，有助於提高測試效率和程式碼品質。
*   **問題:** 執行 `test/test_segmentation_pos.py` 測試時，出現 `pkg_resources` 模組已棄用的警告。
*   **決策:** 測試成功通過，並已將 `pkg_resources` 替換為 `importlib.resources`，成功解決棄用警告。
*   **知識/教訓:** 即使測試通過，也應關注並解決警告，特別是關於棄用模組的警告，以避免未來潛在的問題。

*   **問題:** 為了確保測試始終反映最新的 C++ 擴展模組修改，避免因忘記重新編譯而導致的錯誤。
*   **決策:** 每次執行測試前，強制執行 `uv pip install . --force-reinstall` 命令，以確保所有 C++ 擴展模組都已重新編譯並安裝。
*   **知識/教訓:** 雖然這會增加測試時間，但能有效避免因編譯產物過時導致的潛在問題，確保測試結果的準確性。

*   **問題:** 執行測試時，出現多個 `SyntaxWarning: invalid escape sequence` 警告，主要集中在正則表達式中。
*   **決策:** 這些警告表明正則表達式字串中的反斜杠可能被 Python 解釋器錯誤處理，應將其轉換為原始字串 (raw string)。
*   **知識/教訓:** 在 Python 中定義正則表達式時，使用原始字串 (例如 `r'\s'`) 可以避免 Python 解釋器對反斜杠進行二次處理，確保正則表達式引擎接收到正確的模式。

*   **問題:** 執行測試時，出現多個 `SyntaxWarning: invalid escape sequence` 警告，主要集中在正則表達式中。
*   **決策:** 已將所有相關的正則表達式字串轉換為原始字串 (raw string)，成功解決所有 `SyntaxWarning` 警告。
*   **知識/教訓:** 解決 `SyntaxWarning` 有助於提升程式碼品質，避免潛在的運行時錯誤，並確保程式碼在不同 Python 版本間的兼容性。

*   **問題:** 為了確保 `jieba_fast_dat` 的切詞與詞性標注功能能夠正確處理包含中文、英文和數字的混合文本，需要擴展測試覆蓋率。
*   **決策:** 已在 `test/test_segmentation_pos.py` 中增加新的測試案例，包含中文、英文單字和數字，並細化了詞性標注的斷言，以確保這些混合文本能夠被正確處理。
*   **知識/教訓:** 擴展測試案例以覆蓋更多輸入類型（如混合語言和數字）是確保軟體魯棒性和正確性的關鍵步驟，特別是在處理多語言文本的自然語言處理工具中。

*   **問題:** 執行 `pytest` 時，`test/test_lock.py` 由於字典檔案路徑問題導致 `FileNotFoundError`。
*   **決策:** 修改 `test/test_lock.py`，使用 `os.path` 動態建構字典檔案的絕對路徑，避免硬編碼。
*   **知識/教訓:** 在測試腳本中處理檔案路徑時，應避免使用相對路徑，尤其是在 `pytest` 等測試框架中，因為執行環境的工作目錄可能不固定。使用 `os.path` 動態建構絕對路徑能提高測試的魯棒性。

*   **問題:** 執行 `pytest` 時，`test/test_file.py`, `test/test_pos_file.py`, `test/test_whoosh_file.py` 由於嘗試存取 `sys.argv[1]` 而導致 `IndexError`。
*   **決策:** 將這些腳本重構為符合 `pytest` 規範的測試函數，並為其提供預設的測試資料檔案 (`test/test.txt`)，移除對命令列參數的依賴。
*   **知識/教訓:** 獨立運行的腳本若要納入 `pytest` 測試框架，需要將其邏輯包裝在 `test_` 開頭的函數中，並處理其對外部輸入（如命令列參數）的依賴，改為使用測試內部可控的資料。

*   **問題:** `pytest` 嘗試收集非 Python 測試檔案 `test/test.txt`，導致 `UnicodeDecodeError`。
*   **決策:** 在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 中使用 `addopts = "--ignore test/test.txt"` 選項，明確指示 `pytest` 忽略此檔案。
*   **知識/教訓:** 當 `pytest` 的預設收集規則（如 `python_files` 或 `norecursedirs`）無法有效排除特定檔案時，可以使用 `addopts` 選項結合 `--ignore` 參數進行精確排除。

*   **問題:** 執行 `pytest` 時，出現 `import file mismatch` 錯誤，因為 `test/` 和 `test/parallel/` 目錄下存在同名測試檔案。
*   **決策:** 在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 中使用 `norecursedirs = ["test/parallel", "tmp"]` 選項，明確指示 `pytest` 忽略 `test/parallel` 目錄。
*   **知識/教訓:** 避免在不同目錄中存在同名但內容不同的測試檔案，這會導致 `pytest` 的模組導入衝突。使用 `norecursedirs` 可以有效管理測試收集範圍。

### E. 進度管理與待辦事項 (Progress & To-Do)

*   **DONE:**
    *   移除所有 Python 2 相關的程式碼和配置。
    *   移除所有 Windows 相關的程式碼和配置。
    *   更新 `setup.py` 以反映移除 Python 2 和 Windows 支援。
    *   更新 `MANIFEST.in` 以移除任何 Python 2 或 Windows 相關的檔案包含規則。
    *   檢查並修改 `jieba_fast/_compat.py` 中任何 Python 2 相容性程式碼。
    *   修改 `setup.py` 中的套件名稱、`packages` 和 `package_dir` 為 `jieba_fast_dat`。
    *   將專案根目錄下的 `jieba_fast/` 目錄重新命名為 `jieba_fast_dat/`。
    *   創建 `requirements.txt` 並包含 `setuptools`, `wheel`, `swig`。
    *   創建 `uv` 虛擬環境並安裝依賴。
    *   建制uv虛擬環境後，就先測試是否能在虛擬環境中正確運作。
    *   建立測試檔案，測試切詞與詞性標注是否有正確運作(是否有切詞, 是否能標注詞性(不能全部是x)), 以此每次修改後就執行測試一下是否功能壞掉了。
    *   更新開發和測試環境的設置指南，以包含 `uv` 的使用說明。
    *   檢查 `MANIFEST.in` 是否需要更新 `jieba_fast/` 為 `jieba_fast_dat/`。
    *   處理 `pkg_resources` 模組已棄用的警告，並替換為 `importlib.resources`。
    *   處理 `SyntaxWarning: invalid escape sequence` 警告，將正則表達式字串轉換為原始字串。
    *   將專案的套件安裝和虛擬環境管理流程遷移到 `uv`。
    *   更新 `setup.py` 中的 Python 分類器，以明確支援 Python 3.8+。
    *   移除 `jieba_fast_dat/source/jieba_fast_functions_wrap_py2_wrap.c` 檔案。
    *   **評估 SWIG 實作的現有範圍和限制，以及遷移到 `pybind11` 或 `Cython` 的潛在效益。**
        *   **決策：** 根據對現有 SWIG 實作的分析，並考量 `spec.md` 中已有的決策，確認將 C++ 綁定技術從 SWIG 遷移到 `pybind11`。
    *   **執行 `pybind11` 遷移：**
        *   重寫 `jieba_fast_dat/source/jieba_fast_functions_wrap_py3.i` 中定義的 C 函數（`_calc`, `_get_DAG`, `_get_DAG_and_calc`, `_viterbi`）以使用 `pybind11` 的 API。
        *   更新 `setup.py` 中的擴展模組配置，以編譯 `pybind11` 擴展。
        *   移除所有 SWIG 相關的檔案（例如 `jieba_fast_dat/source/jieba_fast_functions_wrap_py3.i` 和 `jieba_fast_dat/source/jieba_fast_functions_wrap_py3_wrap.c`）。
    *   更新 `requirements.txt`，將 `swig` 替換為 `pybind11`。
    *   將 `test/` 目錄下的所有測試遷移到 `pytest` 框架。
    *   確保所有測試都能被 `pytest` 發現並正確執行。
    *   更新測試執行指令和相關文件，以反映 `pytest` 的使用。
    *   修正 `test/test_lock.py` 中的 `FileNotFoundError`。
    *   重構 `test/test_file.py`, `test/test_pos_file.py`, `test/test_whoosh_file.py` 以符合 `pytest` 規範。
    *   解決 `test/test.txt` 導致的 `UnicodeDecodeError`。
    *   解決 `pytest` 收集時的 `import file mismatch` 錯誤。
*   **TODO:**
    *   较低内存占用: 使用 Double-Array Trie (DAT) 替换原版 Jieba 的内存 Trie，大幅降低词典加载内存
    *   快速加载: 利用 mmap 加载预先构建的 DAT 缓存文件，实现近乎瞬时的初始化（非首次运行）。
    *   动态词典: 不支持 运行时动态添加用户词 (add_word / InsertUserWord 功能被移除)。如需更新用户词典，需要修改用户词典文件并重新运行程序（会自动检测到变更并重建 DAT 缓存）。