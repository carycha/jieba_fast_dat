# uv 測試流程文件

本文件記錄了使用 `uv` 工具進行專案測試的標準流程，包括虛擬環境的建立、依賴安裝、C 擴展編譯和測試執行。

## 流程步驟

1.  **清理現有虛擬環境 (可選，但建議確保環境乾淨)**
    如果存在舊的 `.venv` 目錄，請先將其刪除，以確保建立一個全新的虛擬環境。
    ```bash
    rm -rf .venv
    ```

2.  **建立新的 `uv` 虛擬環境**
    使用 `uv venv` 命令在專案根目錄下建立一個新的虛擬環境。這將在 `.venv` 目錄中建立環境。
    ```bash
    uv venv
    ```

3.  **啟用虛擬環境**
    啟用新建立的虛擬環境。所有後續的 Python 命令都將在此環境中執行。
    ```bash
    source .venv/bin/activate
    ```

4.  **使用 `uv` 安裝專案依賴**
    從 `requirements.txt` 文件中安裝所有專案依賴。`uv pip install` 會高效地處理此過程。
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **使用 `uv` 安裝專案本身並編譯 C 擴展**
    安裝當前專案。由於 `setup.py` 配置了 C 擴展，此步驟將觸發 C/C++ 程式碼的編譯和綁定。
    ```bash
    uv pip install .
    ```

6.  **使用 `uv run pytest` 執行測試**
    在 `uv` 虛擬環境中執行 `pytest`，以驗證專案的功能和 C 擴展的正確性。
    ```bash
    uv run pytest
    ```

## 完整命令範例 (可一次性執行)

```bash
rm -rf .venv && \
uv venv && \
source .venv/bin/activate && \
uv pip install -r requirements.txt && \
uv pip install . && \
uv run pytest
```

**注意事項：**
*   `source .venv/bin/activate` 命令會改變當前 shell 的環境變數，因此在單行命令中，需要使用 `&&` 連接，並確保 `source` 命令在 `uv run` 之前執行。
*   如果 `requirements.txt` 中沒有 `pytest`，請確保在步驟 4 或 5 之前安裝 `pytest`，例如 `uv pip install pytest`。
*   如果遇到 `ModuleNotFoundError`，請檢查測試文件中的導入語句是否正確指向 `jieba_fast_dat` 而非 `jieba`。

## 進階偵錯流程：隔離建置與測試

當標準的建置與測試流程反覆失敗，特別是出現 `AttributeError`，暗示 C++ 擴展的綁定未成功時，可使用此流程來隔離問題。

1.  **建立隔離的 C++ 原始碼檔案**
    建立一個新的 `.cpp` 檔案（例如 `jieba_fast_dat/source/isolate.cpp`），其中只包含有問題的類別或函數的程式碼，以及一個用於測試的、名稱不同的 `PYBIND11_MODULE` 定義。

2.  **建立獨立的建置腳本**
    建立一個新的一次性 Python 腳本（例如 `setup_isolate.py`），其內容只包含建置上述隔離 C++ 檔案所需的 `setuptools.Extension` 和 `setup()` 呼叫。

    ```python
    # 範例：setup_isolate.py
    from setuptools import setup, Extension
    import pybind11

    isolate_ext = Extension(
        'jieba_fast_dat._isolate', # 注意模組名稱
        sources=['jieba_fast_dat/source/isolate.cpp'],
        include_dirs=[pybind11.get_include(), 'jieba_fast_dat/source/'],
        language='c++',
        extra_compile_args=['-std=c++11'],
    )

    setup(
        name='isolate_test',
        version='0.0.1',
        ext_modules=[isolate_ext],
    )
    ```

3.  **建立獨立的測試檔案**
    建立一個新的 Python 測試檔案（例如 `test/final_iso_test.py`），它只導入並測試這個新建立的隔離模組。

4.  **手動建置擴展**
    在已啟用 `uv` 虛擬環境的 shell 中，手動執行獨立的建置腳本。`build_ext --inplace` 會在當前目錄結構中產生 `.so` 檔案，以便 Python 可以直接導入。
    ```bash
    python3 setup_isolate.py build_ext --inplace
    ```

5.  **執行隔離測試**
    使用 `pytest` 執行新的測試檔案。
    ```bash
    pytest test/final_iso_test.py
    ```
    如果這個測試成功，問題就出在原始程式碼的交互作用上。如果失敗，問題就出在被隔離的程式碼本身。