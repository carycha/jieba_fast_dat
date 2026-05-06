#!/bin/bash
set -e

echo "=== jieba_fast_dat Local Build & Test Tool ==="
echo "注意: 本地編譯的 Wheel 可能會因為 GLIBC 版本過高而導致其他電腦無法安裝。"
echo "建議正式發布請使用 git tag 觸發 GitHub Actions 自動建置。"
echo "=============================================="

# 1. 清理舊的編譯檔
rm -rf dist/ build/ *.egg-info

# 2. 建置原始碼包與 Wheel
uv build

# 3. 根據作業系統處理 Wheel
OS="$(uname)"
if [ "$OS" == "Linux" ]; then
    echo "偵測到 Linux，執行 auditwheel..."
    # 修復 Linux Wheel 並移除未修復的版本
    uv run auditwheel repair dist/*.whl -w dist/ && rm dist/*-linux_x86_64.whl
elif [ "$OS" == "Darwin" ]; then
    echo "偵測到 macOS，執行 delocate..."
    # macOS 版本的修復工具
    if ! command -v delocate-wheel &> /dev/null; then
        echo "找不到 delocate，正在安裝..."
        pip install delocate
    fi
    delocate-wheel -v dist/*.whl
else
    echo "未知的作業系統: $OS"
fi

# 4. 檢查與上傳 (可選)
uv run twine check dist/*

read -p "是否要上傳到 TestPyPI 進行測試? (y/N) " confirm
if [[ $confirm == [yY] ]]; then
    uv run twine upload --repository testpypi dist/*
fi

echo "完成！正式發布建議使用: git tag vX.X && git push --tags"
