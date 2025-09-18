#!/bin/bash

# DiffSinger UTAU 构建和发布脚本

set -e

echo "🚀 开始构建和发布 DiffSinger UTAU..."

# 检查 Python 版本
python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python 版本: $python_version"

if [[ "$python_version" != "3.8" ]]; then
    echo "❌ 错误: 需要 Python 3.8，当前版本: $python_version"
    echo "请使用 conda 创建 Python 3.8 环境:"
    echo "conda create -n diffsinger python=3.8"
    echo "conda activate diffsinger"
    exit 1
fi

# 清理之前的构建
echo "🧹 清理之前的构建文件..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf __pycache__/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 安装构建依赖
echo "📦 安装构建依赖..."
pip install --upgrade pip
pip install build twine

# 运行测试（如果有的话）
echo "🧪 运行测试..."
if [ -f "test_*.py" ] || [ -d "tests" ]; then
    pip install pytest
    python -m pytest tests/ || echo "⚠️  测试失败，但继续构建..."
else
    echo "ℹ️  没有找到测试文件，跳过测试"
fi

# 构建包
echo "🔨 构建包..."
python -m build

# 检查构建结果
echo "✅ 检查构建结果..."
ls -la dist/

# 验证包
echo "🔍 验证包..."
python -m twine check dist/*

echo "🎉 构建完成！"
echo ""
echo "📋 下一步操作："
echo "1. 测试安装: pip install dist/diffsinger_utau-*.whl"
echo "2. 上传到 PyPI: python -m twine upload dist/*"
echo "3. 或者上传到测试 PyPI: python -m twine upload --repository testpypi dist/*"
echo ""
echo "📁 构建文件位置:"
echo "   - 源码包: dist/diffsinger_utau-*.tar.gz"
echo "   - 轮子包: dist/diffsinger_utau-*.whl"
