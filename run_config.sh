#!/bin/bash
# 从项目根目录运行配置加载器的便捷脚本

echo "=== 配置加载器快捷运行脚本 ==="
echo "当前目录: $(pwd)"

# 检查是否在正确的目录
if [ ! -d "scripts" ]; then
    echo "❌ 错误: 未找到 scripts 目录"
    echo "请确保您在 /Users/richard/python 目录中运行此脚本"
    exit 1
fi

# 进入 scripts 目录
cd scripts

echo "✅ 已切换到 scripts 目录: $(pwd)"
echo

# 激活虚拟环境（如果存在）
if [ -d "../general_env" ]; then
    echo "激活虚拟环境..."
    source ../general_env/bin/activate
fi

# 运行配置加载器
echo "运行增强版配置加载器..."
python3 enhanced_config_loader.py
echo

echo "=== 完成 ==="
