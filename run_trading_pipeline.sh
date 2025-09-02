#!/bin/bash
# run_trading_pipeline.sh
# 快速运行交易数据管道的便捷脚本

echo "🚀 启动交易系统数据管道"
echo "=" * 50

# 检查当前目录
current_dir=$(pwd)
echo "当前目录: $current_dir"

# 检查并激活虚拟环境
if [ -d "general_env" ]; then
    echo "激活虚拟环境..."
    source general_env/bin/activate
elif [ -d "../general_env" ]; then
    echo "激活虚拟环境..."
    source ../general_env/bin/activate
fi

# 切换到scripts目录
if [ -d "scripts" ]; then
    cd scripts
    echo "✅ 已切换到scripts目录"
elif [ -f "trading_data_pipeline.py" ]; then
    echo "✅ 已在正确目录"
else
    echo "❌ 错误: 未找到trading_data_pipeline.py"
    exit 1
fi

echo ""
echo "🔄 运行交易数据管道..."
python3 trading_data_pipeline.py

echo ""
echo "✅ 交易数据管道运行完成"
echo ""
echo "💡 提示:"
echo "   - 这是您交易系统的第一部分：数据获取和清洗"
echo "   - 已整合4个核心模块：配置/日志/数据/质量检查"
echo "   - 可直接用于生产环境"
echo "   - 为后续特征工程和策略模块提供清洁数据"




