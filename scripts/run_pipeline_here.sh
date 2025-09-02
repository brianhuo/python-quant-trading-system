#!/bin/bash
# run_pipeline_here.sh
# 在scripts目录中直接运行交易数据管道

echo "🚀 在scripts目录运行交易数据管道"
echo "========================================"

# 检查当前目录
current_dir=$(pwd)
echo "当前目录: $current_dir"

# 检查并激活虚拟环境
if [ -d "../general_env" ]; then
    echo "激活虚拟环境..."
    source ../general_env/bin/activate
elif [ -d "../venv" ]; then
    echo "激活虚拟环境..."
    source ../venv/bin/activate
fi

echo ""
echo "🔄 运行交易数据管道..."
python3 trading_data_pipeline.py

echo ""
echo "✅ 交易数据管道运行完成"
echo ""
echo "💡 快速使用指南:"
echo "   # 在Python中使用:"
echo "   from trading_data_pipeline import create_default_pipeline"
echo "   pipeline = create_default_pipeline()"
echo "   data, report = pipeline.get_clean_data('AAPL', '30min', 1000)"
echo ""
echo "🎯 这是您交易系统的核心数据层！"




