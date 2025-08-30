#!/bin/bash
# 日志分析便捷脚本 - 从项目根目录运行

echo "=== 交易系统日志分析工具 ==="
echo "当前目录: $(pwd)"

# 检查是否在正确的目录
if [ ! -d "scripts" ]; then
    echo "❌ 错误: 未找到 scripts 目录"
    echo "请确保您在 /Users/richard/python 目录中运行此脚本"
    exit 1
fi

# 检查logs目录
if [ ! -d "scripts/logs" ]; then
    echo "⚠️  警告: 未找到 logs 目录，将分析所有可用日志"
fi

echo "✅ 环境检查通过"
echo

# 切换到scripts目录并运行分析
cd scripts

echo "🔍 开始分析日志文件..."
python3 log_analyzer.py

echo
echo "📊 分析完成！"
echo

# 显示生成的文件
if [ -f "log_analysis_report.html" ]; then
    echo "📋 分析报告已生成:"
    echo "  - log_analysis_report.html"
    echo
    echo "💡 打开报告查看详细分析:"
    echo "  open scripts/log_analysis_report.html"
fi

if [ -d "log_charts" ]; then
    echo "📈 图表已生成:"
    ls -la log_charts/
    echo
    echo "💡 查看图表:"
    echo "  open scripts/log_charts/"
fi

echo "✨ 分析工具运行完成！"
