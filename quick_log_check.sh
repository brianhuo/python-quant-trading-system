#!/bin/bash
# 快速日志检查工具 - 从项目根目录运行

echo "=== 快速日志检查 ==="
echo "当前目录: $(pwd)"
echo

# 检查scripts目录
if [ ! -d "scripts" ]; then
    echo "❌ 错误: 请在项目根目录运行"
    exit 1
fi

# 检查日志目录
if [ ! -d "scripts/logs" ]; then
    echo "📝 没有找到日志文件，先运行一些演示："
    echo "   cd scripts && python3 logger_usage_examples.py"
    exit 0
fi

echo "📊 日志文件统计:"
echo "总文件数: $(ls scripts/logs/*.log 2>/dev/null | wc -l)"
echo "总大小: $(du -sh scripts/logs 2>/dev/null | cut -f1)"
echo

echo "📋 最新的5个日志文件:"
ls -lt scripts/logs/*.log 2>/dev/null | head -5

echo
echo "📄 示例日志内容 (demo_basic.log):"
if [ -f "scripts/logs/demo_basic.log" ]; then
    echo "--- 前3行 ---"
    head -n 3 scripts/logs/demo_basic.log
    echo "--- 文件大小: $(wc -l < scripts/logs/demo_basic.log) 行 ---"
else
    echo "文件不存在"
fi

echo
echo "🔍 可用的分析命令:"
echo "  1. 从根目录: ./analyze_logs.sh"
echo "  2. 手动分析: cd scripts && python3 log_analyzer.py"
echo "  3. 重新生成日志: cd scripts && python3 logger_usage_examples.py"




