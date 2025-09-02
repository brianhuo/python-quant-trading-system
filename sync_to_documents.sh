#!/bin/bash
# sync_to_documents.sh
# 自动同步主开发文件夹到Documents同步文件夹

echo "🔄 开始同步代码到Documents文件夹..."
echo "源文件夹: /Users/richard/python/"
echo "目标文件夹: /Users/richard/Documents/python-quant-trading-system/"
echo ""

# 检查目标文件夹是否存在
if [ ! -d "/Users/richard/Documents/python-quant-trading-system/" ]; then
    echo "📁 创建目标文件夹..."
    mkdir -p /Users/richard/Documents/python-quant-trading-system/
fi

# 同步scripts目录 (排除缓存和日志)
echo "📂 同步scripts目录..."
rsync -av --progress \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='logs/' \
    --exclude='cache/' \
    --exclude='catboost_info/' \
    --exclude='ibkr_env/' \
    /Users/richard/python/scripts/ \
    /Users/richard/Documents/python-quant-trading-system/scripts/

# 同步根目录重要文件
echo "📄 同步根目录文件..."
rsync -av --progress \
    --include='*.sh' \
    --include='*.md' \
    --exclude='general_env/' \
    --exclude='venv/' \
    --exclude='*' \
    /Users/richard/python/ \
    /Users/richard/Documents/python-quant-trading-system/

# 创建同步日志
echo "📝 创建同步记录..."
echo "最后同步时间: $(date)" > /Users/richard/Documents/python-quant-trading-system/LAST_SYNC.txt
echo "源文件夹: /Users/richard/python/" >> /Users/richard/Documents/python-quant-trading-system/LAST_SYNC.txt
echo "同步的Python文件数量: $(find /Users/richard/Documents/python-quant-trading-system/scripts -name "*.py" | wc -l)" >> /Users/richard/Documents/python-quant-trading-system/LAST_SYNC.txt

echo ""
echo "✅ 同步完成！"
echo "📊 统计信息:"
echo "  源文件夹Python文件: $(find /Users/richard/python/scripts -name "*.py" | wc -l)"
echo "  目标文件夹Python文件: $(find /Users/richard/Documents/python-quant-trading-system/scripts -name "*.py" | wc -l)"
echo ""
echo "🎯 提示: 开发工作请继续在 /Users/richard/python/ 进行"
echo "📱 同步文件夹可用于iMac和MacBook之间的数据传输"



