#!/bin/bash
# sync_pull.sh
# 从Documents同步文件夹拉取更新到主开发文件夹

echo "🔄 开始从Documents文件夹拉取更新..."
echo "源文件夹: /Users/richard/Documents/python-quant-trading-system/"
echo "目标文件夹: /Users/richard/python/"
echo ""

# 检查源文件夹是否存在
if [ ! -d "/Users/richard/Documents/python-quant-trading-system/" ]; then
    echo "❌ 错误: Documents同步文件夹不存在!"
    echo "请先运行 ./sync_to_documents.sh 创建同步文件夹"
    exit 1
fi

# 显示同步信息
if [ -f "/Users/richard/Documents/python-quant-trading-system/LAST_SYNC.txt" ]; then
    echo "📋 上次同步信息:"
    cat /Users/richard/Documents/python-quant-trading-system/LAST_SYNC.txt
    echo ""
fi

# 确认操作
echo "⚠️  警告: 这将会覆盖本地的修改!"
echo "是否继续? (y/N)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "❌ 操作已取消"
    exit 0
fi

# 备份当前更改 (如果有Git)
if [ -d ".git" ]; then
    echo "💾 检查Git状态..."
    if ! git diff --quiet; then
        echo "📦 发现未提交的更改，创建备份分支..."
        git add -A
        git commit -m "Backup before sync_pull $(date)"
        echo "✅ 已创建备份提交"
    fi
fi

# 从Documents拉取scripts目录
echo "📂 拉取scripts目录..."
rsync -av --progress \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='logs/' \
    --exclude='cache/' \
    --exclude='catboost_info/' \
    --exclude='ibkr_env/' \
    /Users/richard/Documents/python-quant-trading-system/scripts/ \
    /Users/richard/python/scripts/

# 拉取根目录文件
echo "📄 拉取根目录文件..."
rsync -av --progress \
    --include='*.sh' \
    --include='*.md' \
    --exclude='LAST_SYNC.txt' \
    --exclude='*' \
    /Users/richard/Documents/python-quant-trading-system/ \
    /Users/richard/python/

# 创建拉取日志
echo "📝 创建拉取记录..."
echo "最后拉取时间: $(date)" > /Users/richard/python/LAST_PULL.txt
echo "源文件夹: /Users/richard/Documents/python-quant-trading-system/" >> /Users/richard/python/LAST_PULL.txt
echo "拉取的Python文件数量: $(find /Users/richard/python/scripts -name "*.py" | wc -l)" >> /Users/richard/python/LAST_PULL.txt

echo ""
echo "✅ 拉取完成！"
echo "📊 统计信息:"
echo "  本地Python文件: $(find /Users/richard/python/scripts -name "*.py" | wc -l)"
echo "  Documents Python文件: $(find /Users/richard/Documents/python-quant-trading-system/scripts -name "*.py" | wc -l)"
echo ""
echo "🎯 提示: 拉取的文件已覆盖本地版本"
echo "📱 如需查看更改历史，请检查Git日志"

