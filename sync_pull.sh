#!/bin/bash
# Git拉取同步脚本

echo "🔄 开始Git同步拉取..."

# 检查当前分支
current_branch=$(git branch --show-current)
echo "📍 当前分支: $current_branch"

# 拉取最新更改
echo "⬇️ 拉取远程更新..."
git pull origin main

if [ $? -eq 0 ]; then
    echo "✅ 同步拉取成功！"
    echo "📊 当前仓库状态:"
    git status --short
    echo ""
    echo "📈 最近3次提交:"
    git log --oneline -3
    echo ""
    echo "🎯 现在可以安全地开始工作了"
else
    echo "❌ 拉取失败，可能存在冲突需要解决"
    echo "🔧 建议检查:"
    echo "   1. 网络连接是否正常"
    echo "   2. 是否有本地未提交的更改"
    echo "   3. 是否存在合并冲突"
fi

