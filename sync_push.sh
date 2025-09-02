#!/bin/bash
# Git推送同步脚本

echo "🔄 开始Git同步推送..."

# 检查是否有更改
if [[ `git status --porcelain` ]]; then
    echo "📝 发现更改，准备提交..."
    
    # 显示状态
    git status
    
    # 添加所有更改（排除.gitignore中的文件）
    git add .
    
    # 提示输入提交信息
    echo "💬 请输入提交信息（按Enter使用默认信息）:"
    read commit_msg
    
    # 使用默认信息如果没有输入
    if [ -z "$commit_msg" ]; then
        commit_msg="Update: $(date +'%Y-%m-%d %H:%M:%S')"
    fi
    
    # 提交更改
    git commit -m "$commit_msg"
    
    # 推送到远程
    echo "🚀 推送到GitHub..."
    git push origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ 同步推送成功！"
        echo "📱 现在可以安全地切换到其他设备了"
    else
        echo "❌ 推送失败，请检查网络连接或权限"
    fi
else
    echo "✨ 没有发现更改，无需推送"
fi

