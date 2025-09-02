#!/bin/bash
# 增强版Git推送同步脚本 - 处理合并冲突和分支分叉

echo "🔄 开始增强版Git同步推送..."

# 检查Git仓库状态
check_git_status() {
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "✅ Git仓库检查通过"
    else
        echo "❌ 当前目录不是Git仓库"
        exit 1
    fi
}

# 处理未完成的合并
handle_ongoing_merge() {
    if [ -f .git/MERGE_HEAD ]; then
        echo "🔧 检测到未完成的合并，正在处理..."
        
        # 检查是否有冲突文件
        if git diff --name-only --diff-filter=U | grep -q .; then
            echo "⚠️ 发现合并冲突，需要手动解决："
            git diff --name-only --diff-filter=U
            echo "请先解决冲突，然后运行 git add . && git commit"
            exit 1
        else
            echo "✅ 没有冲突，完成合并..."
            git commit -m "自动完成合并: $(date +'%Y-%m-%d %H:%M:%S')"
        fi
    fi
}

# 同步远程分支
sync_with_remote() {
    echo "🌐 检查远程更新..."
    
    # 获取远程更新
    git fetch origin main
    
    # 检查本地和远程的差异
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse @{u} 2>/dev/null)
    BASE=$(git merge-base @ @{u} 2>/dev/null)
    
    if [ -z "$REMOTE" ]; then
        echo "⚠️ 无法获取远程分支信息"
    elif [ "$LOCAL" = "$REMOTE" ]; then
        echo "✅ 本地和远程已同步"
    elif [ "$LOCAL" = "$BASE" ]; then
        echo "📥 远程有新提交，正在拉取..."
        git pull origin main --no-edit
    elif [ "$REMOTE" = "$BASE" ]; then
        echo "📤 本地有新提交，准备推送..."
    else
        echo "🔀 分支已分叉，正在合并..."
        git pull origin main --no-edit
        if [ $? -ne 0 ]; then
            echo "❌ 自动合并失败，可能需要手动解决冲突"
            exit 1
        fi
    fi
}

# 智能添加文件
smart_add_files() {
    echo "📝 智能添加文件..."
    
    # 添加所有文件，但排除一些临时文件
    git add .
    
    # 显示将要提交的文件
    echo "📋 将要提交的文件："
    git diff --cached --name-status | head -20
    
    if [ $(git diff --cached --name-status | wc -l) -gt 20 ]; then
        echo "... 还有 $(( $(git diff --cached --name-status | wc -l) - 20 )) 个文件"
    fi
}

# 智能提交
smart_commit() {
    local commit_msg="$1"
    
    if [ -z "$commit_msg" ]; then
        # 生成智能提交信息
        local file_count=$(git diff --cached --name-only | wc -l)
        local modified_files=$(git diff --cached --name-status | grep "^M" | wc -l)
        local new_files=$(git diff --cached --name-status | grep "^A" | wc -l)
        local deleted_files=$(git diff --cached --name-status | grep "^D" | wc -l)
        
        commit_msg="Auto-update: $(date +'%Y-%m-%d %H:%M:%S')"
        if [ $file_count -gt 0 ]; then
            commit_msg="$commit_msg - $file_count files"
            [ $modified_files -gt 0 ] && commit_msg="$commit_msg (${modified_files}M"
            [ $new_files -gt 0 ] && commit_msg="$commit_msg ${new_files}A"
            [ $deleted_files -gt 0 ] && commit_msg="$commit_msg ${deleted_files}D"
            commit_msg="$commit_msg)"
        fi
    fi
    
    echo "💬 提交信息: $commit_msg"
    git commit -m "$commit_msg"
}

# 安全推送
safe_push() {
    echo "🚀 推送到GitHub..."
    
    # 推送前再次检查远程状态
    git fetch origin main
    
    # 尝试推送
    if git push origin main; then
        echo "✅ 推送成功！"
        echo "📊 提交历史："
        git log --oneline -5
        echo "📱 现在可以安全地切换到其他设备了"
        return 0
    else
        echo "❌ 推送失败，尝试解决..."
        
        # 如果推送失败，可能是远程又有新提交
        echo "🔄 重新同步并推送..."
        sync_with_remote
        
        # 再次尝试推送
        if git push origin main; then
            echo "✅ 重新推送成功！"
            return 0
        else
            echo "❌ 推送仍然失败，可能需要手动处理"
            echo "🔧 建议手动执行:"
            echo "   git status"
            echo "   git pull origin main"
            echo "   git push origin main"
            return 1
        fi
    fi
}

# 主函数
main() {
    echo "🎯 增强版Git同步开始..."
    
    # 1. 检查Git状态
    check_git_status
    
    # 2. 处理未完成的合并
    handle_ongoing_merge
    
    # 3. 同步远程分支
    sync_with_remote
    
    # 4. 检查是否有更改需要提交
    if [[ `git status --porcelain` ]]; then
        echo "📝 发现更改，准备提交..."
        
        # 显示详细状态
        git status --short
        
        # 5. 智能添加文件
        smart_add_files
        
        # 6. 获取提交信息
        echo "💬 请输入提交信息（按Enter使用智能生成的信息）:"
        read commit_msg
        
        # 7. 智能提交
        smart_commit "$commit_msg"
        
        # 8. 安全推送
        safe_push
        
    else
        echo "✨ 没有发现更改，检查远程同步状态..."
        sync_with_remote
        echo "🎉 仓库已是最新状态！"
    fi
}

# 执行主函数
main "$@"
