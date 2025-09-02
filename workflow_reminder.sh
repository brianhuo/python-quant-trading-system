#!/bin/bash
# 工作流程提醒脚本

echo "📋 Git多设备同步工作流程"
echo "=================================="
echo ""
echo "🔄 开始工作前 (切换到新设备时)："
echo "   ./sync_pull.sh"
echo "   或手动执行: git pull origin main"
echo ""
echo "✅ 完成工作后 (切换设备前)："
echo "   ./sync_push.sh" 
echo "   或手动执行:"
echo "     git add ."
echo "     git commit -m \"你的提交信息\""
echo "     git push origin main"
echo ""
echo "💡 重要提醒："
echo "   • 每次切换设备前务必推送"
echo "   • 每次开始工作前务必拉取"
echo "   • 提交信息要清晰描述改动"
echo "   • 遇到冲突及时解决"
echo ""
echo "🎯 当前仓库状态："
git status --short

