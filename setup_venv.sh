#!/bin/bash
# Python虚拟环境设置脚本
# 适用于iMac ↔ MacBook同步的量化交易系统

echo "🐍 Python量化交易系统 - 虚拟环境设置"
echo "========================================"

# 检查Python3是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: Python3未找到，请先安装Python3"
    echo "   建议: brew install python3"
    exit 1
fi

echo "✅ Python版本: $(python3 --version)"

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

echo "📁 项目目录: ${PROJECT_DIR}"

# 创建虚拟环境
if [ ! -d "${VENV_DIR}" ]; then
    echo "🔨 创建Python虚拟环境..."
    python3 -m venv "${VENV_DIR}"
    
    if [ $? -eq 0 ]; then
        echo "✅ 虚拟环境创建成功: ${VENV_DIR}"
    else
        echo "❌ 虚拟环境创建失败"
        exit 1
    fi
else
    echo "📦 虚拟环境已存在: ${VENV_DIR}"
fi

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source "${VENV_DIR}/bin/activate"

# 升级pip
echo "📈 升级pip..."
pip install --upgrade pip

# 安装依赖包
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    echo "📦 安装项目依赖..."
    pip install -r "${PROJECT_DIR}/requirements.txt"
    
    if [ $? -eq 0 ]; then
        echo "✅ 依赖安装成功"
    else
        echo "❌ 依赖安装失败"
        exit 1
    fi
else
    echo "⚠️  未找到requirements.txt，手动安装基础依赖..."
    pip install pyyaml numpy pandas scikit-learn lightgbm xgboost ta numba imbalanced-learn python-dotenv cryptography
fi

# 测试环境
echo "🧪 测试虚拟环境..."
python -c "import numpy, pandas, sklearn, lightgbm, xgboost; print('✅ 核心包导入成功')"

if [ $? -eq 0 ]; then
    echo "🎉 虚拟环境设置完成！"
    echo ""
    echo "💡 使用说明:"
    echo "   激活环境: source venv/bin/activate"
    echo "   退出环境: deactivate"
    echo "   运行测试: python scripts/data_preprocessor_demo.py"
    echo ""
    echo "📝 VS Code用户:"
    echo "   1. Cmd+Shift+P"
    echo "   2. 搜索 'Python: Select Interpreter'"
    echo "   3. 选择: ${VENV_DIR}/bin/python"
else
    echo "❌ 环境测试失败"
    exit 1
fi

# 创建激活脚本的快捷方式
cat > "${PROJECT_DIR}/activate_env.sh" << EOF
#!/bin/bash
# 快速激活虚拟环境
source "${VENV_DIR}/bin/activate"
echo "🐍 已激活Python虚拟环境"
echo "💡 退出请输入: deactivate"
exec "\$SHELL"
EOF

chmod +x "${PROJECT_DIR}/activate_env.sh"
echo "✨ 创建了快捷激活脚本: ./activate_env.sh"
