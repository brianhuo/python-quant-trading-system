#!/bin/bash
# 环境设置脚本 - 用于从iMac同步到MacBook后的初始化

echo "🚀 开始设置Python量化交易系统环境..."

# 检查Python3是否可用
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未找到，请先安装Python3"
    exit 1
fi

echo "✅ Python3版本: $(python3 --version)"

# 安装必要的依赖包
echo "📦 安装依赖包..."
pip3 install --user pyyaml numpy pandas scikit-learn lightgbm xgboost ta numba imbalanced-learn python-dotenv cryptography

# 检查是否有配置文件需要创建
if [ ! -f .config_key ]; then
    echo "🔑 创建配置密钥文件..."
    # 这里可以添加创建配置文件的逻辑
fi

# 运行测试验证环境
echo "🧪 运行环境测试..."
if python3 scripts/data_preprocessor_demo.py > /dev/null 2>&1; then
    echo "✅ 环境设置成功！data_preprocessor_demo.py运行正常"
else
    echo "❌ 环境测试失败，请检查错误信息"
    python3 scripts/data_preprocessor_demo.py
fi

echo "🎉 环境设置完成！"

