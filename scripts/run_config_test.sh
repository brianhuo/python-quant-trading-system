#!/bin/bash
# 配置加载器测试运行脚本

echo "=== 配置加载器测试脚本 ==="
echo "当前目录: $(pwd)"
echo "Python版本: $(python3 --version)"
echo

# 激活虚拟环境（如果存在）
if [ -d "../general_env" ]; then
    echo "激活 general_env 虚拟环境..."
    source ../general_env/bin/activate
    echo "虚拟环境已激活: $(which python)"
    echo
fi

# 检查依赖包
echo "检查依赖包..."
python -c "
try:
    import yaml, cryptography, dotenv
    print('✅ 所有依赖包已安装')
except ImportError as e:
    print(f'❌ 缺少依赖包: {e}')
    print('请运行: pip install pyyaml cryptography python-dotenv')
"
echo

# 运行配置加载器测试
echo "运行配置加载器测试..."
python enhanced_config_loader.py
echo

echo "运行使用示例..."
python config_usage_example.py
echo

echo "=== 测试完成 ==="
