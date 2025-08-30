# 配置加载器快速开始指南

## 问题解决

### Python命令未找到错误
如果遇到 `zsh: command not found: python` 错误，有以下解决方案：

#### 方案1：直接使用 python3（最简单）
```bash
python3 enhanced_config_loader.py
python3 config_usage_example.py
```

#### 方案2：使用别名（已为您设置）
我已经在您的 `~/.zshrc` 中添加了别名：
```bash
alias python=python3
alias pip=pip3
```

重新打开终端或运行 `source ~/.zshrc` 后，可以直接使用：
```bash
python enhanced_config_loader.py
pip install package_name
```

#### 方案3：使用虚拟环境（推荐）
```bash
# 激活虚拟环境
source ../general_env/bin/activate

# 然后可以直接使用 python 命令
python enhanced_config_loader.py
```

## 快速测试

### 从项目根目录运行（推荐）
```bash
# 在 /Users/richard/python 目录下运行
./run_config.sh
```

### 从 scripts 目录运行
```bash
# 先切换到 scripts 目录
cd scripts

# 运行自动测试脚本
./run_config_test.sh

# 或手动测试
python3 enhanced_config_loader.py
python3 config_usage_example.py
```

### 目录导航说明
- **项目根目录**: `/Users/richard/python`
- **脚本目录**: `/Users/richard/python/scripts`
- **配置文件**: 都在 `scripts` 目录中

⚠️ **重要**: 如果您看到 "No such file or directory" 错误，请检查当前目录：
```bash
pwd  # 查看当前目录
ls   # 查看当前目录内容
```

## 常用命令

### 安装依赖
```bash
pip3 install pyyaml cryptography python-dotenv
# 或使用别名后
pip install pyyaml cryptography python-dotenv
```

### 运行不同环境配置
```bash
# 开发环境
python3 -c "from enhanced_config_loader import load_config; config = load_config('development'); print(f'Environment: development, Capital: {config[\"INIT_CAPITAL\"]}')"

# 测试环境
python3 -c "from enhanced_config_loader import load_config; config = load_config('testing'); print(f'Environment: testing, Capital: {config[\"INIT_CAPITAL\"]}')"

# 生产环境
python3 -c "from enhanced_config_loader import load_config; config = load_config('production'); print(f'Environment: production, Capital: {config[\"INIT_CAPITAL\"]}')"
```

## 故障排除

### 检查Python版本
```bash
python3 --version
which python3
```

### 检查pip版本
```bash
pip3 --version
which pip3
```

### 检查虚拟环境
```bash
# 查看可用的虚拟环境
ls -la ../*env*/

# 激活虚拟环境
source ../general_env/bin/activate
# 或
source ../venv/bin/activate
```

### 验证依赖安装
```bash
python3 -c "import yaml, cryptography, dotenv; print('All dependencies installed!')"
```

## 下一步

1. 根据需要修改 `config.development.yaml` 等配置文件
2. 设置 `.env` 文件（复制 `.env.example`）
3. 在您的主程序中集成新的配置加载器

有任何问题请参考 `CONFIG_MIGRATION_GUIDE.md` 或 `CONFIG_OPTIMIZATION_REPORT.md`。
