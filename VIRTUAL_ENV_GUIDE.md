# 🐍 Python虚拟环境使用指南

## 📋 概述

本项目使用Python虚拟环境来管理依赖，确保iMac和MacBook之间同步时的环境一致性。

## 🚀 快速开始

### 1. 首次设置（新MacBook或重新同步后）

```bash
# 运行自动设置脚本
./setup_venv.sh
```

### 2. 日常使用

#### 激活虚拟环境
```bash
# 方法一：使用快捷脚本
./activate_env.sh

# 方法二：手动激活
source venv/bin/activate
```

#### 验证环境
```bash
# 检查Python路径
which python
# 应该显示: /path/to/project/venv/bin/python

# 测试核心包
python -c "import numpy, pandas, lightgbm; print('✅ 环境正常')"
```

#### 运行项目
```bash
# 在激活的虚拟环境中
python scripts/data_preprocessor_demo.py
python scripts/unified_data_client_demo.py
```

#### 退出环境
```bash
deactivate
```

## 🔧 VS Code集成

### 自动配置
项目已包含`.vscode/settings.json`配置文件，VS Code会自动：
- 识别虚拟环境
- 使用正确的Python解释器
- 激活终端环境

### 手动配置
如果需要手动配置：
1. `Cmd+Shift+P`
2. 搜索："Python: Select Interpreter"
3. 选择：`./venv/bin/python`

## 📦 依赖管理

### 安装新包
```bash
# 激活环境后
pip install package_name

# 更新requirements.txt
pip freeze > requirements.txt
```

### 同步依赖
```bash
# 在新机器上
pip install -r requirements.txt
```

## 🔄 跨设备同步

### 同步设置清单
- ✅ 包含`requirements.txt`
- ✅ 包含`setup_venv.sh`
- ✅ 包含`.vscode/settings.json`
- ❌ **不要**同步`venv/`目录（太大且不必要）

### 同步后操作
1. 确保Python 3已安装
2. 运行：`./setup_venv.sh`
3. 测试：`python scripts/data_preprocessor_demo.py`

## 🐛 常见问题

### Q: 提示"python命令未找到"
**A:** 确保已激活虚拟环境：`source venv/bin/activate`

### Q: VS Code使用错误的Python
**A:** 手动选择解释器：`Cmd+Shift+P` → "Python: Select Interpreter"

### Q: 依赖包导入失败
**A:** 重新安装依赖：`pip install -r requirements.txt`

### Q: 虚拟环境损坏
**A:** 删除并重建：
```bash
rm -rf venv
./setup_venv.sh
```

## 📁 项目结构

```
python-quant-trading-system/
├── venv/                 # 虚拟环境（不同步）
├── scripts/              # Python脚本
├── requirements.txt      # 依赖列表
├── setup_venv.sh        # 环境设置脚本
├── activate_env.sh      # 快速激活脚本
├── .vscode/             # VS Code配置
│   └── settings.json
└── VIRTUAL_ENV_GUIDE.md # 本指南
```

## 💡 最佳实践

1. **始终在虚拟环境中工作**
2. **定期更新requirements.txt**
3. **不要提交venv目录到git**
4. **在每台新机器上重新创建虚拟环境**
5. **保持Python版本一致**

## 🔗 相关文件

- `requirements.txt` - 依赖包列表
- `setup_venv.sh` - 自动设置脚本
- `activate_env.sh` - 快速激活脚本
- `.vscode/settings.json` - VS Code配置

---
更新日期: 2025-09-02
维护者: AI Assistant
