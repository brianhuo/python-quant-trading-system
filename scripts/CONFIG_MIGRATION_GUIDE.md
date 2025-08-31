# 配置加载器迁移指南

## 概述

本指南帮助您从旧版 `config_loader.py` 迁移到增强版 `enhanced_config_loader.py`。

## 主要改进

### 1. 多格式支持
- ✅ **新增**: YAML配置文件支持（推荐）
- ✅ **保留**: JSON配置文件支持
- ✅ **新增**: 多环境配置文件

### 2. 类型安全与验证
- ✅ **新增**: 完整的配置参数验证
- ✅ **新增**: 类型检查和范围验证
- ✅ **新增**: 友好的错误信息

### 3. 安全性增强
- ✅ **新增**: API密钥格式验证
- ✅ **新增**: 敏感数据加密支持
- ✅ **新增**: 安全的环境变量管理

### 4. 多环境支持
- ✅ **新增**: 开发/测试/生产环境配置
- ✅ **新增**: 环境特定参数优化
- ✅ **新增**: 灵活的配置覆盖机制

## 迁移步骤

### 第1步: 安装依赖
```bash
pip install -r requirements_config.txt
```

### 第2步: 备份现有配置
```bash
cp config.json config.json.backup
cp config_loader.py config_loader.py.backup
```

### 第3步: 转换配置文件

#### 选项A: 使用YAML格式（推荐）
1. 将现有的 `config.json` 内容转换为 `config.yaml`
2. 使用提供的模板文件作为参考

#### 选项B: 继续使用JSON格式
- 现有的 `config.json` 可以直接使用，无需修改

### 第4步: 设置环境变量
1. 复制 `.env.example` 为 `.env`
2. 填入您的API密钥和其他敏感信息

### 第5步: 更新代码调用

#### 旧版调用方式:
```python
from config_loader import load_config
config = load_config()
```

#### 新版调用方式:
```python
from enhanced_config_loader import load_config
config = load_config(environment="development")
```

### 第6步: 验证迁移
运行测试脚本验证配置加载正常:
```bash
python config_usage_example.py
```

## 配置文件优先级

新版配置加载器按以下优先级合并配置:

1. **默认配置** (内置)
2. **配置文件** (按环境选择)
3. **环境变量** (覆盖文件配置)
4. **best_params.json** (如果启用)

## 环境选择

配置文件查找优先级:
1. `config.{environment}.yaml`
2. `config.{environment}.yml`
3. `config.{environment}.json`
4. `config.yaml`
5. `config.yml`
6. `config.json`

## 兼容性说明

### 完全兼容
- ✅ 所有现有的配置参数名称
- ✅ 配置字典的结构和访问方式
- ✅ `best_params.json` 合并功能
- ✅ 环境变量覆盖机制

### 新增功能
- 🆕 类型验证和错误检查
- 🆕 多环境配置支持
- 🆕 YAML格式支持
- 🆕 安全性增强

### 行为变化
- ⚠️ 默认启用配置验证（可禁用）
- ⚠️ 更严格的参数范围检查
- ⚠️ 更详细的错误信息

## 故障排除

### 常见问题

#### 1. 配置验证失败
```
ValueError: Configuration validation failed
```
**解决方案**: 检查配置参数是否在有效范围内，参考 `config_schema.py`

#### 2. API密钥验证失败
```
WARNING: Invalid API key format
```
**解决方案**: 确保API密钥格式正确，至少16位字符

#### 3. 配置文件未找到
```
WARNING: No config file found, using defaults
```
**解决方案**: 确保配置文件存在且命名正确

### 调试技巧

1. **启用详细日志**:
   ```python
   config = load_config(environment="development")  # DEBUG级别日志
   ```

2. **禁用验证**:
   ```python
   config = load_config(validate=False)
   ```

3. **检查配置合并结果**:
   ```python
   loader = EnhancedConfigLoader(environment="development")
   config = loader.load()
   print(json.dumps(config, indent=2))
   ```

## 性能对比

| 功能 | 旧版 | 新版 |
|------|------|------|
| 配置加载时间 | ~10ms | ~15ms |
| 内存使用 | 基础 | +20% (验证开销) |
| 错误检测 | 基础 | 完整 |
| 类型安全 | 无 | 完整 |

## 迁移检查清单

- [ ] 安装依赖包
- [ ] 备份现有配置
- [ ] 创建环境配置文件
- [ ] 设置 `.env` 文件
- [ ] 更新代码调用
- [ ] 运行测试验证
- [ ] 检查日志输出
- [ ] 验证所有环境配置

## 回滚计划

如果需要回滚到旧版本:
1. 恢复备份的配置文件
2. 使用旧版 `config_loader.py`
3. 移除新增的依赖包

## 支持

如有问题，请检查:
1. 配置文件格式是否正确
2. 环境变量是否设置
3. 依赖包是否安装
4. 日志输出中的错误信息



