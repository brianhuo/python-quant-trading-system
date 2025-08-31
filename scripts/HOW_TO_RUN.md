# 🚀 如何运行交易数据管道

## ⚡ 快速运行方法

### 方法1：从项目根目录运行（推荐）
```bash
cd /Users/richard/python
./run_trading_pipeline.sh
```

### 方法2：在scripts目录中运行
```bash
cd /Users/richard/python/scripts
./run_pipeline_here.sh
```

### 方法3：直接运行Python脚本
```bash
cd /Users/richard/python/scripts
python3 trading_data_pipeline.py
```

### 方法4：运行干净演示版本（无警告）
```bash
cd /Users/richard/python/scripts
python3 clean_demo.py
```

## 📊 运行结果
成功运行后，您会看到：
```
🚀 交易系统数据管道演示
✅ 成功获取数据: 97 行
📈 数据时间范围: 2025-08-29 14:30:00 到 2025-08-31 14:30:00
🔍 处理状态: success
⏱️ 处理时间: 0.0319 秒

📈 批量处理示例:
   AAPL: ✅ 49 行
   GOOGL: ✅ 49 行

📊 管道状态:
   成功率: 100.0%
   数据保留率: 100.0%
   平均处理时间: 0.0240 秒
```

## 💻 在您的代码中使用

```python
from trading_data_pipeline import create_default_pipeline

# 创建数据管道
pipeline = create_default_pipeline()

# 获取清洗后的数据
clean_data, report = pipeline.get_clean_data("AAPL", "30min", 1000)

# 检查结果
if report["status"] == "success":
    print(f"获得 {clean_data.shape[0]} 行清洁数据")
    # 数据已经清洗完毕，可以直接用于策略分析
else:
    print(f"数据获取失败: {report.get('error')}")
```

## 🎯 这就是您交易系统的数据层！

现在您有了一个完整的数据获取和清洗系统，它整合了：
- ✅ 配置管理 (ConfigLoader)
- ✅ 日志系统 (LoggerSetup)  
- ✅ 数据客户端 (UnifiedDataClient)
- ✅ 数据质量检查 (DataHealthChecker)

准备好为您的交易策略提供清洁、可靠的数据！🎉
