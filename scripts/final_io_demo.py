"""
数据健康检查器 - 最终输入输出演示
清晰展示输入什么数据，输出什么结果
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from enhanced_data_health_checker import EnhancedDataHealthChecker
from data_cleaner import DataCleaner, CleaningConfig, CleaningMethod


def create_sample_data():
    """创建示例数据"""
    # 创建30行的示例股票数据
    dates = pd.date_range(start='2024-01-01 09:30:00', periods=30, freq='30min')
    
    # 生成模拟价格数据
    np.random.seed(42)  # 固定随机种子确保结果一致
    base_price = 150.0
    price_changes = np.random.normal(0, 0.02, 30)
    prices = []
    current_price = base_price
    
    for change in price_changes:
        current_price *= (1 + change)
        prices.append(current_price)
    
    # 创建OHLCV数据
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'datetime'
    return df


def main_demo():
    """主要演示"""
    print("🧪 数据健康检查器 - 输入输出演示")
    print("=" * 70)
    
    # 1. 创建示例数据
    print("\n📊 1. 输入数据展示")
    print("-" * 50)
    df = create_sample_data()
    
    print("✅ 输入要求:")
    print("   - pandas DataFrame")
    print("   - 包含 ['open', 'high', 'low', 'close', 'volume'] 列")
    print("   - DatetimeIndex 时间索引 (推荐)")
    print("   - 数值型数据")
    
    print(f"\n📥 实际输入:")
    print(f"   数据类型: {type(df)}")
    print(f"   数据形状: {df.shape}")
    print(f"   列名: {list(df.columns)}")
    print(f"   时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"   数据类型:\n{df.dtypes.to_string()}")
    
    print(f"\n📋 数据预览 (前5行):")
    print(df.head().to_string())
    
    # 2. 模拟数据问题
    print(f"\n🔧 2. 模拟数据问题")
    print("-" * 50)
    df_with_issues = df.copy()
    
    # 添加各种问题
    df_with_issues.iloc[5:8, 1] = np.nan  # 缺失值
    df_with_issues.iloc[10, 3] = df_with_issues.iloc[10, 3] * 20  # 异常值
    df_with_issues.iloc[15, 4] = -1000  # 负成交量
    df_with_issues.iloc[20, 0] = 0  # 零开盘价
    
    print("添加的问题:")
    print("   ➕ 缺失值: 第6-8行的high列")
    print("   ➕ 异常值: 第11行的close价格 (20倍)")
    print("   ➕ 负成交量: 第16行的volume")
    print("   ➕ 零开盘价: 第21行的open")
    
    print(f"\n📊 问题统计:")
    print(f"   缺失值总数: {df_with_issues.isnull().sum().sum()}")
    print(f"   负值数量: {(df_with_issues < 0).sum().sum()}")
    print(f"   零值数量: {(df_with_issues == 0).sum().sum()}")
    
    # 3. 执行健康检查
    print(f"\n🔍 3. 健康检查过程")
    print("-" * 50)
    checker = EnhancedDataHealthChecker()
    
    print("执行步骤:")
    print("   1️⃣ 缺失值检查和处理")
    print("   2️⃣ 异常值检测和处理") 
    print("   3️⃣ 时间连续性验证")
    print("   4️⃣ 数据频率一致性检查")
    print("   5️⃣ 价格逻辑关系验证")
    print("   6️⃣ 成交量异常检查")
    print("   7️⃣ 统计信息计算")
    
    # 执行检查
    report = checker.comprehensive_health_check(df_with_issues, clean_data=True, save_report=False)
    
    # 4. 输出结果
    print(f"\n📤 4. 输出结果详解")
    print("-" * 50)
    
    print(f"🎯 整体结果:")
    print(f"   健康状态: {report.status.value.upper()}")
    print(f"   发现问题: {len(report.issues)} 个")
    print(f"   处理时间: {report.processing_time:.4f} 秒")
    print(f"   数据变化: {report.original_shape[0]} → {report.cleaned_shape[0]} 行")
    
    if report.original_shape[0] > 0:
        retention = (report.cleaned_shape[0] / report.original_shape[0]) * 100
        print(f"   数据保留率: {retention:.1f}%")
    
    # 显示检测到的问题
    if report.issues:
        print(f"\n🔍 检测到的问题:")
        for i, issue in enumerate(report.issues, 1):
            severity_emoji = "🚨" if issue.severity.value == "critical" else "⚠️"
            print(f"   {i}. {severity_emoji} [{issue.issue_type.value}] {issue.description}")
            if issue.suggestion:
                print(f"      💡 处理建议: {issue.suggestion}")
    else:
        print(f"\n✅ 未发现任何问题")
    
    # 显示清洗后数据
    if report.cleaned_data is not None:
        print(f"\n📋 清洗后数据:")
        print(f"   数据形状: {report.cleaned_data.shape}")
        print(f"   数据质量:")
        print(f"     缺失值: {report.cleaned_data.isnull().sum().sum()}")
        print(f"     负值: {(report.cleaned_data < 0).sum().sum()}")
        print(f"     零值: {(report.cleaned_data == 0).sum().sum()}")
        
        print(f"\n   清洗后预览 (前3行):")
        print(report.cleaned_data.head(3).to_string())
    
    # 显示统计信息
    if report.statistics:
        print(f"\n📊 数据统计信息:")
        if 'basic' in report.statistics:
            basic = report.statistics['basic']
            print(f"   内存使用: {basic['memory_usage'] / 1024:.2f} KB")
        
        if 'financial' in report.statistics:
            financial = report.statistics['financial']
            print(f"   价格波动率: {financial['volatility']:.6f}")
            print(f"   平均收益率: {financial['mean_return']:.6f}")
            print(f"   收益率范围: {financial['min_return']:.6f} 到 {financial['max_return']:.6f}")
    
    # 5. 不同清洗模式对比
    print(f"\n🧹 5. 清洗模式对比")
    print("-" * 50)
    
    modes = {
        "保守模式": CleaningConfig(
            missing_value_method=CleaningMethod.INTERPOLATE,
            outlier_method=CleaningMethod.MEDIAN_FILL,
            remove_invalid_ohlc=False
        ),
        "标准模式": CleaningConfig(
            missing_value_method=CleaningMethod.INTERPOLATE,
            outlier_method=CleaningMethod.MEDIAN_FILL,
            remove_invalid_ohlc=True
        ),
        "严格模式": CleaningConfig(
            missing_value_method=CleaningMethod.DROP,
            outlier_method=CleaningMethod.DROP,
            remove_invalid_ohlc=True
        )
    }
    
    for mode_name, config in modes.items():
        cleaner = DataCleaner(config)
        cleaned_df, log = cleaner.comprehensive_clean(df_with_issues.copy())
        retention = (len(cleaned_df) / len(df_with_issues)) * 100
        
        print(f"\n🔧 {mode_name}:")
        print(f"   数据保留: {len(df_with_issues)} → {len(cleaned_df)} 行 ({retention:.1f}%)")
        print(f"   清洗操作: {len(log)} 次")
        print(f"   最终质量:")
        print(f"     缺失值: {cleaned_df.isnull().sum().sum()}")
        print(f"     负值: {(cleaned_df < 0).sum().sum()}")
    
    # 6. 输出格式汇总
    print(f"\n📋 6. 可用的输出格式")
    print("-" * 50)
    print("✅ 主要输出:")
    print("   1. report.status - 健康状态 (healthy/warning/critical/failed)")
    print("   2. report.cleaned_data - 清洗后的pandas DataFrame")
    print("   3. report.issues - 问题列表 (类型、严重程度、建议)")
    print("   4. report.statistics - 详细统计信息")
    print("   5. report.get_summary() - 摘要报告字典")
    print("   6. report.to_json() - JSON格式报告")
    
    print(f"\n✅ 数据清洗器输出:")
    print("   1. cleaned_dataframe - 清洗后数据")
    print("   2. cleaning_log - 详细清洗日志")
    print("   3. get_cleaning_summary() - 清洗摘要")


if __name__ == "__main__":
    main_demo()
    
    print(f"\n" + "=" * 70)
    print("🎉 演示完成!")
    print("\n💡 核心要点总结:")
    print("📥 输入: pandas DataFrame (OHLCV格式 + DatetimeIndex)")
    print("🔍 检查: 7类问题 (缺失值/异常值/时间/频率/价格逻辑/成交量/统计)")
    print("🧹 清洗: 3种模式 (保守/标准/严格)")
    print("📤 输出: 清洗数据 + 问题报告 + 统计信息 + 处理日志")
    print("⚙️ 配置: 完全可配置的阈值和策略")
    print("🎯 状态: 4级健康状态分级管理")
    print("📊 格式: DataFrame/字典/JSON多种输出格式")
    print("\n🚀 现在您完全掌握了数据健康检查器的输入输出！")




