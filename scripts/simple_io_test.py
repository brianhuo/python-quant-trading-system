"""
简化的输入输出测试
清晰展示DataHealthChecker的输入输出
"""

import pandas as pd
import numpy as np
from datetime import datetime

from enhanced_data_health_checker import EnhancedDataHealthChecker
from data_cleaner import DataCleaner, CleaningConfig, CleaningMethod
from mock_data_generator import MockDataGenerator


def demo_basic_input_output():
    """演示基本输入输出"""
    print("📊 数据健康检查器 - 基本输入输出演示")
    print("="*60)
    
    print("\n🔧 1. 输入数据要求:")
    print("   ✅ pandas DataFrame")
    print("   ✅ 包含OHLCV列 (open, high, low, close, volume)")
    print("   ✅ DatetimeIndex时间索引 (推荐)")
    print("   ✅ 数值型数据")
    
    # 创建示例数据
    generator = MockDataGenerator()
    df = generator.generate_historical_data("DEMO", "30min", records=20)
    
    print(f"\n📥 输入示例:")
    print(f"   数据类型: {type(df)}")
    print(f"   数据形状: {df.shape}")
    print(f"   列名: {list(df.columns)}")
    print(f"   时间跨度: {df.index[0]} 到 {df.index[-1]}")
    print(f"\n   前3行数据:")
    print(df.head(3).to_string())
    
    # 添加一些问题用于演示
    print(f"\n🔧 2. 模拟数据问题:")
    df_with_issues = df.copy()
    df_with_issues.iloc[5, 1] = np.nan  # 缺失值
    df_with_issues.iloc[8, 3] = df_with_issues.iloc[8, 3] * 10  # 异常值
    df_with_issues.iloc[12, 4] = -500  # 负成交量
    
    print("   ➕ 添加缺失值 (第6行, high列)")
    print("   ➕ 添加异常值 (第9行, close列, 10倍价格)")
    print("   ➕ 添加负成交量 (第13行, volume列)")
    
    # 执行健康检查
    print(f"\n🔍 3. 执行健康检查:")
    checker = EnhancedDataHealthChecker()
    report = checker.comprehensive_health_check(df_with_issues, clean_data=True, save_report=False)
    
    print(f"\n📤 4. 输出结果:")
    print(f"   状态级别: {report.status.value}")
    print(f"   发现问题: {len(report.issues)} 个")
    print(f"   处理耗时: {report.processing_time:.4f} 秒")
    print(f"   原始数据: {report.original_shape[0]} 行 × {report.original_shape[1]} 列")
    print(f"   清洗后: {report.cleaned_shape[0]} 行 × {report.cleaned_shape[1]} 列")
    
    if report.issues:
        print(f"\n   🔍 检测到的问题:")
        for i, issue in enumerate(report.issues, 1):
            severity = "🚨" if issue.severity.value == "critical" else "⚠️"
            print(f"     {i}. {severity} {issue.description}")
    
    if report.cleaned_data is not None:
        print(f"\n   📋 清洗后数据预览:")
        print(report.cleaned_data.head(3).to_string())
        print(f"\n   ✅ 清洗效果:")
        print(f"     缺失值: {report.cleaned_data.isnull().sum().sum()}")
        print(f"     负值数量: {(report.cleaned_data < 0).sum().sum()}")


def demo_different_data_scenarios():
    """演示不同数据场景"""
    print(f"\n\n🧪 不同数据场景测试")
    print("="*60)
    
    checker = EnhancedDataHealthChecker()
    
    scenarios = [
        ("完美数据", "无任何问题的理想数据"),
        ("缺失值数据", "包含各种缺失值"),
        ("异常值数据", "包含价格和成交量异常"),
        ("时间问题数据", "时间间隙和重复时间戳"),
    ]
    
    for i, (name, description) in enumerate(scenarios, 1):
        print(f"\n📝 场景 {i}: {name}")
        print(f"   描述: {description}")
        
        # 创建对应的测试数据
        generator = MockDataGenerator()
        test_df = generator.generate_historical_data(f"TEST{i}", "30min", records=20)
        
        if name == "缺失值数据":
            test_df.iloc[3:6, 1] = np.nan
            test_df.iloc[8, 3] = np.nan
        elif name == "异常值数据":
            test_df.iloc[5, 1] = test_df.iloc[5, 1] * 20
            test_df.iloc[10, 4] = -1000
        elif name == "时间问题数据":
            test_df = test_df.drop(test_df.index[8:12])  # 删除时间点
        
        # 执行检查
        report = checker.comprehensive_health_check(test_df, clean_data=True, save_report=False)
        
        # 显示结果
        print(f"   📊 输入: {test_df.shape[0]} 行")
        print(f"   📊 输出: {report.cleaned_shape[0]} 行")
        print(f"   🎯 状态: {report.status.value}")
        print(f"   ⚠️ 问题: {len(report.issues)} 个")


def demo_cleaner_modes():
    """演示不同清洗模式"""
    print(f"\n\n🧹 数据清洗模式对比")
    print("="*60)
    
    # 创建有问题的数据
    generator = MockDataGenerator()
    dirty_data = generator.generate_historical_data("DIRTY", "30min", records=30)
    
    # 添加各种问题
    dirty_data.iloc[5:8, 1] = np.nan  # 缺失值
    dirty_data.iloc[10, 3] = dirty_data.iloc[10, 3] * 50  # 异常值
    dirty_data.iloc[15, 4] = -2000  # 负成交量
    dirty_data.iloc[20, 0] = 0  # 零开盘价
    
    print(f"📥 原始问题数据:")
    print(f"   数据形状: {dirty_data.shape}")
    print(f"   缺失值: {dirty_data.isnull().sum().sum()}")
    print(f"   负值: {(dirty_data < 0).sum().sum()}")
    print(f"   零值: {(dirty_data == 0).sum().sum()}")
    
    # 测试不同模式
    modes = {
        "保守模式": CleaningConfig(
            missing_value_method=CleaningMethod.INTERPOLATE,
            outlier_method=CleaningMethod.MEDIAN_FILL,
            remove_invalid_ohlc=False
        ),
        "平衡模式": CleaningConfig(
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
        print(f"\n🔧 {mode_name}:")
        cleaner = DataCleaner(config)
        cleaned_df, _ = cleaner.comprehensive_clean(dirty_data.copy())
        
        retention_rate = (len(cleaned_df) / len(dirty_data)) * 100
        
        print(f"   📊 结果: {dirty_data.shape[0]} → {cleaned_df.shape[0]} 行")
        print(f"   📈 保留率: {retention_rate:.1f}%")
        print(f"   ✅ 清洗效果:")
        print(f"     缺失值: {cleaned_df.isnull().sum().sum()}")
        print(f"     负值: {(cleaned_df < 0).sum().sum()}")
        print(f"     零值: {(cleaned_df == 0).sum().sum()}")


def demo_output_formats():
    """演示输出格式"""
    print(f"\n\n📄 输出格式演示")
    print("="*60)
    
    generator = MockDataGenerator()
    test_data = generator.generate_historical_data("OUTPUT_TEST", "30min", records=15)
    test_data.iloc[3, 1] = np.nan  # 添加一个问题
    
    checker = EnhancedDataHealthChecker()
    report = checker.comprehensive_health_check(test_data, clean_data=True, save_report=False)
    
    print("📊 可用的输出格式:")
    
    # 1. 报告摘要
    print("\n1️⃣ 报告摘要 (get_summary()):")
    summary = report.get_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    # 2. 清洗后数据
    print(f"\n2️⃣ 清洗后数据 (cleaned_data):")
    if report.cleaned_data is not None:
        print(f"   类型: {type(report.cleaned_data)}")
        print(f"   形状: {report.cleaned_data.shape}")
        print("   前3行:")
        print(report.cleaned_data.head(3).to_string())
    
    # 3. 问题列表
    print(f"\n3️⃣ 问题详情 (issues):")
    for i, issue in enumerate(report.issues, 1):
        print(f"   问题 {i}:")
        print(f"     类型: {issue.issue_type.value}")
        print(f"     严重程度: {issue.severity.value}")
        print(f"     列: {issue.column}")
        print(f"     描述: {issue.description}")
    
    # 4. 统计信息
    print(f"\n4️⃣ 统计信息 (statistics):")
    if 'basic' in report.statistics:
        print(f"   数据形状: {report.statistics['basic']['shape']}")
        print(f"   内存使用: {report.statistics['basic']['memory_usage'] / 1024:.2f} KB")
    
    if 'financial' in report.statistics:
        financial = report.statistics['financial']
        print(f"   价格波动率: {financial['volatility']:.6f}")
        print(f"   平均收益率: {financial['mean_return']:.6f}")


if __name__ == "__main__":
    print("🚀 数据健康检查器 - 输入输出测试")
    print("目标：清晰展示输入什么，输出什么")
    
    demo_basic_input_output()
    demo_different_data_scenarios()
    demo_cleaner_modes()
    demo_output_formats()
    
    print(f"\n" + "="*60)
    print("🎉 测试完成!")
    print("\n💡 核心要点:")
    print("📥 输入: pandas DataFrame (OHLCV格式)")
    print("🔍 处理: 7类问题检测 + 智能清洗")
    print("📤 输出: 清洗后数据 + 详细报告 + 统计信息")
    print("⚙️ 配置: 灵活的清洗策略和阈值设置")
    print("🎯 状态: 4级健康状态 (healthy/warning/critical/failed)")
    print("\n🚀 现在您完全了解了数据健康检查器的输入输出！")



