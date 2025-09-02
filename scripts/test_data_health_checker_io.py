"""
数据健康检查器输入输出测试
详细展示输入什么数据，输出什么结果
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from enhanced_data_health_checker import EnhancedDataHealthChecker, HealthStatus
from data_cleaner import DataCleaner, CleaningConfig, CleaningMethod
from mock_data_generator import MockDataGenerator


def create_test_datasets():
    """创建各种测试数据集"""
    print("🔧 创建测试数据集...")
    
    datasets = {}
    
    # 1. 完美数据 - 无任何问题
    print("  📊 创建完美数据集...")
    generator = MockDataGenerator(base_price=150.0, volatility=0.01)
    perfect_data = generator.generate_historical_data("PERFECT", "30min", records=50)
    datasets['perfect'] = perfect_data
    print(f"     形状: {perfect_data.shape}")
    
    # 2. 有缺失值的数据
    print("  📊 创建缺失值数据集...")
    missing_data = perfect_data.copy()
    # 添加各种缺失值
    missing_data.iloc[5:8, 1] = np.nan  # high列缺失
    missing_data.iloc[10:12, 3] = np.nan  # close列缺失
    missing_data.iloc[15, :] = np.nan  # 整行缺失
    missing_data.iloc[20:22, 4] = np.nan  # volume列缺失
    datasets['missing'] = missing_data
    print(f"     形状: {missing_data.shape}, 缺失值: {missing_data.isnull().sum().sum()}")
    
    # 3. 有异常值的数据
    print("  📊 创建异常值数据集...")
    outlier_data = perfect_data.copy()
    # 添加各种异常值
    outlier_data.iloc[8, 1] = outlier_data.iloc[8, 1] * 10  # 极高价格
    outlier_data.iloc[12, 2] = outlier_data.iloc[12, 2] * 0.1  # 极低价格
    outlier_data.iloc[16, 4] = outlier_data.iloc[16, 4] * 100  # 极高成交量
    outlier_data.iloc[25, 4] = -1000  # 负成交量
    datasets['outlier'] = outlier_data
    print(f"     形状: {outlier_data.shape}")
    
    # 4. 时间连续性问题数据
    print("  📊 创建时间问题数据集...")
    time_data = perfect_data.copy()
    # 删除一些时间点造成间隙
    time_data = time_data.drop(time_data.index[10:15])  # 删除5个时间点
    # 添加重复时间戳
    duplicate_row = time_data.iloc[5:6].copy()
    time_data = pd.concat([time_data, duplicate_row])
    time_data = time_data.sort_index()
    datasets['time_issues'] = time_data
    print(f"     形状: {time_data.shape}")
    
    # 5. 价格逻辑错误数据
    print("  📊 创建价格逻辑错误数据集...")
    logic_data = perfect_data.copy()
    # 创建OHLC逻辑错误
    logic_data.iloc[6, 1] = logic_data.iloc[6, 2] - 10  # high < low
    logic_data.iloc[14, 0] = 0  # open = 0
    logic_data.iloc[18, 3] = -5  # close < 0
    datasets['logic_errors'] = logic_data
    print(f"     形状: {logic_data.shape}")
    
    # 6. 综合问题数据
    print("  📊 创建综合问题数据集...")
    complex_data = perfect_data.copy()
    # 添加多种问题
    complex_data.iloc[3:6, 1] = np.nan  # 缺失值
    complex_data.iloc[8, 3] = complex_data.iloc[8, 3] * 20  # 异常值
    complex_data.iloc[12, 4] = -500  # 负成交量
    complex_data.iloc[16, 1] = complex_data.iloc[16, 2] - 5  # 逻辑错误
    complex_data = complex_data.drop(complex_data.index[20:25])  # 时间间隙
    datasets['complex'] = complex_data
    print(f"     形状: {complex_data.shape}")
    
    print(f"✅ 创建了 {len(datasets)} 个测试数据集")
    return datasets


def test_input_output_detailed():
    """详细测试输入输出"""
    print("\n" + "="*80)
    print("🧪 数据健康检查器 - 详细输入输出测试")
    print("="*80)
    
    # 创建测试数据
    datasets = create_test_datasets()
    
    # 创建检查器
    checker = EnhancedDataHealthChecker()
    
    for data_name, df in datasets.items():
        print(f"\n{'🔍 测试数据集: ' + data_name.upper():-^70}")
        
        # 显示输入数据信息
        print(f"\n📥 输入数据:")
        print(f"   类型: {type(df)}")
        print(f"   形状: {df.shape}")
        print(f"   列名: {list(df.columns)}")
        print(f"   索引类型: {type(df.index)}")
        print(f"   数据类型:\n{df.dtypes.to_string()}")
        print(f"   内存使用: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # 显示数据样本
        print(f"\n📋 数据样本 (前3行):")
        print(df.head(3).to_string())
        
        # 显示数据问题概览
        print(f"\n⚠️ 数据问题概览:")
        print(f"   缺失值总数: {df.isnull().sum().sum()}")
        print(f"   零值数量: {(df == 0).sum().sum()}")
        print(f"   负值数量: {(df < 0).sum().sum()}")
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"   重复时间戳: {df.index.duplicated().sum()}")
            time_diffs = df.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                print(f"   时间间隔变化: {len(time_diffs.unique())} 种不同间隔")
        
        # 执行健康检查
        print(f"\n🔍 执行健康检查...")
        try:
            report = checker.comprehensive_health_check(df, clean_data=True, save_report=False)
            
            # 显示输出结果
            print(f"\n📤 输出结果:")
            print(f"   检查状态: {report.status.value}")
            print(f"   问题总数: {len(report.issues)}")
            print(f"   处理时间: {report.processing_time:.4f} 秒")
            print(f"   原始数据: {report.original_shape}")
            print(f"   清洗后数据: {report.cleaned_shape}")
            
            if report.cleaned_data is not None:
                reduction = report.original_shape[0] - report.cleaned_shape[0]
                retention = (report.cleaned_shape[0] / report.original_shape[0]) * 100
                print(f"   数据删除: {reduction} 行 ({100-retention:.1f}%)")
                print(f"   数据保留: {retention:.1f}%")
            
            # 显示问题详情
            if report.issues:
                print(f"\n🔍 发现的问题:")
                for i, issue in enumerate(report.issues[:5], 1):  # 只显示前5个
                    severity_emoji = "🚨" if issue.severity == HealthStatus.CRITICAL else "⚠️"
                    print(f"   {i}. {severity_emoji} [{issue.issue_type.value}] {issue.description}")
                    if issue.suggestion:
                        print(f"      💡 建议: {issue.suggestion}")
                
                if len(report.issues) > 5:
                    print(f"   ... 还有 {len(report.issues) - 5} 个问题")
            else:
                print(f"   ✅ 未发现问题")
            
            # 显示统计信息
            if report.statistics:
                print(f"\n📊 统计信息:")
                if 'basic' in report.statistics:
                    basic = report.statistics['basic']
                    print(f"   内存使用: {basic.get('memory_usage', 0) / 1024:.2f} KB")
                
                if 'financial' in report.statistics:
                    financial = report.statistics['financial']
                    print(f"   价格波动率: {financial.get('volatility', 0):.4f}")
                    print(f"   平均收益率: {financial.get('mean_return', 0):.4f}")
            
            # 显示清洗后数据样本
            if report.cleaned_data is not None and not report.cleaned_data.empty:
                print(f"\n📋 清洗后数据样本 (前3行):")
                print(report.cleaned_data.head(3).to_string())
                
                print(f"\n✅ 清洗后数据质量:")
                print(f"   缺失值: {report.cleaned_data.isnull().sum().sum()}")
                print(f"   负值: {(report.cleaned_data < 0).sum().sum()}")
            
        except Exception as e:
            print(f"❌ 检查失败: {e}")
        
        print(f"\n{'-'*70}")


def test_different_input_formats():
    """测试不同输入格式"""
    print(f"\n{'🔧 测试不同输入格式':-^70}")
    
    checker = EnhancedDataHealthChecker()
    
    # 1. 测试空DataFrame
    print(f"\n📝 测试1: 空DataFrame")
    empty_df = pd.DataFrame()
    print(f"   输入: 空DataFrame")
    report = checker.comprehensive_health_check(empty_df, clean_data=False, save_report=False)
    print(f"   输出状态: {report.status.value}")
    print(f"   问题数: {len(report.issues)}")
    
    # 2. 测试只有一行数据
    print(f"\n📝 测试2: 单行数据")
    single_row = pd.DataFrame({
        'open': [100.0],
        'high': [105.0], 
        'low': [95.0],
        'close': [102.0],
        'volume': [1000000]
    }, index=pd.DatetimeIndex(['2024-01-01']))
    print(f"   输入: {single_row.shape[0]} 行数据")
    report = checker.comprehensive_health_check(single_row, clean_data=False, save_report=False)
    print(f"   输出状态: {report.status.value}")
    print(f"   问题数: {len(report.issues)}")
    
    # 3. 测试非标准列名
    print(f"\n📝 测试3: 非标准列名")
    custom_df = pd.DataFrame({
        'price_open': [100.0, 101.0],
        'price_close': [102.0, 103.0],
        'trade_volume': [1000, 1500]
    })
    print(f"   输入: 自定义列名 {list(custom_df.columns)}")
    report = checker.comprehensive_health_check(custom_df, clean_data=False, save_report=False)
    print(f"   输出状态: {report.status.value}")
    print(f"   问题数: {len(report.issues)}")
    
    # 4. 测试超大数据集模拟
    print(f"\n📝 测试4: 大数据集处理")
    generator = MockDataGenerator()
    large_df = generator.generate_historical_data("LARGE", "1min", records=5000)
    print(f"   输入: {large_df.shape[0]} 行数据 ({large_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB)")
    start_time = datetime.now()
    report = checker.comprehensive_health_check(large_df, clean_data=True, save_report=False)
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"   输出状态: {report.status.value}")
    print(f"   处理时间: {processing_time:.2f} 秒")
    print(f"   处理速度: {large_df.shape[0] / processing_time:.0f} 行/秒")


def test_cleaner_standalone():
    """测试独立的数据清洗器"""
    print(f"\n{'🧹 测试独立数据清洗器':-^70}")
    
    # 创建有问题的数据
    generator = MockDataGenerator()
    dirty_data = generator.generate_historical_data("DIRTY", "30min", records=100)
    
    # 人工添加各种问题
    dirty_data.iloc[10:15, 1] = np.nan  # 缺失值
    dirty_data.iloc[20, 3] = dirty_data.iloc[20, 3] * 50  # 异常值
    dirty_data.iloc[30, 4] = -1000  # 负成交量
    dirty_data.iloc[40, 0] = 0  # 零开盘价
    
    print(f"\n📥 清洗器输入:")
    print(f"   数据形状: {dirty_data.shape}")
    print(f"   缺失值: {dirty_data.isnull().sum().sum()}")
    print(f"   负值数量: {(dirty_data < 0).sum().sum()}")
    print(f"   零值数量: {(dirty_data == 0).sum().sum()}")
    
    # 测试不同清洗配置
    configs = {
        "宽松模式": CleaningConfig(
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
    
    for mode_name, config in configs.items():
        print(f"\n🔧 {mode_name}:")
        cleaner = DataCleaner(config)
        cleaned_data, cleaning_log = cleaner.comprehensive_clean(dirty_data.copy())
        
        print(f"   📤 清洗器输出:")
        print(f"     清洗后形状: {cleaned_data.shape}")
        print(f"     数据保留率: {(len(cleaned_data) / len(dirty_data)) * 100:.1f}%")
        print(f"     清洗操作数: {len(cleaning_log)}")
        print(f"     缺失值: {cleaned_data.isnull().sum().sum()}")
        print(f"     负值数量: {(cleaned_data < 0).sum().sum()}")
        
        # 显示清洗操作
        if cleaning_log:
            print(f"     主要操作:")
            for log_entry in cleaning_log[:3]:
                print(f"       - {log_entry['action']}: {log_entry['details']}")


def demonstrate_json_output():
    """演示JSON输出格式"""
    print(f"\n{'📄 JSON输出格式演示':-^70}")
    
    generator = MockDataGenerator()
    test_data = generator.generate_historical_data("JSON_TEST", "30min", records=20)
    
    # 添加一些问题
    test_data.iloc[5, 1] = np.nan
    test_data.iloc[10, 3] = test_data.iloc[10, 3] * 10
    
    checker = EnhancedDataHealthChecker()
    report = checker.comprehensive_health_check(test_data, clean_data=True, save_report=False)
    
    print(f"\n📤 JSON报告输出:")
    try:
        json_output = report.to_json()
        # 解析并美化显示
        data = json.loads(json_output)
        
        print(f"\n🔍 报告结构:")
        print(f"   - summary: 摘要信息")
        print(f"   - issues: 问题列表 ({len(data.get('issues', []))} 个)")
        print(f"   - statistics: 统计信息")
        print(f"   - timestamp: 生成时间")
        
        print(f"\n📊 摘要信息:")
        summary = data.get('summary', {})
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n⚠️ 问题示例 (前2个):")
        for i, issue in enumerate(data.get('issues', [])[:2], 1):
            print(f"   问题 {i}:")
            print(f"     类型: {issue.get('issue_type')}")
            print(f"     严重程度: {issue.get('severity')}")
            print(f"     描述: {issue.get('description')}")
            print(f"     建议: {issue.get('suggestion', '无')}")
        
    except Exception as e:
        print(f"❌ JSON生成失败: {e}")


if __name__ == "__main__":
    print("🧪 数据健康检查器 - 输入输出测试")
    print("="*80)
    print("测试目标：详细了解输入什么数据，输出什么结果")
    
    # 执行所有测试
    test_input_output_detailed()
    test_different_input_formats()
    test_cleaner_standalone()
    demonstrate_json_output()
    
    print(f"\n" + "="*80)
    print("🎉 测试完成!")
    print("\n💡 总结:")
    print("✅ 支持多种数据格式和大小")
    print("✅ 提供详细的问题诊断")
    print("✅ 自动数据清洗和修复")
    print("✅ 结构化的JSON报告输出")
    print("✅ 灵活的配置选项")
    print("✅ 高性能处理能力")




