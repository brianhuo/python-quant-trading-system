#!/usr/bin/env python3
"""
优化后的EnhancedFeatureEngineer演示脚本

展示批处理和实时模式的性能差异和功能特性
"""

import pandas as pd
import numpy as np
import time
import sys
import os

# 添加脚本目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_feature_engineer import EnhancedFeatureEngineer
from logger_setup import setup_logging

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """生成模拟的OHLCV数据"""
    np.random.seed(42)
    
    # 生成基础价格走势
    base_price = 100
    price_changes = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # 确保价格为正
    
    prices = np.array(prices[1:])
    
    # 生成OHLCV数据
    data = pd.DataFrame({
        'open': prices * np.random.uniform(0.98, 1.02, n_samples),
        'high': prices * np.random.uniform(1.00, 1.05, n_samples),
        'low': prices * np.random.uniform(0.95, 1.00, n_samples),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # 确保OHLC关系正确
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
    
    # 设置时间索引
    data.index = pd.date_range('2024-01-01', periods=n_samples, freq='30min')
    
    return data

def benchmark_performance():
    """性能基准测试"""
    print("=" * 60)
    print("EnhancedFeatureEngineer 性能基准测试")
    print("=" * 60)
    
    logger = setup_logging()
    fe = EnhancedFeatureEngineer(logger=logger)
    
    # 生成不同规模的测试数据
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        print(f"\n测试数据规模: {size} 行")
        print("-" * 40)
        
        data = generate_sample_data(size)
        
        # 批处理模式测试
        start_time = time.time()
        batch_features = fe.create_features(data, mode='batch')
        batch_time = time.time() - start_time
        
        # 实时模式测试
        start_time = time.time()
        realtime_features = fe.create_features(data, mode='realtime')
        realtime_time = time.time() - start_time
        
        # 第二次实时测试（测试缓存效果）
        start_time = time.time()
        realtime_features_cached = fe.create_features(data, mode='realtime')
        realtime_cached_time = time.time() - start_time
        
        print(f"批处理模式: {batch_time:.3f}秒, 输出形状: {batch_features.shape}")
        print(f"实时模式: {realtime_time:.3f}秒, 输出形状: {realtime_features.shape}")
        print(f"实时模式(缓存): {realtime_cached_time:.3f}秒")
        print(f"性能提升: {batch_time/realtime_time:.1f}x (实时 vs 批处理)")
        
        if batch_time > 0:
            print(f"缓存效果: {realtime_time/realtime_cached_time:.1f}x (缓存 vs 首次)")

def demonstrate_features():
    """演示特征生成功能"""
    print("\n" + "=" * 60)
    print("特征生成功能演示")
    print("=" * 60)
    
    logger = setup_logging()
    fe = EnhancedFeatureEngineer(logger=logger)
    
    # 生成示例数据
    data = generate_sample_data(200)
    print(f"原始数据形状: {data.shape}")
    print(f"原始数据列: {list(data.columns)}")
    
    # 批处理模式
    print("\n1. 批处理模式特征生成:")
    batch_features = fe.create_features(data, mode='batch')
    print(f"批处理特征形状: {batch_features.shape}")
    print("批处理特征样本:")
    if not batch_features.empty:
        print(batch_features.tail(3).round(4))
    
    # 实时模式
    print("\n2. 实时模式特征生成:")
    realtime_features = fe.create_features(data, mode='realtime')
    print(f"实时特征形状: {realtime_features.shape}")
    print("实时特征样本:")
    if not realtime_features.empty:
        print(realtime_features.round(4))
    
    # 性能统计
    print("\n3. 性能统计:")
    stats = fe.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

def test_incremental_updates():
    """测试增量更新功能"""
    print("\n" + "=" * 60)
    print("增量更新功能测试")
    print("=" * 60)
    
    logger = setup_logging()
    fe = EnhancedFeatureEngineer(logger=logger)
    
    # 基础数据
    base_data = generate_sample_data(500)
    print(f"基础数据: {base_data.shape}")
    
    # 首次计算
    start_time = time.time()
    features1 = fe.create_features(base_data, mode='realtime')
    time1 = time.time() - start_time
    print(f"首次计算: {time1:.3f}秒")
    
    # 相同数据的第二次计算（测试缓存）
    start_time = time.time()
    features_cached = fe.create_features(base_data, mode='realtime')
    time_cached = time.time() - start_time
    print(f"相同数据缓存计算: {time_cached:.4f}秒")
    
    # 添加新数据（模拟实时更新）
    new_data = generate_sample_data(50)
    new_data.index = pd.date_range(base_data.index[-1] + pd.Timedelta('30min'), 
                                   periods=50, freq='30min')
    combined_data = pd.concat([base_data, new_data])
    
    # 增量计算
    start_time = time.time()
    features2 = fe.create_features(combined_data, mode='realtime')
    time2 = time.time() - start_time
    print(f"增量计算: {time2:.3f}秒")
    print(f"缓存效果: {time1/time_cached:.1f}x")
    print(f"增量计算效率: {time1/time2:.1f}x")

def compare_feature_quality():
    """比较批处理和实时模式的特征质量"""
    print("\n" + "=" * 60)
    print("特征质量对比")
    print("=" * 60)
    
    logger = setup_logging()
    fe = EnhancedFeatureEngineer(logger=logger)
    
    # 生成测试数据
    data = generate_sample_data(300)
    
    # 生成两种模式的特征
    batch_features = fe.create_features(data, mode='batch')
    realtime_features = fe.create_features(data, mode='realtime')
    
    print(f"批处理特征数量: {batch_features.shape[1] if not batch_features.empty else 0}")
    print(f"实时特征数量: {realtime_features.shape[1] if not realtime_features.empty else 0}")
    
    if not batch_features.empty and not realtime_features.empty:
        # 找到共同特征
        common_features = set(batch_features.columns) & set(realtime_features.columns)
        print(f"共同特征数量: {len(common_features)}")
        
        if common_features:
            print("\n共同特征统计对比:")
            for feature in list(common_features)[:5]:  # 只显示前5个
                if feature in batch_features.columns and feature in realtime_features.columns:
                    batch_mean = batch_features[feature].mean()
                    realtime_mean = realtime_features[feature].mean()
                    print(f"{feature:15s}: 批处理={batch_mean:.4f}, 实时={realtime_mean:.4f}")

def main():
    """主函数"""
    print("优化后的EnhancedFeatureEngineer演示")
    print("作者: AI Assistant")
    print("日期:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # 运行所有演示
        benchmark_performance()
        demonstrate_features()
        test_incremental_updates()
        compare_feature_quality()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
