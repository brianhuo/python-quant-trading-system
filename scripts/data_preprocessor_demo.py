#!/usr/bin/env python3
"""
Enhanced DataPreprocessor 演示脚本

展示优化后的数据预处理器的各项功能和性能提升
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from pathlib import Path

# 添加脚本目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_data_preprocessor import EnhancedDataPreprocessor, PreprocessingConfig
from enhanced_feature_engineer import EnhancedFeatureEngineer
from logger_setup import setup_logging

def generate_sample_trading_data(n_samples: int = 2000) -> pd.DataFrame:
    """生成模拟的交易特征数据"""
    np.random.seed(42)
    
    # 时间索引
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='30min')
    
    # 基础价格数据
    base_price = 100
    price_changes = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))
    
    prices = np.array(prices[1:])
    
    # 创建OHLCV数据
    data = pd.DataFrame({
        'open': prices * np.random.uniform(0.98, 1.02, n_samples),
        'high': prices * np.random.uniform(1.00, 1.05, n_samples),
        'low': prices * np.random.uniform(0.95, 1.00, n_samples),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # 确保OHLC关系正确
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
    
    # 生成技术指标特征
    fe = EnhancedFeatureEngineer()
    features = fe.create_features(data, mode='batch')
    
    return features

def compare_preprocessing_methods():
    """比较不同预处理方法的性能"""
    print("=" * 70)
    print("数据预处理方法对比")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 生成测试数据
    data = generate_sample_trading_data(1500)
    print(f"测试数据: {data.shape}")
    print(f"特征列: {list(data.columns)}")
    
    # 测试不同标准化方法
    methods = ['robust', 'standard', 'minmax', 'quantile']
    results = {}
    
    for method in methods:
        print(f"\n--- 测试标准化方法: {method} ---")
        
        config = PreprocessingConfig(
            normalization_method=method,
            base_window_size=50,
            imbalance_strategy='weights',
            use_parallel=False  # 小数据集关闭并行
        )
        
        preprocessor = EnhancedDataPreprocessor(config=config, logger=logger)
        
        start_time = time.time()
        result = preprocessor.process_pipeline(data)
        processing_time = time.time() - start_time
        
        if 'error' not in result:
            train_shape = result['data']['X_train'].shape
            test_shape = result['data']['X_test'].shape
            val_shape = result['data']['X_val'].shape
            
            results[method] = {
                'processing_time': processing_time,
                'train_samples': train_shape[0],
                'test_samples': test_shape[0],
                'val_samples': val_shape[0],
                'window_size': train_shape[1] if len(train_shape) > 1 else 0,
                'features': train_shape[2] if len(train_shape) > 2 else 0
            }
            
            print(f"  处理时间: {processing_time:.3f}秒")
            print(f"  训练集: {train_shape}")
            print(f"  验证集: {val_shape}")
            print(f"  测试集: {test_shape}")
        else:
            print(f"  处理失败: {result['error']}")
            results[method] = {'error': result['error']}
    
    # 结果汇总
    print(f"\n{'方法':<12} {'时间(秒)':<10} {'训练样本':<10} {'窗口大小':<10} {'特征数':<8}")
    print("-" * 60)
    for method, stats in results.items():
        if 'error' not in stats:
            print(f"{method:<12} {stats['processing_time']:<10.3f} "
                  f"{stats['train_samples']:<10} {stats['window_size']:<10} {stats['features']:<8}")

def test_imbalance_handling():
    """测试类别不平衡处理"""
    print("\n" + "=" * 70)
    print("类别不平衡处理测试")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 生成不平衡数据
    data = generate_sample_trading_data(1000)
    
    # 人工创建真实的类别不平衡
    market_states = data['market_state'].copy()
    
    # 创建三个类别的不平衡分布
    total_samples = len(data)
    
    # 类别0: 60%, 类别1: 35%, 类别2: 5% (严重不平衡)
    n_class_0 = int(total_samples * 0.60)
    n_class_1 = int(total_samples * 0.35) 
    n_class_2 = total_samples - n_class_0 - n_class_1
    
    # 重新分配类别
    new_labels = ([0] * n_class_0 + [1] * n_class_1 + [2] * n_class_2)
    np.random.shuffle(new_labels)
    data['market_state'] = new_labels[:total_samples]
    
    print("原始类别分布:")
    print(data['market_state'].value_counts())
    print(f"不平衡比例: {data['market_state'].value_counts().min() / data['market_state'].value_counts().max():.3f}")
    
    # 测试不同不平衡处理策略
    strategies = ['weights', 'auto']
    if 'imblearn' in sys.modules or True:  # 假设可用
        strategies.extend(['smote', 'undersample'])
    
    for strategy in strategies:
        print(f"\n--- 策略: {strategy} ---")
        
        config = PreprocessingConfig(
            normalization_method='robust',
            base_window_size=40,
            imbalance_strategy=strategy,
            use_parallel=False
        )
        
        preprocessor = EnhancedDataPreprocessor(config=config, logger=logger)
        
        start_time = time.time()
        result = preprocessor.process_pipeline(data)
        processing_time = time.time() - start_time
        
        if 'error' not in result:
            imbalance_info = result['pipeline_results']['imbalance_handling']
            print(f"  处理时间: {processing_time:.3f}秒")
            print(f"  策略: {imbalance_info['strategy']}")
            if 'resampled_counts' in imbalance_info:
                print(f"  重采样后分布: {imbalance_info['resampled_counts']}")
            print(f"  类别权重: {imbalance_info['class_weights']}")
        else:
            print(f"  处理失败: {result['error']}")

def test_window_creation_strategies():
    """测试不同窗口创建策略"""
    print("\n" + "=" * 70)
    print("窗口创建策略测试")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 生成测试数据
    data = generate_sample_trading_data(800)
    
    # 测试不同窗口配置
    window_configs = [
        {'base_window_size': 30, 'min_window_size': 20, 'max_window_size': 60},
        {'base_window_size': 60, 'min_window_size': 30, 'max_window_size': 120},
        {'base_window_size': 90, 'min_window_size': 50, 'max_window_size': 150},
    ]
    
    for i, window_config in enumerate(window_configs):
        print(f"\n--- 窗口配置 {i+1}: {window_config} ---")
        
        config = PreprocessingConfig(
            normalization_method='robust',
            imbalance_strategy='weights',
            use_parallel=True,
            **window_config
        )
        
        preprocessor = EnhancedDataPreprocessor(config=config, logger=logger)
        
        start_time = time.time()
        result = preprocessor.process_pipeline(data)
        processing_time = time.time() - start_time
        
        if 'error' not in result:
            stats = result['pipeline_results']['processing_stats']
            data_shapes = result['pipeline_results']['data_splits']
            
            print(f"  处理时间: {processing_time:.3f}秒")
            print(f"  生成样本数: {stats['samples_processed']}")
            print(f"  窗口大小: {stats['window_size']}")
            print(f"  训练集形状: {data_shapes['train_shape']}")
            print(f"  内存效率: {stats['samples_processed'] / processing_time:.0f} 样本/秒")
        else:
            print(f"  处理失败: {result['error']}")

def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 70)
    print("性能基准测试")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 测试不同数据规模，包括大数据集测试并行效果
    data_sizes = [1000, 5000, 10000, 15000]
    
    for size in data_sizes:
        print(f"\n--- 数据规模: {size} 样本 ---")
        
        data = generate_sample_trading_data(size)
        
        # 标准配置
        config = PreprocessingConfig(
            normalization_method='robust',
            base_window_size=60,
            imbalance_strategy='auto',
            use_parallel=True,
            max_workers=4
        )
        
        preprocessor = EnhancedDataPreprocessor(config=config, logger=logger)
        
        # 测试并行处理
        start_time = time.time()
        result_parallel = preprocessor.process_pipeline(data)
        parallel_time = time.time() - start_time
        
        # 测试顺序处理
        config.use_parallel = False
        preprocessor_seq = EnhancedDataPreprocessor(config=config, logger=logger)
        
        start_time = time.time()
        result_sequential = preprocessor_seq.process_pipeline(data)
        sequential_time = time.time() - start_time
        
        if 'error' not in result_parallel and 'error' not in result_sequential:
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1
            
            print(f"  并行处理: {parallel_time:.3f}秒")
            print(f"  顺序处理: {sequential_time:.3f}秒")
            print(f"  加速比: {speedup:.2f}x")
            
            # 内存使用估算
            train_shape = result_parallel['data']['X_train'].shape
            memory_mb = np.prod(train_shape) * 4 / (1024 * 1024)  # float32
            print(f"  训练数据内存: {memory_mb:.1f} MB")
            print(f"  处理效率: {size / parallel_time:.0f} 样本/秒")
        else:
            print(f"  处理失败")

def test_real_imbalance_handling():
    """专门测试真实的类别不平衡处理效果"""
    print("\n" + "=" * 70)
    print("真实类别不平衡处理验证")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 生成有真实不平衡的数据
    data = generate_sample_trading_data(2000)
    
    # 创建严重不平衡的类别分布
    total_samples = len(data)
    n_class_0 = int(total_samples * 0.70)  # 70%
    n_class_1 = int(total_samples * 0.25)  # 25%  
    n_class_2 = total_samples - n_class_0 - n_class_1  # 5%
    
    new_labels = ([0] * n_class_0 + [1] * n_class_1 + [2] * n_class_2)
    np.random.shuffle(new_labels)
    data['market_state'] = new_labels[:total_samples]
    
    original_counts = pd.Series(data['market_state']).value_counts().sort_index()
    print(f"严重不平衡数据:")
    print(f"  类别分布: {original_counts.to_dict()}")
    print(f"  不平衡比例: {original_counts.min() / original_counts.max():.3f}")
    
    # 测试各种不平衡处理策略
    strategies = ['weights', 'smote', 'undersample']
    
    for strategy in strategies:
        print(f"\n--- 测试策略: {strategy} ---")
        
        config = PreprocessingConfig(
            normalization_method='robust',
            base_window_size=40,
            imbalance_strategy=strategy,
            use_parallel=False  # 专注测试不平衡处理
        )
        
        preprocessor = EnhancedDataPreprocessor(config=config, logger=logger)
        
        try:
            # 直接测试不平衡处理
            X, y, timestamps = preprocessor.create_advanced_windows(data)
            if X.size > 0:
                X_balanced, y_balanced, info = preprocessor.handle_class_imbalance_advanced(X, y)
                
                original_dist = pd.Series(y).value_counts().sort_index()
                balanced_dist = pd.Series(y_balanced).value_counts().sort_index()
                
                print(f"  原始分布: {original_dist.to_dict()}")
                print(f"  处理后分布: {balanced_dist.to_dict()}")
                print(f"  策略: {info['strategy']}")
                print(f"  样本数变化: {len(y)} -> {len(y_balanced)}")
                
                if 'resampled_counts' in info:
                    improvement = balanced_dist.min() / balanced_dist.max()
                    original_ratio = original_dist.min() / original_dist.max()
                    print(f"  平衡改善: {original_ratio:.3f} -> {improvement:.3f}")
            else:
                print(f"  窗口创建失败")
                
        except Exception as e:
            print(f"  策略测试失败: {str(e)}")

def demonstrate_complete_pipeline():
    """演示完整的预处理流水线"""
    print("\n" + "=" * 70)
    print("完整预处理流水线演示")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 生成较大的测试数据集
    data = generate_sample_trading_data(1500)
    print(f"原始数据: {data.shape}")
    print(f"时间范围: {data.index[0]} 到 {data.index[-1]}")
    print(f"特征列数: {len(data.columns)}")
    
    # 优化配置
    config = PreprocessingConfig(
        normalization_method='robust',
        base_window_size=60,
        min_window_size=30,
        max_window_size=120,
        stride=1,
        test_size=0.2,
        validation_size=0.1,
        imbalance_strategy='auto',
        use_parallel=True,
        max_workers=4,
        enable_caching=True
    )
    
    print(f"\n预处理配置:")
    print(f"  标准化方法: {config.normalization_method}")
    print(f"  基础窗口大小: {config.base_window_size}")
    print(f"  不平衡处理: {config.imbalance_strategy}")
    print(f"  并行处理: {config.use_parallel}")
    
    # 执行完整流水线
    preprocessor = EnhancedDataPreprocessor(config=config, logger=logger)
    
    start_time = time.time()
    result = preprocessor.process_pipeline(data)
    total_time = time.time() - start_time
    
    if 'error' not in result:
        # 显示结果
        data_info = result['pipeline_results']['data_splits']
        processing_stats = result['pipeline_results']['processing_stats']
        imbalance_info = result['pipeline_results']['imbalance_handling']
        
        print(f"\n流水线执行成功!")
        print(f"总处理时间: {total_time:.3f}秒")
        print(f"\n数据集拆分:")
        print(f"  训练集: {data_info['train_shape']}")
        print(f"  验证集: {data_info['val_shape']}")
        print(f"  测试集: {data_info['test_shape']}")
        
        print(f"\n类别不平衡处理:")
        print(f"  策略: {imbalance_info['strategy']}")
        print(f"  原始分布: {imbalance_info['original_counts']}")
        if 'resampled_counts' in imbalance_info:
            print(f"  重采样后: {imbalance_info['resampled_counts']}")
        
        print(f"\n性能统计:")
        print(f"  处理样本数: {processing_stats['samples_processed']}")
        print(f"  特征数量: {processing_stats['feature_count']}")
        print(f"  窗口大小: {processing_stats['window_size']}")
        
        # 获取性能统计
        perf_stats = preprocessor.get_performance_stats()
        if perf_stats.get('avg_processing_time'):
            print(f"  平均处理时间: {perf_stats['avg_processing_time']:.3f}秒")
        
        # 保存结果
        output_dir = "enhanced_preprocessed_data"
        saved_path = preprocessor.save_preprocessing_artifacts(output_dir, result)
        if saved_path:
            print(f"\n结果已保存至: {saved_path}")
        
        return result
    else:
        print(f"流水线执行失败: {result['error']}")
        return None

def main():
    """主函数"""
    print("Enhanced DataPreprocessor 演示")
    print("作者: AI Assistant")
    print("日期:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # 运行所有演示
        compare_preprocessing_methods()
        test_imbalance_handling()
        test_window_creation_strategies()
        test_real_imbalance_handling()  # 新增真实不平衡测试
        performance_benchmark()
        result = demonstrate_complete_pipeline()
        
        print("\n" + "=" * 70)
        print("演示完成！")
        print("=" * 70)
        
        if result:
            print("\n主要改进点:")
            print("✅ 多种标准化方法支持")
            print("✅ 智能动态窗口创建")
            print("✅ 高级类别不平衡处理")
            print("✅ 时间感知的数据集拆分")
            print("✅ 并行处理优化")
            print("✅ 完整的性能监控")
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
