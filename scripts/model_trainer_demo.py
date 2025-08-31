#!/usr/bin/env python3
"""
Enhanced ModelTrainer V2 演示脚本

展示优化后的模型训练器的各项功能和性能提升
"""

import numpy as np
import pandas as pd
import time
import sys
import os
from pathlib import Path

# 添加脚本目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_model_trainer_v2 import (
    EnhancedModelTrainer, ModelConfig, ModelVersionManager,
    FeatureSelector, ModelEvaluator
)
from enhanced_data_preprocessor import EnhancedDataPreprocessor, PreprocessingConfig
from enhanced_feature_engineer import EnhancedFeatureEngineer
from logger_setup import setup_logging

def generate_sample_data(n_samples: int = 2000) -> tuple:
    """生成模拟的训练数据"""
    np.random.seed(42)
    
    # 生成特征数据 (样本数, 特征数)
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    
    # 添加一些有意义的模式
    # 特征 0-2: 价格相关
    X[:, 0] = np.cumsum(np.random.randn(n_samples) * 0.01)  # 价格走势
    X[:, 1] = np.diff(np.concatenate([[0], X[:, 0]]))        # 价格变化
    X[:, 2] = np.abs(X[:, 1])                               # 波动率
    
    # 特征 3-5: 技术指标
    X[:, 3] = np.convolve(X[:, 0], np.ones(5)/5, mode='same')  # 移动平均
    X[:, 4] = np.random.randn(n_samples) * 0.5                 # RSI模拟
    X[:, 5] = np.random.randn(n_samples) * 0.3                 # MACD模拟
    
    # 生成目标变量 (0: 下跌, 1: 横盘, 2: 上涨)
    # 基于价格变化生成标签
    price_change = X[:, 1]
    y = np.zeros(n_samples, dtype=int)
    y[price_change < -0.01] = 0  # 下跌
    y[price_change > 0.01] = 2   # 上涨
    y[(price_change >= -0.01) & (price_change <= 0.01)] = 1  # 横盘
    
    # 创建特征名称
    feature_names = [
        'price_level', 'price_change', 'volatility', 'ma_5', 'rsi', 'macd',
        'volume', 'atr', 'bb_width', 'momentum', 
        'support', 'resistance', 'trend', 'oscillator',
        'volume_ma', 'price_ma', 'volatility_ma', 'trend_strength',
        'market_regime', 'liquidity'
    ]
    
    return X, y, feature_names

def test_multi_model_training():
    """测试多模型支持"""
    print("=" * 70)
    print("多模型训练测试")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 生成测试数据
    X, y, feature_names = generate_sample_data(1000)
    
    # 数据拆分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"训练数据: {X_train.shape}, 测试数据: {X_test.shape}")
    print(f"类别分布: {pd.Series(y_train).value_counts().to_dict()}")
    
    # 测试不同模型
    model_types = ['lightgbm', 'xgboost', 'random_forest']
    results = {}
    
    for model_type in model_types:
        print(f"\n--- 测试模型: {model_type} ---")
        
        try:
            config = ModelConfig(
                model_type=model_type,
                feature_selection_method='auto',
                n_features=10,
                use_optuna=False,  # 关闭超参优化以加快演示
                cv_folds=3
            )
            
            trainer = EnhancedModelTrainer(config=config, logger=logger)
            
            start_time = time.time()
            result = trainer.train(X_train, y_train, X_test, y_test, feature_names)
            training_time = time.time() - start_time
            
            if 'error' not in result and 'evaluation_results' in result:
                evaluation_results = result['evaluation_results']
                if 'metrics' in evaluation_results:
                    metrics = evaluation_results['metrics']
                    results[model_type] = {
                        'training_time': training_time,
                        'f1_score': metrics.get('f1_score', 0.0),
                        'accuracy': metrics.get('accuracy', 0.0),
                        'selected_features': len(result.get('selected_features', [])),
                        'version_id': result.get('version_id', 'unknown')
                    }
                else:
                    print(f"  评估结果缺少metrics字段")
                    results[model_type] = {'error': '评估结果不完整'}
            elif 'error' in result:
                print(f"  训练失败: {result['error']}")
                results[model_type] = {'error': result['error']}
            else:
                print(f"  结果格式异常: {list(result.keys())}")
                results[model_type] = {'error': '结果格式异常'}
            
            # 只在成功时显示详细信息
            if model_type in results and 'error' not in results[model_type]:
                stats = results[model_type]
                print(f"  训练时间: {stats['training_time']:.2f}秒")
                print(f"  F1分数: {stats['f1_score']:.4f}")
                print(f"  准确率: {stats['accuracy']:.4f}")
                print(f"  选择特征数: {stats['selected_features']}")
            else:
                print(f"  训练失败: {result['error']}")
                results[model_type] = {'error': result['error']}
                
        except Exception as e:
            print(f"  测试失败: {str(e)}")
            results[model_type] = {'error': str(e)}
    
    # 结果汇总
    print(f"\n{'模型':<15} {'F1分数':<10} {'准确率':<10} {'时间(秒)':<10} {'特征数':<8}")
    print("-" * 70)
    for model_type, stats in results.items():
        if 'error' not in stats:
            print(f"{model_type:<15} {stats['f1_score']:<10.4f} "
                  f"{stats['accuracy']:<10.4f} {stats['training_time']:<10.2f} {stats['selected_features']:<8}")
    
    return results

def test_feature_selection_methods():
    """测试特征选择方法"""
    print("\n" + "=" * 70)
    print("特征选择方法测试")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 生成测试数据
    X, y, feature_names = generate_sample_data(800)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 测试不同特征选择方法
    selection_methods = ['auto', 'importance', 'statistical', 'mutual_info']
    results = {}
    
    for method in selection_methods:
        print(f"\n--- 特征选择方法: {method} ---")
        
        try:
            config = ModelConfig(
                model_type='lightgbm',
                feature_selection_method=method,
                n_features=8,
                use_optuna=False
            )
            
            trainer = EnhancedModelTrainer(config=config, logger=logger)
            
            start_time = time.time()
            result = trainer.train(X_train, y_train, X_test, y_test, feature_names)
            training_time = time.time() - start_time
            
            if 'error' not in result:
                metrics = result['evaluation_results']['metrics']
                selected_features = result['selected_features']
                
                results[method] = {
                    'training_time': training_time,
                    'f1_score': metrics['f1_score'],
                    'accuracy': metrics['accuracy'],
                    'selected_features': selected_features
                }
                
                print(f"  训练时间: {training_time:.2f}秒")
                print(f"  F1分数: {metrics['f1_score']:.4f}")
                print(f"  选择的特征: {selected_features[:5]}...")  # 显示前5个
            else:
                print(f"  方法失败: {result['error']}")
                
        except Exception as e:
            print(f"  测试失败: {str(e)}")
    
    # 分析特征选择差异
    if len(results) > 1:
        print(f"\n特征选择对比:")
        for method, stats in results.items():
            if 'selected_features' in stats:
                print(f"{method}: {stats['selected_features']}")

def test_model_version_management():
    """测试模型版本管理"""
    print("\n" + "=" * 70)
    print("模型版本管理测试")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 创建版本管理器
    version_manager = ModelVersionManager()
    
    # 生成测试数据
    X, y, feature_names = generate_sample_data(600)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 训练多个版本的模型
    versions = []
    configs = [
        {'model_type': 'lightgbm', 'n_features': 8},
        {'model_type': 'lightgbm', 'n_features': 12},
        {'model_type': 'random_forest', 'n_features': 10}
    ]
    
    for i, config_dict in enumerate(configs):
        print(f"\n--- 训练模型版本 {i+1}: {config_dict} ---")
        
        try:
            config = ModelConfig(**config_dict, use_optuna=False)
            trainer = EnhancedModelTrainer(config=config, logger=logger)
            
            result = trainer.train(X_train, y_train, X_test, y_test, feature_names)
            
            if 'error' not in result:
                version_id = result['version_id']
                versions.append(version_id)
                
                metrics = result['evaluation_results']['metrics']
                print(f"  版本ID: {version_id}")
                print(f"  F1分数: {metrics['f1_score']:.4f}")
            else:
                print(f"  训练失败: {result['error']}")
                
        except Exception as e:
            print(f"  版本创建失败: {str(e)}")
    
    # 查看版本信息
    print(f"\n--- 版本管理功能演示 ---")
    
    # 列出所有模型
    all_models = version_manager.list_models()
    print(f"总共有 {len(all_models)} 个模型版本")
    
    # 显示最新的几个版本
    for model in all_models[-3:]:
        print(f"模型: {model['model_type']}, "
              f"F1: {model['metrics'].get('f1_score', 0):.4f}, "
              f"时间: {model['created_at']}")
    
    # 获取最佳模型
    best_version = version_manager.get_best_model('f1_score')
    if best_version:
        best_info = version_manager.get_model_info(best_version)
        print(f"\n最佳模型 (F1分数): {best_version}")
        print(f"  类型: {best_info['model_type']}")
        print(f"  F1分数: {best_info['metrics']['f1_score']:.4f}")

def test_hyperparameter_optimization():
    """测试超参数优化"""
    print("\n" + "=" * 70)
    print("超参数优化测试")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 生成测试数据
    X, y, feature_names = generate_sample_data(800)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 对比没有优化和有优化的情况
    configs = [
        {'use_optuna': False, 'name': '默认参数'},
        {'use_optuna': True, 'optuna_trials': 20, 'name': 'Optuna优化(20次)'}
    ]
    
    results = {}
    
    for config_dict in configs:
        name = config_dict.pop('name')
        print(f"\n--- 测试: {name} ---")
        
        try:
            config = ModelConfig(
                model_type='lightgbm',
                feature_selection_method='auto',
                n_features=10,
                **config_dict
            )
            
            trainer = EnhancedModelTrainer(config=config, logger=logger)
            
            start_time = time.time()
            result = trainer.train(X_train, y_train, X_test, y_test, feature_names)
            training_time = time.time() - start_time
            
            if 'error' not in result:
                metrics = result['evaluation_results']['metrics']
                training_stats = result['training_stats']
                
                results[name] = {
                    'training_time': training_time,
                    'f1_score': metrics['f1_score'],
                    'accuracy': metrics['accuracy'],
                    'n_trials': training_stats.get('n_trials', 0)
                }
                
                print(f"  训练时间: {training_time:.2f}秒")
                print(f"  F1分数: {metrics['f1_score']:.4f}")
                print(f"  准确率: {metrics['accuracy']:.4f}")
                if training_stats.get('n_trials', 0) > 0:
                    print(f"  优化试验次数: {training_stats['n_trials']}")
            else:
                print(f"  训练失败: {result['error']}")
                
        except Exception as e:
            print(f"  测试失败: {str(e)}")
    
    # 对比结果
    if len(results) == 2:
        print(f"\n--- 优化效果对比 ---")
        default_f1 = list(results.values())[0]['f1_score']
        optimized_f1 = list(results.values())[1]['f1_score']
        improvement = ((optimized_f1 - default_f1) / default_f1) * 100
        print(f"F1分数改善: {improvement:.2f}%")

def demonstrate_complete_pipeline():
    """演示完整的训练流水线"""
    print("\n" + "=" * 70)
    print("完整训练流水线演示")
    print("=" * 70)
    
    logger = setup_logging()
    
    # 生成较大的数据集
    X, y, feature_names = generate_sample_data(1500)
    
    print(f"原始数据: {X.shape}")
    print(f"特征名称: {feature_names[:5]}...")
    print(f"目标分布: {pd.Series(y).value_counts().to_dict()}")
    
    # 数据拆分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 最佳配置
    config = ModelConfig(
        model_type='lightgbm',
        feature_selection_method='auto',
        n_features=15,
        use_optuna=True,
        optuna_trials=30,
        cv_folds=5
    )
    
    print(f"\n训练配置:")
    print(f"  模型类型: {config.model_type}")
    print(f"  特征选择: {config.feature_selection_method}")
    print(f"  目标特征数: {config.n_features}")
    print(f"  超参优化: {config.use_optuna} ({config.optuna_trials}次试验)")
    
    # 创建训练器
    trainer = EnhancedModelTrainer(config=config, logger=logger)
    
    # 执行训练
    start_time = time.time()
    result = trainer.train(X_train, y_train, X_test, y_test, feature_names)
    total_time = time.time() - start_time
    
    if 'error' not in result:
        # 显示详细结果
        evaluation_results = result['evaluation_results']
        training_stats = result['training_stats']
        
        print(f"\n--- 训练完成 ---")
        print(f"总训练时间: {total_time:.2f}秒")
        print(f"版本ID: {result['version_id']}")
        
        print(f"\n--- 模型性能 ---")
        metrics = evaluation_results['metrics']
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\n--- 特征选择结果 ---")
        selected_features = result['selected_features']
        print(f"  选择特征数: {len(selected_features)}")
        print(f"  选择的特征: {selected_features[:8]}...")
        
        print(f"\n--- 训练统计 ---")
        print(f"  超参优化试验: {training_stats.get('n_trials', 0)}")
        print(f"  最佳F1分数: {training_stats.get('best_score', 0):.4f}")
        
        # 获取训练摘要
        summary = trainer.get_training_summary()
        print(f"\n--- 训练摘要 ---")
        print(f"  模型已拟合: {summary['is_fitted']}")
        print(f"  配置: {summary['config']['model_type']}")
        
        return result
    else:
        print(f"训练失败: {result['error']}")
        return None

def main():
    """主函数"""
    print("Enhanced ModelTrainer V2 演示")
    print("作者: AI Assistant")
    print("日期:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # 运行所有演示
        model_results = test_multi_model_training()
        test_feature_selection_methods()
        test_model_version_management()
        test_hyperparameter_optimization()
        complete_result = demonstrate_complete_pipeline()
        
        print("\n" + "=" * 70)
        print("演示完成！")
        print("=" * 70)
        
        if complete_result:
            print("\n主要优化成果:")
            print("✅ 多模型支持 (LightGBM, XGBoost, RandomForest)")
            print("✅ 智能特征选择 (auto, importance, statistical, mutual_info)")  
            print("✅ 完整版本管理 (自动版本控制, 最佳模型追踪)")
            print("✅ 超参数优化 (Optuna自动调参)")
            print("✅ 全面性能评估 (多指标, 详细报告)")
            print("✅ 生产就绪流水线 (错误处理, 状态追踪)")
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
