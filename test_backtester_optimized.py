#!/usr/bin/env python3
"""
测试优化后的Backtester - 简化版本验证
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scripts.backtester import Backtester
from scripts.logger_setup import setup_logging
import json

def create_mock_data():
    """创建模拟数据用于测试"""
    logger = setup_logging()
    logger.info("创建模拟测试数据...")
    
    # 创建30天的30分钟数据
    dates = pd.date_range(
        start='2024-01-01', 
        end='2024-01-30', 
        freq='30min'
    )
    
    # 模拟价格数据
    np.random.seed(42)
    n_periods = len(dates)
    
    # 生成随机游走价格
    returns = np.random.normal(0.0001, 0.02, n_periods)
    prices = 100 * np.exp(returns.cumsum())
    
    # 创建特征数据
    data = {
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
        'volume': np.random.lognormal(15, 0.5, n_periods),
        'volatility': np.random.uniform(0.01, 0.06, n_periods),
        'rsi': np.random.uniform(20, 80, n_periods),
        'atr': prices * np.random.uniform(0.005, 0.02, n_periods),
        'returns_1': returns,
        'returns_3': np.random.normal(0, 0.015, n_periods),
        'returns_6': np.random.normal(0, 0.02, n_periods),
        'macd_hist': np.random.normal(0, 0.5, n_periods),
        'volume_osc': np.random.normal(0, 1, n_periods),
        'market_state': np.random.choice([0, 1, 2], n_periods, p=[0.3, 0.4, 0.3])
    }
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'datetime'
    
    # 确保high >= close >= low
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    logger.info(f"创建了 {len(df)} 行测试数据")
    return df

def create_mock_model():
    """创建模拟模型管道"""
    class MockModel:
        def predict_proba(self, X):
            """模拟预测概率"""
            n_samples = X.shape[0]
            # 生成随机概率，倾向于中性状态
            probs = np.random.dirichlet([0.25, 0.5, 0.25], n_samples)
            return probs
            
        def predict(self, X):
            """模拟预测类别"""
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)
    
    return {
        'model': MockModel(),
        'scaler': None,
        'feature_selector': ['close', 'volume', 'volatility', 'rsi', 'atr', 'returns_1']
    }

def create_mock_config():
    """创建模拟配置"""
    return {
        'window_size': 20,  # 减少窗口大小以便测试
        'stop_method': 'atr',
        'atr_multiplier': 2,
        'risk_threshold': 0.08
    }

class MockBacktester(Backtester):
    """模拟版本的Backtester，用于测试"""
    
    def __init__(self, test_data, test_model, test_config):
        self.logger = setup_logging()
        self.pipeline = test_model
        self.model = self.pipeline['model']
        self.scaler = self.pipeline.get('scaler', None)
        self.selected_features = self.pipeline.get('feature_selector', [])
        self.config = test_config
        self.test_data = test_data
        self.window_size = self.config.get('window_size', 20)
        self.stop_method = self.config.get('stop_method', 'fixed')
        self.atr_multiplier = self.config.get('atr_multiplier', 2)
        self.last_trade_time = None
        self.min_trade_interval = 30
        self.max_drawdown = 0.2
        self.atr_cache = {}
        self.current_position = 0
        
        # 交易参数
        self.MAX_DAILY_TRADES = 3
        self.HIGH_VOL_LIMIT = 1
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # 成本模型
        self.SLIPPAGE_MODEL = {
            'normal': {'mean': 0.0003, 'std': 0.0001},
            'high_vol': {'mean': 0.0012, 'std': 0.0003}
        }
        
        self.total_slippage_cost = 0
        self.total_commission_cost = 0
        self.commission_rate = 0.0005
        self.risk_threshold = self.config.get('risk_threshold', 0.08)
    
    def load_and_check_features(self):
        """返回测试数据"""
        return self.test_data.copy()

def test_backtester():
    """测试优化后的Backtester"""
    logger = setup_logging()
    logger.info("开始测试优化后的Backtester...")
    
    try:
        # 创建测试数据
        test_data = create_mock_data()
        test_model = create_mock_model()
        test_config = create_mock_config()
        
        # 创建测试实例
        backtester = MockBacktester(test_data, test_model, test_config)
        
        # 运行回测
        logger.info("运行回测...")
        report = backtester.run_backtest()
        
        # 验证结果格式
        logger.info("验证结果格式...")
        expected_keys = ['annual_return', 'max_drawdown', 'win_rate', 'sharpe_ratio']
        
        for key in expected_keys:
            if key not in report:
                logger.error(f"缺少关键指标: {key}")
                return False
            
            value = report[key]
            if not isinstance(value, (int, float)):
                logger.error(f"指标 {key} 不是数值类型: {type(value)}")
                return False
                
            logger.info(f"{key}: {value}")
        
        # 显示完整报告
        logger.info("=" * 50)
        logger.info("回测结果:")
        logger.info("=" * 50)
        logger.info(f"年化收益率: {report['annual_return']:.2%}")
        logger.info(f"最大回撤: {report['max_drawdown']:.2%}")
        logger.info(f"胜率: {report['win_rate']:.2%}")
        logger.info(f"夏普比率: {report['sharpe_ratio']:.2f}")
        
        if 'details' in report:
            details = report['details']
            logger.info(f"总交易次数: {details.get('total_trades', 0)}")
            logger.info(f"最终价值: ${details.get('final_value', 0):.2f}")
        
        logger.info("=" * 50)
        logger.info("测试成功! Backtester已优化完成")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_backtester()
    sys.exit(0 if success else 1)
