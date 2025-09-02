"""
数据客户端集成模块
与配置系统和日志系统深度集成
"""

from unified_data_client import UnifiedDataClient, MarketData, DataType
from enhanced_config_loader import load_config
from logger_config_integration import get_strategy_logger
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
import asyncio
import threading
import time


class DataClientManager:
    """数据客户端管理器 - 与交易系统集成"""
    
    def __init__(self, environment: str = "development", strategy_name: str = "main"):
        """
        初始化数据客户端管理器
        
        Args:
            environment: 环境名称
            strategy_name: 策略名称
        """
        self.environment = environment
        self.strategy_name = strategy_name
        
        # 加载配置
        self.config = load_config(environment=environment)
        
        # 设置日志
        self.logger = get_strategy_logger(f"data_manager_{strategy_name}", environment)
        
        # 创建统一数据客户端
        self.client = UnifiedDataClient(config=self.config, logger=self.logger)
        
        # 数据缓存
        self.historical_cache: Dict[str, pd.DataFrame] = {}
        self.realtime_subscribers: Dict[str, List[Callable]] = {}
        
        # 配置参数
        self.default_timeframe = self.config.get('DATA_TIMEFRAME', '30min')
        self.history_years = self.config.get('HISTORY_YEARS', 5)
        self.ticker = self.config.get('TICKER', 'AAPL')
        
        self.logger.info(f"数据客户端管理器初始化完成 - 环境: {environment}, 策略: {strategy_name}")
    
    def get_strategy_data(self, 
                         symbol: str = None,
                         timeframe: str = None,
                         lookback_days: int = None) -> pd.DataFrame:
        """
        获取策略所需的历史数据
        
        Args:
            symbol: 股票代码，默认使用配置中的TICKER
            timeframe: 时间框架，默认使用配置中的DATA_TIMEFRAME
            lookback_days: 回看天数，默认根据配置计算
            
        Returns:
            DataFrame: 历史数据
        """
        symbol = symbol or self.ticker
        timeframe = timeframe or self.default_timeframe
        
        if lookback_days is None:
            # 根据配置计算回看天数
            train_days = self.config.get('TRAIN_DAYS', 1260)
            lookback_days = min(train_days, self.history_years * 365)
        
        cache_key = f"{symbol}_{timeframe}_{lookback_days}"
        
        # 检查缓存
        if cache_key in self.historical_cache:
            cached_data = self.historical_cache[cache_key]
            # 检查缓存是否过期（1小时）
            if (datetime.now() - cached_data.index[-1]).total_seconds() < 3600:
                self.logger.debug(f"使用缓存数据: {cache_key}")
                return cached_data
        
        # 获取新数据
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"获取策略数据: {symbol}, {timeframe}, {start_date} to {end_date}")
        
        df = self.client.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            # 缓存数据
            self.historical_cache[cache_key] = df
            
            # 记录数据统计
            self.logger.info(f"获取到 {len(df)} 条数据", extra={
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'price_range': {
                    'high': float(df['high'].max()),
                    'low': float(df['low'].min()),
                    'latest': float(df['close'].iloc[-1])
                }
            })
        
        return df
    
    def setup_realtime_monitoring(self, 
                                 symbols: List[str] = None,
                                 callback: Callable[[MarketData], None] = None) -> bool:
        """
        设置实时数据监控
        
        Args:
            symbols: 要监控的股票代码列表，默认使用配置中的TICKER
            callback: 数据回调函数
            
        Returns:
            bool: 设置是否成功
        """
        symbols = symbols or [self.ticker]
        
        if callback is None:
            callback = self._default_realtime_callback
        
        self.logger.info(f"设置实时数据监控: {symbols}")
        
        success_count = 0
        for symbol in symbols:
            try:
                success = self.client.subscribe_realtime(symbol, callback)
                if success:
                    success_count += 1
                    # 记录订阅
                    if symbol not in self.realtime_subscribers:
                        self.realtime_subscribers[symbol] = []
                    self.realtime_subscribers[symbol].append(callback)
                    
                    self.logger.info(f"实时数据监控设置成功: {symbol}")
                else:
                    self.logger.warning(f"实时数据监控设置失败: {symbol}")
                    
            except Exception as e:
                self.logger.error(f"设置实时监控失败 {symbol}: {e}")
        
        return success_count == len(symbols)
    
    def _default_realtime_callback(self, data: MarketData):
        """默认实时数据回调函数"""
        self.logger.log_market_data(
            ticker=data.symbol,
            price=data.price or 0,
            data_type="realtime_callback"
        )
        
        # 可以在这里添加其他实时数据处理逻辑
        # 例如：触发交易信号、更新监控面板等
    
    def get_current_price(self, symbol: str = None) -> Optional[float]:
        """
        获取当前价格
        
        Args:
            symbol: 股票代码
            
        Returns:
            Optional[float]: 当前价格
        """
        symbol = symbol or self.ticker
        price = self.client.get_latest_price(symbol)
        
        if price:
            self.logger.debug(f"当前价格 {symbol}: ${price:.2f}")
        else:
            self.logger.warning(f"无法获取当前价格: {symbol}")
        
        return price
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            df: 数据DataFrame
            symbol: 股票代码
            
        Returns:
            Dict: 数据质量报告
        """
        if df.empty:
            return {"valid": False, "error": "数据为空"}
        
        quality_report = {
            "valid": True,
            "symbol": symbol,
            "total_records": len(df),
            "date_range": {
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat()
            },
            "missing_data": {
                "count": df.isnull().sum().sum(),
                "percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            "price_statistics": {
                "mean": float(df['close'].mean()),
                "std": float(df['close'].std()),
                "min": float(df['close'].min()),
                "max": float(df['close'].max())
            },
            "data_gaps": self._detect_data_gaps(df),
            "outliers": self._detect_outliers(df)
        }
        
        # 记录数据质量
        self.logger.info(f"数据质量检查: {symbol}", extra=quality_report)
        
        return quality_report
    
    def _detect_data_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检测数据缺口"""
        try:
            expected_freq = pd.infer_freq(df.index)
            if expected_freq:
                full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
                missing_dates = full_range.difference(df.index)
                return {
                    "expected_frequency": expected_freq,
                    "missing_count": len(missing_dates),
                    "missing_percentage": (len(missing_dates) / len(full_range)) * 100
                }
        except Exception:
            pass
        
        return {"detected": False, "reason": "无法检测频率"}
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检测价格异常值"""
        try:
            # 使用IQR方法检测异常值
            Q1 = df['close'].quantile(0.25)
            Q3 = df['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df['close'] < lower_bound) | (df['close'] > upper_bound)]
            
            return {
                "count": len(outliers),
                "percentage": (len(outliers) / len(df)) * 100,
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound)
                }
            }
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    def refresh_data(self, symbol: str = None, force: bool = False):
        """
        刷新数据缓存
        
        Args:
            symbol: 股票代码
            force: 是否强制刷新
        """
        symbol = symbol or self.ticker
        
        if force:
            # 清理缓存
            keys_to_remove = [key for key in self.historical_cache.keys() if key.startswith(symbol)]
            for key in keys_to_remove:
                del self.historical_cache[key]
            
            # 清理客户端缓存
            self.client.clear_cache(symbol)
            
            self.logger.info(f"强制刷新数据缓存: {symbol}")
        
        # 重新获取数据
        self.get_strategy_data(symbol)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取数据客户端性能统计"""
        client_status = self.client.get_status()
        
        stats = {
            "client_status": client_status,
            "cached_datasets": len(self.historical_cache),
            "realtime_subscriptions": len(self.realtime_subscribers),
            "environment": self.environment,
            "strategy": self.strategy_name,
            "config": {
                "ticker": self.ticker,
                "timeframe": self.default_timeframe,
                "history_years": self.history_years
            }
        }
        
        return stats
    
    def close(self):
        """关闭数据客户端管理器"""
        self.logger.info("关闭数据客户端管理器")
        
        # 清理缓存
        self.historical_cache.clear()
        self.realtime_subscribers.clear()
        
        # 关闭客户端
        self.client.close()


# ==================== 便捷函数 ====================

def setup_trading_data(environment: str = "development", 
                      strategy_name: str = "main") -> DataClientManager:
    """设置交易数据客户端的便捷函数"""
    return DataClientManager(environment, strategy_name)


def get_strategy_historical_data(symbol: str = None,
                                environment: str = "development") -> pd.DataFrame:
    """获取策略历史数据的便捷函数"""
    manager = setup_trading_data(environment)
    return manager.get_strategy_data(symbol)


if __name__ == "__main__":
    # 演示用法
    print("=== 数据客户端集成演示 ===")
    
    # 创建数据管理器
    manager = setup_trading_data(environment="development", strategy_name="demo")
    
    # 获取策略数据
    print("1. 获取策略历史数据...")
    df = manager.get_strategy_data("AAPL")
    if not df.empty:
        print(f"获取到 {len(df)} 条数据")
        print(f"价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # 验证数据质量
    print("\n2. 验证数据质量...")
    quality = manager.validate_data_quality(df, "AAPL")
    print(f"数据质量: {'✅ 良好' if quality['valid'] else '❌ 异常'}")
    print(f"缺失数据: {quality['missing_data']['percentage']:.1f}%")
    
    # 获取当前价格
    print("\n3. 获取当前价格...")
    current_price = manager.get_current_price("AAPL")
    if current_price:
        print(f"AAPL 当前价格: ${current_price:.2f}")
    
    # 设置实时监控
    print("\n4. 设置实时监控...")
    def demo_callback(data):
        print(f"实时更新: {data.symbol} = ${data.price:.2f} at {data.timestamp}")
    
    success = manager.setup_realtime_monitoring(["AAPL"], demo_callback)
    print(f"实时监控: {'✅ 成功' if success else '❌ 失败'}")
    
    # 获取性能统计
    print("\n5. 性能统计...")
    stats = manager.get_performance_stats()
    print(f"缓存数据集: {stats['cached_datasets']}")
    print(f"实时订阅: {stats['realtime_subscriptions']}")
    print(f"客户端状态: {stats['client_status']['websocket_status']}")
    
    # 关闭管理器
    manager.close()
    print("\n演示完成！")


