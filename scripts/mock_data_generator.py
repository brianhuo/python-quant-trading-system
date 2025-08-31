"""
模拟数据生成器
在没有真实API的情况下提供测试数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
import threading
import time
import random
from unified_data_client import MarketData, DataType
from data_client_integration import DataClientManager


class MockDataGenerator:
    """模拟数据生成器 - 替代真实API进行开发测试"""
    
    def __init__(self, base_price: float = 150.0, volatility: float = 0.02):
        """
        初始化模拟数据生成器
        
        Args:
            base_price: 基础价格
            volatility: 波动率
        """
        self.base_price = base_price
        self.volatility = volatility
        self.current_price = base_price
        self.is_running = False
        self.callbacks = []
        
    def generate_historical_data(self, 
                                symbol: str = "AAPL",
                                timeframe: str = "30min", 
                                start_date: str = None,
                                end_date: str = None,
                                records: int = 1000) -> pd.DataFrame:
        """
        生成模拟历史数据
        
        Args:
            symbol: 股票代码
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            records: 记录数量
            
        Returns:
            DataFrame: 模拟历史数据
        """
        print(f"🎭 生成模拟历史数据: {symbol} {timeframe} ({records}条)")
        
        # 计算时间间隔
        freq_map = {
            "1min": "1T",
            "5min": "5T", 
            "15min": "15T",
            "30min": "30T",
            "1h": "1H",
            "1day": "1D"
        }
        freq = freq_map.get(timeframe, "30T")
        
        # 生成时间序列
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            dates = pd.date_range(start=start, end=end, freq=freq)
        else:
            # 从当前时间向前推算
            end_time = datetime.now()
            start_time = end_time - timedelta(days=records//48 if timeframe=="30min" else records)
            dates = pd.date_range(start=start_time, end=end_time, freq=freq.replace('T', 'min'))
        
        # 限制记录数量
        if len(dates) > records:
            dates = dates[-records:]
        
        # 生成价格数据（随机游走）
        price_changes = np.random.normal(0, self.volatility, len(dates))
        prices = [self.base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            # 添加一些边界限制
            new_price = max(new_price, self.base_price * 0.5)  # 不低于50%
            new_price = min(new_price, self.base_price * 2.0)  # 不高于200%
            prices.append(new_price)
        
        # 生成OHLCV数据
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # 生成当期的OHLC
            daily_volatility = self.volatility * 0.5
            high = price * (1 + random.uniform(0, daily_volatility))
            low = price * (1 - random.uniform(0, daily_volatility))
            
            # 确保逻辑正确：low <= open/close <= high
            open_price = price
            close_price = prices[i] if i < len(prices) else price
            
            # 调整确保 low <= open,close <= high
            low = min(low, open_price, close_price)
            high = max(high, open_price, close_price)
            
            volume = random.randint(100000, 5000000)
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high, 2), 
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'datetime'
        
        print(f"✅ 生成了 {len(df)} 条模拟数据")
        print(f"   价格范围: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"   最新价格: ${df['close'].iloc[-1]:.2f}")
        
        return df
    
    def start_realtime_simulation(self, 
                                 symbol: str = "AAPL",
                                 callback: Callable[[MarketData], None] = None,
                                 update_interval: float = 1.0):
        """
        开始实时数据模拟
        
        Args:
            symbol: 股票代码
            callback: 数据回调函数
            update_interval: 更新间隔（秒）
        """
        if callback:
            self.callbacks.append(callback)
        
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self._simulate_realtime, 
                           args=(symbol, update_interval), 
                           daemon=True).start()
            print(f"🎭 开始实时数据模拟: {symbol}")
    
    def _simulate_realtime(self, symbol: str, interval: float):
        """实时数据模拟线程"""
        while self.is_running:
            # 生成价格变化
            change = random.gauss(0, self.volatility * 0.1)
            self.current_price *= (1 + change)
            
            # 边界限制
            self.current_price = max(self.current_price, self.base_price * 0.8)
            self.current_price = min(self.current_price, self.base_price * 1.2)
            
            # 创建市场数据
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=round(self.current_price, 2),
                data_type=DataType.REALTIME,
                source="mock_generator"
            )
            
            # 调用回调函数
            for callback in self.callbacks:
                try:
                    callback(market_data)
                except Exception as e:
                    print(f"回调函数错误: {e}")
            
            time.sleep(interval)
    
    def stop_realtime_simulation(self):
        """停止实时数据模拟"""
        self.is_running = False
        self.callbacks.clear()
        print("🛑 实时数据模拟已停止")
    
    def get_latest_price(self, symbol: str = "AAPL") -> float:
        """获取最新价格"""
        return round(self.current_price, 2)


class MockDataClient:
    """模拟数据客户端 - 替代UnifiedDataClient进行测试"""
    
    def __init__(self):
        self.generator = MockDataGenerator()
        print("🎭 模拟数据客户端初始化完成")
    
    def get_historical_data(self, 
                          symbol: str,
                          timeframe: str = "30min",
                          start_date: str = None,
                          end_date: str = None,
                          limit: int = 1000,
                          **kwargs) -> pd.DataFrame:
        """获取历史数据（模拟）"""
        return self.generator.generate_historical_data(
            symbol=symbol,
            timeframe=timeframe, 
            start_date=start_date,
            end_date=end_date,
            records=limit
        )
    
    def subscribe_realtime(self, 
                          symbol: str,
                          callback: Callable[[MarketData], None],
                          **kwargs) -> bool:
        """订阅实时数据（模拟）"""
        self.generator.start_realtime_simulation(symbol, callback)
        return True
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格（模拟）"""
        return self.generator.get_latest_price(symbol)
    
    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        return {
            "websocket_status": "connected" if self.generator.is_running else "disconnected",
            "subscribed_symbols": ["AAPL"] if self.generator.is_running else [],
            "latest_data_count": 1 if self.generator.is_running else 0,
            "cache_enabled": True,
            "api_key_configured": True,
            "data_source": "mock_generator"
        }
    
    def close(self):
        """关闭客户端"""
        self.generator.stop_realtime_simulation()
        print("🎭 模拟数据客户端已关闭")


def create_mock_trading_environment():
    """创建完整的模拟交易环境"""
    print("🎭 创建模拟交易环境")
    
    # 创建模拟数据客户端
    mock_client = MockDataClient()
    
    # 获取历史数据演示
    print("\n📊 获取模拟历史数据...")
    df = mock_client.get_historical_data("AAPL", "30min", limit=100)
    print(f"历史数据: {len(df)} 条记录")
    print(df.tail())
    
    # 最新价格演示  
    print(f"\n💰 最新价格: ${mock_client.get_latest_price('AAPL')}")
    
    # 实时数据演示
    print("\n📡 实时数据模拟...")
    def price_callback(data: MarketData):
        print(f"  实时更新: {data.symbol} = ${data.price} at {data.timestamp.strftime('%H:%M:%S')}")
    
    mock_client.subscribe_realtime("AAPL", price_callback)
    
    # 运行5秒查看实时数据
    print("运行5秒实时数据...")
    time.sleep(5)
    
    # 状态检查
    print(f"\n📈 客户端状态: {mock_client.get_status()}")
    
    # 关闭
    mock_client.close()
    print("\n✅ 模拟交易环境演示完成")


if __name__ == "__main__":
    print("🎭 模拟数据生成器演示")
    print("=" * 50)
    
    # 演示基础功能
    generator = MockDataGenerator(base_price=150.0, volatility=0.02)
    
    # 生成历史数据
    df = generator.generate_historical_data("AAPL", "30min", records=50)
    print(f"\n📊 历史数据样本:\n{df.head()}")
    
    # 完整环境演示
    print("\n" + "=" * 50)
    create_mock_trading_environment()
    
    print("\n🎯 模拟数据生成器的优势:")
    print("✅ 无API成本 - 完全免费")
    print("✅ 可控数据 - 自定义参数") 
    print("✅ 快速测试 - 即时响应")
    print("✅ 离线开发 - 无网络依赖")
    print("✅ 一致接口 - 无缝切换")

