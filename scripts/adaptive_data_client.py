"""
自适应数据客户端
根据配置自动选择真实API或模拟数据
"""

from unified_data_client import UnifiedDataClient
from mock_data_generator import MockDataClient
from enhanced_config_loader import load_config
from logger_config_integration import get_strategy_logger
from typing import Union, Optional, Callable, Dict, Any
import pandas as pd


class AdaptiveDataClient:
    """
    自适应数据客户端
    根据配置自动选择使用真实API还是模拟数据
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger=None, force_mock: bool = False):
        """
        初始化自适应数据客户端
        
        Args:
            config: 配置字典
            logger: 日志器
            force_mock: 强制使用模拟数据
        """
        self.config = config or load_config()
        self.logger = logger or get_strategy_logger("adaptive_data_client")
        
        # 决定使用哪种客户端
        use_mock = force_mock or self.config.get('USE_MOCK_DATA', False) or not self._has_valid_api_key()
        
        if use_mock:
            self.client = MockDataClient()
            self.client_type = "mock"
            self.logger.info("使用模拟数据客户端 🎭")
        else:
            self.client = UnifiedDataClient(config, logger)
            self.client_type = "real"
            self.logger.info("使用真实API数据客户端 🌐")
    
    def _has_valid_api_key(self) -> bool:
        """检查是否有有效的API密钥"""
        api_key = self.config.get('TWELVE_DATA_API_KEY')
        return api_key and api_key != "demo" and len(api_key) > 10
    
    def get_historical_data(self, 
                          symbol: str,
                          timeframe: str = "30min",
                          start_date: str = None,
                          end_date: str = None,
                          limit: int = 1000,
                          **kwargs) -> pd.DataFrame:
        """获取历史数据（自适应）"""
        self.logger.info(f"获取历史数据 ({self.client_type}): {symbol} {timeframe}")
        return self.client.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            **kwargs
        )
    
    def subscribe_realtime(self, 
                          symbol: str,
                          callback: Callable,
                          **kwargs) -> bool:
        """订阅实时数据（自适应）"""
        self.logger.info(f"订阅实时数据 ({self.client_type}): {symbol}")
        return self.client.subscribe_realtime(symbol, callback, **kwargs)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格（自适应）"""
        return self.client.get_latest_price(symbol)
    
    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        status = self.client.get_status()
        status['client_type'] = self.client_type
        status['adaptive_mode'] = True
        return status
    
    def close(self):
        """关闭客户端"""
        self.logger.info(f"关闭{self.client_type}数据客户端")
        self.client.close()
    
    def switch_to_mock(self):
        """切换到模拟数据模式"""
        if self.client_type != "mock":
            self.client.close()
            self.client = MockDataClient()
            self.client_type = "mock"
            self.logger.info("已切换到模拟数据模式 🎭")
    
    def is_mock_mode(self) -> bool:
        """检查是否为模拟模式"""
        return self.client_type == "mock"


def demo_adaptive_client():
    """演示自适应数据客户端"""
    print("🔄 自适应数据客户端演示")
    print("=" * 50)
    
    # 创建自适应客户端（自动检测API）
    client = AdaptiveDataClient()
    
    # 显示客户端类型
    status = client.get_status()
    client_type = "🎭 模拟数据" if status['client_type'] == "mock" else "🌐 真实API"
    print(f"当前使用: {client_type}")
    
    # 获取历史数据
    print(f"\n📊 获取历史数据...")
    df = client.get_historical_data("AAPL", "30min", limit=10)
    if not df.empty:
        print(f"成功获取 {len(df)} 条数据")
        print(f"最新价格: ${df['close'].iloc[-1]:.2f}")
        print(f"价格范围: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # 获取最新价格
    print(f"\n💰 最新价格...")
    latest_price = client.get_latest_price("AAPL")
    if latest_price:
        print(f"AAPL: ${latest_price:.2f}")
    
    # 实时数据演示
    print(f"\n📡 实时数据测试...")
    def demo_callback(data):
        print(f"  📈 {data.symbol}: ${data.price:.2f} from {data.source}")
    
    success = client.subscribe_realtime("AAPL", demo_callback)
    if success:
        print("✅ 实时数据订阅成功")
        import time
        time.sleep(3)  # 运行3秒
    else:
        print("❌ 实时数据订阅失败")
    
    # 显示完整状态
    print(f"\n📊 客户端状态:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    client.close()
    print("\n✅ 演示完成")


if __name__ == "__main__":
    demo_adaptive_client()
