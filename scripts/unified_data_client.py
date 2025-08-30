"""
统一数据客户端 - 整合历史数据和实时数据获取
支持TwelveData API的REST和WebSocket接口
"""

import asyncio
import threading
import time
import json
import requests
import pandas as pd
import websocket
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
import queue
import os
from pathlib import Path
from dateutil import tz

# 集成我们的配置和日志系统
from enhanced_config_loader import load_config
from logger_config_integration import get_strategy_logger


class DataType(Enum):
    """数据类型枚举"""
    HISTORICAL = "historical"
    REALTIME = "realtime"
    LATEST = "latest"


class ConnectionStatus(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class MarketData:
    """标准化市场数据格式"""
    symbol: str
    timestamp: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    price: Optional[float] = None  # 用于实时价格
    data_type: DataType = DataType.LATEST
    source: str = "twelvedata"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'price': self.price,
            'data_type': self.data_type.value,
            'source': self.source
        }


class UnifiedDataClient:
    """统一数据客户端 - 整合REST API和WebSocket"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        """
        初始化统一数据客户端
        
        Args:
            config: 配置字典，如果为None则自动加载
            logger: 日志器，如果为None则自动创建
        """
        # 加载配置
        self.config = config or load_config()
        
        # 设置日志
        self.logger = logger or get_strategy_logger("data_client")
        
        # API配置
        self.api_key = self.config.get('TWELVE_DATA_API_KEY')
        if not self.api_key:
            self.logger.warning("未找到API密钥，某些功能将不可用")
            self.api_key = "demo"  # 使用demo密钥进行基础功能测试
        
        # REST API配置
        self.base_url = "https://api.twelvedata.com/time_series"
        self.max_retries = self.config.get('MAX_RETRIES', 3)
        
        # WebSocket配置
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_status = ConnectionStatus.DISCONNECTED
        self.ws_thread: Optional[threading.Thread] = None
        self.data_callbacks: Dict[str, List[Callable]] = {}
        self.latest_data: Dict[str, MarketData] = {}
        self.data_lock = threading.Lock()
        
        # 缓存配置
        self.cache_dir = Path(self.config.get('CACHE_DIRECTORY', 'cache'))
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_cache = self.config.get('ENABLE_DATA_CACHE', True)
        
        # 数据队列（用于异步处理）
        self.data_queue = queue.Queue()
        
        self.logger.info("统一数据客户端初始化完成")
    
    # ==================== 历史数据获取 ====================
    
    def get_historical_data(self, 
                          symbol: str,
                          timeframe: str = "30min",
                          start_date: str = None,
                          end_date: str = None,
                          limit: int = 1000,
                          timezone: str = "America/New_York") -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbol: 股票代码
            timeframe: 时间框架 (1min, 5min, 30min, 1h, 1day)
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            limit: 数据条数限制
            timezone: 时区
            
        Returns:
            DataFrame: 历史数据
        """
        self.logger.info(f"获取历史数据: {symbol}, {timeframe}, {start_date} to {end_date}")
        
        # 设置默认日期
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            days_back = self._calculate_days_back(timeframe, limit)
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # 检查缓存
        if self.enable_cache:
            cached_data = self._get_cached_data(symbol, timeframe, start_date, end_date)
            if cached_data is not None:
                self.logger.info("从缓存加载历史数据")
                return cached_data
        
        # 从API获取数据
        try:
            df = self._fetch_historical_data(symbol, timeframe, start_date, end_date, timezone, limit)
            
            # 缓存数据
            if self.enable_cache and not df.empty:
                self._cache_data(df, symbol, timeframe, start_date, end_date)
            
            # 记录日志
            self.logger.log_market_data(
                ticker=symbol,
                price=df['close'].iloc[-1] if not df.empty else 0,
                volume=int(df['volume'].iloc[-1]) if not df.empty else 0,
                data_type="historical",
                records_count=len(df)
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _fetch_historical_data(self, symbol: str, interval: str, start_date: str, 
                             end_date: str, timezone: str, limit: int) -> pd.DataFrame:
        """从API获取历史数据"""
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "apikey": self.api_key,
            "outputsize": min(limit, 5000),
            "timezone": timezone
        }
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"API请求尝试 {attempt + 1}/{self.max_retries}")
                
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("status") != "ok":
                    error_msg = f"API错误: {data.get('code', 'Unknown')} - {data.get('message', 'Unknown error')}"
                    self.logger.error(error_msg)
                    return pd.DataFrame()
                
                if not data.get("values"):
                    self.logger.info(f"无数据返回: {symbol} {start_date} to {end_date}")
                    return pd.DataFrame()
                
                # 处理数据
                df = pd.DataFrame(data["values"])
                df = self._process_historical_data(df, timezone)
                
                self.logger.info(f"成功获取 {len(df)} 条历史数据")
                return df
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 10))
            except Exception as e:
                self.logger.error(f"处理历史数据时出错: {e}")
                return pd.DataFrame()
        
        self.logger.error(f"获取历史数据失败，已重试 {self.max_retries} 次")
        return pd.DataFrame()
    
    def _process_historical_data(self, df: pd.DataFrame, timezone: str) -> pd.DataFrame:
        """处理和清洗历史数据"""
        try:
            # 转换时间索引
            df["datetime"] = pd.to_datetime(df["datetime"])
            if df["datetime"].dt.tz is None:
                df["datetime"] = df["datetime"].dt.tz_localize(timezone)
            
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)
            
            # 转换数值列
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除重复和无效数据
            df = df[~df.index.duplicated(keep='first')]
            df.dropna(subset=['close'], inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"处理历史数据失败: {e}")
            return pd.DataFrame()
    
    # ==================== 实时数据获取 ====================
    
    def subscribe_realtime(self, 
                          symbol: str, 
                          callback: Callable[[MarketData], None],
                          auto_reconnect: bool = True) -> bool:
        """
        订阅实时数据
        
        Args:
            symbol: 股票代码
            callback: 数据回调函数
            auto_reconnect: 是否自动重连
            
        Returns:
            bool: 订阅是否成功
        """
        self.logger.info(f"订阅实时数据: {symbol}")
        
        # 添加回调函数
        if symbol not in self.data_callbacks:
            self.data_callbacks[symbol] = []
        self.data_callbacks[symbol].append(callback)
        
        # 启动WebSocket连接
        if self.ws_status == ConnectionStatus.DISCONNECTED:
            return self._start_websocket(auto_reconnect)
        elif self.ws_status == ConnectionStatus.CONNECTED:
            # 如果已连接，发送订阅消息
            self._subscribe_symbol(symbol)
            return True
        
        return False
    
    def unsubscribe_realtime(self, symbol: str, callback: Callable = None) -> bool:
        """
        取消订阅实时数据
        
        Args:
            symbol: 股票代码
            callback: 要移除的回调函数，如果为None则移除所有
            
        Returns:
            bool: 取消订阅是否成功
        """
        self.logger.info(f"取消订阅实时数据: {symbol}")
        
        if symbol in self.data_callbacks:
            if callback:
                if callback in self.data_callbacks[symbol]:
                    self.data_callbacks[symbol].remove(callback)
            else:
                self.data_callbacks[symbol].clear()
            
            # 如果没有回调函数了，取消订阅
            if not self.data_callbacks[symbol]:
                del self.data_callbacks[symbol]
                self._unsubscribe_symbol(symbol)
                
        return True
    
    def _start_websocket(self, auto_reconnect: bool = True) -> bool:
        """启动WebSocket连接"""
        if self.ws_status in [ConnectionStatus.CONNECTING, ConnectionStatus.CONNECTED]:
            return True
        
        self.ws_status = ConnectionStatus.CONNECTING
        self.logger.info("启动WebSocket连接")
        
        try:
            ws_url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_ws_open,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            
            # 在单独线程中运行WebSocket
            self.ws_thread = threading.Thread(
                target=self._run_websocket,
                args=(auto_reconnect,),
                daemon=True
            )
            self.ws_thread.start()
            
            # 等待连接建立
            start_time = time.time()
            while self.ws_status == ConnectionStatus.CONNECTING and time.time() - start_time < 10:
                time.sleep(0.1)
            
            return self.ws_status == ConnectionStatus.CONNECTED
            
        except Exception as e:
            self.logger.error(f"启动WebSocket失败: {e}")
            self.ws_status = ConnectionStatus.ERROR
            return False
    
    def _run_websocket(self, auto_reconnect: bool):
        """运行WebSocket连接"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
                break
            except Exception as e:
                retry_count += 1
                self.logger.error(f"WebSocket连接失败 (尝试 {retry_count}/{max_retries}): {e}")
                
                if auto_reconnect and retry_count < max_retries:
                    self.ws_status = ConnectionStatus.RECONNECTING
                    time.sleep(min(2 ** retry_count, 30))
                else:
                    self.ws_status = ConnectionStatus.ERROR
                    break
    
    def _on_ws_open(self, ws):
        """WebSocket连接打开"""
        self.logger.info("WebSocket连接已建立")
        self.ws_status = ConnectionStatus.CONNECTED
        
        # 订阅所有待订阅的符号
        for symbol in self.data_callbacks.keys():
            self._subscribe_symbol(symbol)
    
    def _on_ws_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            
            if data.get('event') == 'price':
                symbol = data.get('symbol', '')
                price = float(data.get('price', 0))
                timestamp = datetime.now(tz=tz.UTC)
                
                # 创建标准化数据
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=price,
                    data_type=DataType.REALTIME,
                    source="twelvedata_ws"
                )
                
                # 更新最新数据
                with self.data_lock:
                    self.latest_data[symbol] = market_data
                
                # 调用回调函数
                if symbol in self.data_callbacks:
                    for callback in self.data_callbacks[symbol]:
                        try:
                            callback(market_data)
                        except Exception as e:
                            self.logger.error(f"回调函数执行失败: {e}")
                
                # 记录实时数据日志
                self.logger.log_market_data(
                    ticker=symbol,
                    price=price,
                    data_type="realtime"
                )
                
        except json.JSONDecodeError:
            self.logger.error(f"无法解析WebSocket消息: {message}")
        except Exception as e:
            self.logger.error(f"处理WebSocket消息失败: {e}")
    
    def _on_ws_error(self, ws, error):
        """WebSocket错误处理"""
        self.logger.error(f"WebSocket错误: {error}")
        self.ws_status = ConnectionStatus.ERROR
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """WebSocket连接关闭"""
        self.logger.info(f"WebSocket连接关闭: {close_status_code} - {close_msg}")
        self.ws_status = ConnectionStatus.DISCONNECTED
    
    def _subscribe_symbol(self, symbol: str):
        """订阅指定符号"""
        if self.ws and self.ws_status == ConnectionStatus.CONNECTED:
            subscribe_msg = {
                "action": "subscribe",
                "params": {"symbols": symbol}
            }
            self.ws.send(json.dumps(subscribe_msg))
            self.logger.info(f"已发送订阅请求: {symbol}")
    
    def _unsubscribe_symbol(self, symbol: str):
        """取消订阅指定符号"""
        if self.ws and self.ws_status == ConnectionStatus.CONNECTED:
            unsubscribe_msg = {
                "action": "unsubscribe",
                "params": {"symbols": symbol}
            }
            self.ws.send(json.dumps(unsubscribe_msg))
            self.logger.info(f"已发送取消订阅请求: {symbol}")
    
    # ==================== 统一接口方法 ====================
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        获取最新价格（优先使用实时数据，回退到API查询）
        
        Args:
            symbol: 股票代码
            
        Returns:
            Optional[float]: 最新价格
        """
        # 优先从实时数据获取
        with self.data_lock:
            if symbol in self.latest_data:
                latest = self.latest_data[symbol]
                # 检查数据新鲜度（5分钟内）
                if (datetime.now(tz=tz.UTC) - latest.timestamp).total_seconds() < 300:
                    return latest.price
        
        # 回退到API查询
        self.logger.debug(f"从API获取最新价格: {symbol}")
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            df = self.get_historical_data(symbol, "1min", today, today, limit=1)
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception as e:
            self.logger.error(f"获取最新价格失败: {e}")
        
        return None
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """
        获取最新的完整市场数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            Optional[MarketData]: 最新市场数据
        """
        with self.data_lock:
            return self.latest_data.get(symbol)
    
    # ==================== 缓存管理 ====================
    
    def _get_cache_filename(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """生成缓存文件名"""
        return f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
    
    def _get_cached_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        cache_file = self.cache_dir / self._get_cache_filename(symbol, timeframe, start_date, end_date)
        
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col="datetime", parse_dates=True)
                self.logger.debug(f"从缓存加载数据: {cache_file}")
                return df
            except Exception as e:
                self.logger.warning(f"读取缓存文件失败: {e}")
        
        return None
    
    def _cache_data(self, df: pd.DataFrame, symbol: str, timeframe: str, start_date: str, end_date: str):
        """缓存数据到文件"""
        cache_file = self.cache_dir / self._get_cache_filename(symbol, timeframe, start_date, end_date)
        
        try:
            df.to_csv(cache_file)
            self.logger.debug(f"数据已缓存: {cache_file}")
        except Exception as e:
            self.logger.warning(f"缓存数据失败: {e}")
    
    def clear_cache(self, symbol: str = None):
        """清理缓存"""
        try:
            if symbol:
                # 清理特定符号的缓存
                pattern = f"{symbol}_*.csv"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                self.logger.info(f"已清理 {symbol} 的缓存")
            else:
                # 清理所有缓存
                for cache_file in self.cache_dir.glob("*.csv"):
                    cache_file.unlink()
                self.logger.info("已清理所有缓存")
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
    
    # ==================== 工具方法 ====================
    
    def _calculate_days_back(self, timeframe: str, limit: int) -> int:
        """根据时间框架和限制计算需要回溯的天数"""
        points_per_day = {
            "1min": 1440,  # 24 * 60
            "5min": 288,   # 24 * 12
            "15min": 96,   # 24 * 4
            "30min": 48,   # 24 * 2
            "1h": 24,
            "1day": 1
        }
        
        daily_points = points_per_day.get(timeframe, 24)
        return max(limit // daily_points, 30)  # 至少30天
    
    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        return {
            "websocket_status": self.ws_status.value,
            "subscribed_symbols": list(self.data_callbacks.keys()),
            "latest_data_count": len(self.latest_data),
            "cache_enabled": self.enable_cache,
            "api_key_configured": bool(self.api_key)
        }
    
    def close(self):
        """关闭客户端，清理资源"""
        self.logger.info("关闭统一数据客户端")
        
        # 关闭WebSocket
        if self.ws:
            self.ws.close()
        
        # 清理回调函数
        self.data_callbacks.clear()
        
        # 清理最新数据
        with self.data_lock:
            self.latest_data.clear()
        
        self.ws_status = ConnectionStatus.DISCONNECTED
        self.logger.info("统一数据客户端已关闭")


# ==================== 便捷函数 ====================

def create_data_client(config: Dict[str, Any] = None) -> UnifiedDataClient:
    """创建统一数据客户端的便捷函数"""
    return UnifiedDataClient(config)


def get_historical_data(symbol: str, 
                       timeframe: str = "30min",
                       days_back: int = 30,
                       config: Dict[str, Any] = None) -> pd.DataFrame:
    """获取历史数据的便捷函数"""
    client = create_data_client(config)
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    return client.get_historical_data(symbol, timeframe, start_date, end_date)


if __name__ == "__main__":
    # 演示用法
    print("=== 统一数据客户端演示 ===")
    
    # 创建客户端
    client = create_data_client()
    
    # 获取历史数据
    print("1. 获取历史数据...")
    df = client.get_historical_data("AAPL", "1day", limit=5)
    if not df.empty:
        print(f"获取到 {len(df)} 条历史数据")
        print(df.tail())
    
    # 获取最新价格
    print("\n2. 获取最新价格...")
    latest_price = client.get_latest_price("AAPL")
    if latest_price:
        print(f"AAPL 最新价格: ${latest_price:.2f}")
    
    # 订阅实时数据
    print("\n3. 订阅实时数据...")
    def price_callback(data: MarketData):
        print(f"实时价格更新: {data.symbol} = ${data.price:.2f}")
    
    # 注意：实际使用时需要保持程序运行来接收实时数据
    success = client.subscribe_realtime("AAPL", price_callback)
    if success:
        print("实时数据订阅成功")
        # time.sleep(10)  # 运行10秒接收数据
    
    # 获取状态
    print("\n4. 客户端状态:")
    status = client.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 关闭客户端
    client.close()
    print("\n演示完成！")
