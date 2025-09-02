"""
统一数据客户端演示
展示完整功能和最佳实践
"""

import time
from datetime import datetime
from unified_data_client import UnifiedDataClient, MarketData, DataType
from data_client_integration import DataClientManager, setup_trading_data


def demo_basic_usage():
    """基础使用演示"""
    print("=== 基础使用演示 ===")
    
    # 创建客户端（不需要真实API密钥）
    client = UnifiedDataClient()
    
    # 获取客户端状态
    status = client.get_status()
    print(f"客户端状态: {status}")
    
    # 测试缓存目录
    print(f"缓存目录: {client.cache_dir}")
    
    # 测试工具方法
    days_back = client._calculate_days_back("30min", 1000)
    print(f"30分钟数据1000条需要回看: {days_back} 天")
    
    client.close()
    print("✅ 基础使用演示完成")


def demo_market_data_structure():
    """市场数据结构演示"""
    print("\n=== 市场数据结构演示 ===")
    
    # 创建历史数据
    historical_data = MarketData(
        symbol="AAPL",
        timestamp=datetime.now(),
        open=150.0,
        high=155.0,
        low=149.0,
        close=154.0,
        volume=1000000,
        data_type=DataType.HISTORICAL
    )
    
    print("历史数据:")
    print(f"  {historical_data.symbol}: ${historical_data.close}")
    print(f"  时间: {historical_data.timestamp}")
    print(f"  类型: {historical_data.data_type.value}")
    
    # 创建实时数据
    realtime_data = MarketData(
        symbol="AAPL",
        timestamp=datetime.now(),
        price=154.25,
        data_type=DataType.REALTIME,
        source="twelvedata_ws"
    )
    
    print("\n实时数据:")
    print(f"  {realtime_data.symbol}: ${realtime_data.price}")
    print(f"  来源: {realtime_data.source}")
    
    # 转换为字典
    data_dict = historical_data.to_dict()
    print(f"\n数据字典: {data_dict}")
    
    print("✅ 市场数据结构演示完成")


def demo_data_manager():
    """数据管理器演示"""
    print("\n=== 数据管理器演示 ===")
    
    # 创建数据管理器
    manager = setup_trading_data(environment="development", strategy_name="demo")
    
    # 获取配置信息
    print(f"默认股票: {manager.ticker}")
    print(f"默认时间框架: {manager.default_timeframe}")
    print(f"历史年数: {manager.history_years}")
    
    # 获取性能统计
    stats = manager.get_performance_stats()
    print(f"\n性能统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试数据质量验证（使用空DataFrame）
    import pandas as pd
    empty_df = pd.DataFrame()
    quality = manager.validate_data_quality(empty_df, "TEST")
    print(f"\n空数据质量检查: {quality}")
    
    manager.close()
    print("✅ 数据管理器演示完成")


def demo_caching_system():
    """缓存系统演示"""
    print("\n=== 缓存系统演示 ===")
    
    client = UnifiedDataClient()
    
    # 测试缓存文件名生成
    filename = client._get_cache_filename("AAPL", "1day", "2025-01-01", "2025-12-31")
    print(f"缓存文件名: {filename}")
    
    # 测试日期计算
    days_back_1min = client._calculate_days_back("1min", 1440)  # 1天的分钟数据
    days_back_1day = client._calculate_days_back("1day", 365)   # 1年的日数据
    
    print(f"1分钟数据1440条需要: {days_back_1min} 天")
    print(f"1日数据365条需要: {days_back_1day} 天")
    
    # 测试缓存清理
    client.clear_cache("TEST")
    print("测试缓存清理完成")
    
    client.close()
    print("✅ 缓存系统演示完成")


def demo_logging_integration():
    """日志集成演示"""
    print("\n=== 日志集成演示 ===")
    
    from logger_config_integration import get_strategy_logger
    
    # 创建日志器
    logger = get_strategy_logger("demo_data_client", "development")
    
    # 记录市场数据
    logger.log_market_data(
        ticker="AAPL",
        price=154.25,
        volume=2500000,
        data_type="demo"
    )
    
    # 记录策略信号
    logger.log_strategy(
        strategy_name="DataDemo",
        signal="DATA_READY",
        confidence=1.0,
        ticker="AAPL"
    )
    
    print("✅ 日志集成演示完成")


def demo_configuration_integration():
    """配置集成演示"""
    print("\n=== 配置集成演示 ===")
    
    from enhanced_config_loader import load_config
    
    # 加载不同环境配置
    environments = ["development", "testing", "production"]
    
    for env in environments:
        try:
            config = load_config(environment=env, validate=False)
            print(f"\n{env.upper()} 环境配置:")
            print(f"  股票代码: {config.get('TICKER', 'N/A')}")
            print(f"  数据框架: {config.get('DATA_TIMEFRAME', 'N/A')}")
            print(f"  历史年数: {config.get('HISTORY_YEARS', 'N/A')}")
            print(f"  初始资金: ${config.get('INIT_CAPITAL', 0):,}")
            print(f"  实盘交易: {config.get('LIVE_TRADING', False)}")
            
        except Exception as e:
            print(f"{env} 环境配置加载失败: {e}")
    
    print("\n✅ 配置集成演示完成")


def demo_error_handling():
    """错误处理演示"""
    print("\n=== 错误处理演示 ===")
    
    client = UnifiedDataClient()
    
    # 测试无效数据处理
    try:
        # 尝试获取无效符号的数据（会失败，但不会崩溃）
        price = client.get_latest_price("INVALID_SYMBOL")
        print(f"无效符号价格: {price}")
    except Exception as e:
        print(f"预期错误（已处理）: {e}")
    
    # 测试连接状态检查
    print(f"WebSocket状态: {client.ws_status.value}")
    
    client.close()
    print("✅ 错误处理演示完成")


def demo_advanced_features():
    """高级功能演示"""
    print("\n=== 高级功能演示 ===")
    
    # 演示回调函数
    def price_callback(data: MarketData):
        print(f"价格更新: {data.symbol} = ${data.price:.2f} at {data.timestamp}")
    
    def data_validator(data: MarketData) -> bool:
        """数据验证器"""
        if data.price and data.price > 0:
            return True
        return False
    
    # 演示数据处理流程
    print("数据处理流程演示:")
    print("1. 数据接收 -> 2. 数据验证 -> 3. 数据缓存 -> 4. 回调通知")
    
    # 创建测试数据
    test_data = MarketData(
        symbol="DEMO",
        timestamp=datetime.now(),
        price=100.0,
        data_type=DataType.REALTIME
    )
    
    # 验证数据
    is_valid = data_validator(test_data)
    print(f"数据验证结果: {'✅ 有效' if is_valid else '❌ 无效'}")
    
    # 模拟回调
    if is_valid:
        price_callback(test_data)
    
    print("✅ 高级功能演示完成")


def run_complete_demo():
    """运行完整演示"""
    print("🚀 统一数据客户端完整演示")
    print("=" * 60)
    
    demos = [
        demo_basic_usage,
        demo_market_data_structure,
        demo_data_manager,
        demo_caching_system,
        demo_logging_integration,
        demo_configuration_integration,
        demo_error_handling,
        demo_advanced_features
    ]
    
    for demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"❌ 演示 {demo_func.__name__} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 统一数据客户端演示完成！")
    print("\n主要特性:")
    print("✅ 统一的历史和实时数据接口")
    print("✅ 智能缓存和数据管理")
    print("✅ 深度配置和日志集成")
    print("✅ 强大的错误处理机制")
    print("✅ 灵活的回调和验证系统")
    print("✅ 多环境支持和优化")


if __name__ == "__main__":
    run_complete_demo()


