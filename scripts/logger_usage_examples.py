"""
增强版日志系统使用示例
展示完整的日志功能和最佳实践
"""

import time
import random
from logger_config_integration import setup_trading_logging, get_strategy_logger, log_trading_operation
from enhanced_logger_setup import LoggingContext, get_default_logger
from enhanced_config_loader import load_config


def demo_basic_logging():
    """基础日志功能演示"""
    print("=== 基础日志功能演示 ===")
    
    # 获取默认日志器
    logger = get_default_logger("demo_basic")
    
    # 基础日志记录
    logger.debug("调试信息：系统初始化")
    logger.info("信息：程序启动")
    logger.warning("警告：内存使用率较高")
    logger.error("错误：连接失败")
    
    # 带自定义字段的结构化日志
    logger.info("用户登录", extra={
        'user_id': 'user_123',
        'ip_address': '192.168.1.1',
        'login_time': time.time(),
        'session_id': 'sess_456'
    })
    
    print("✅ 基础日志演示完成")


def demo_trading_specific_logging():
    """交易特定日志演示"""
    print("\n=== 交易特定日志演示 ===")
    
    # 获取策略专用日志器
    strategy_logger = get_strategy_logger("RSI_Strategy", environment="development")
    
    # 策略信号日志
    strategy_logger.log_strategy(
        strategy_name="RSI_Strategy",
        signal="BUY",
        confidence=0.85,
        ticker="AAPL",
        rsi_value=25.5,
        price=150.25
    )
    
    # 交易执行日志
    strategy_logger.log_trade(
        action="BUY",
        ticker="AAPL", 
        quantity=100,
        price=150.25,
        strategy="RSI_Strategy",
        commission=1.0,
        slippage=0.05
    )
    
    # 市场数据日志
    strategy_logger.log_market_data(
        ticker="AAPL",
        price=150.30,
        volume=2500000,
        data_type="real_time",
        bid=150.28,
        ask=150.32
    )
    
    print("✅ 交易特定日志演示完成")


def demo_complete_system_logging():
    """完整系统日志演示"""
    print("\n=== 完整系统日志演示 ===")
    
    # 设置完整的日志系统
    loggers = setup_trading_logging(
        environment="development",
        strategy_name="complete_demo"
    )
    
    strategy_logger = loggers['strategy']
    data_logger = loggers['data']
    model_logger = loggers['model']
    risk_logger = loggers['risk']
    
    # 模拟交易流程
    
    # 1. 数据获取
    data_logger.info("开始获取市场数据", extra={
        'ticker': 'AAPL',
        'timeframe': '30min',
        'start_date': '2025-01-01'
    })
    
    # 2. 模型预测
    model_logger.info("开始模型预测", extra={
        'model_type': 'RandomForest',
        'features': ['RSI', 'MACD', 'ATR'],
        'samples': 1000
    })
    
    # 模拟模型训练时间
    time.sleep(0.1)
    
    model_logger.info("模型预测完成", extra={
        'prediction': 'BUY',
        'confidence': 0.78,
        'execution_time': 0.1
    })
    
    # 3. 风险检查
    risk_logger.info("执行风险检查", extra={
        'current_position': 0.05,
        'max_position': 0.1,
        'available_capital': 50000,
        'risk_per_trade': 0.02
    })
    
    # 4. 交易执行
    strategy_logger.log_trade(
        action="BUY",
        ticker="AAPL",
        quantity=100,
        price=150.50,
        strategy="complete_demo"
    )
    
    print("✅ 完整系统日志演示完成")


def demo_performance_monitoring():
    """性能监控演示"""
    print("\n=== 性能监控演示 ===")
    
    logger = get_default_logger("performance_demo")
    
    # 使用性能监控装饰器
    @logger.log_with_metrics
    def slow_operation():
        """模拟慢操作"""
        time.sleep(random.uniform(0.1, 0.5))
        return "操作完成"
    
    @logger.log_with_metrics 
    def fast_operation():
        """模拟快操作"""
        time.sleep(random.uniform(0.01, 0.05))
        return "快速完成"
    
    # 执行多次操作
    for i in range(5):
        result1 = slow_operation()
        result2 = fast_operation()
    
    print("✅ 性能监控演示完成")


def demo_context_logging():
    """上下文日志演示"""
    print("\n=== 上下文日志演示 ===")
    
    logger = get_default_logger("context_demo")
    
    # 使用日志上下文管理器
    with LoggingContext(logger, "数据预处理", dataset="AAPL_1D", rows=1000):
        time.sleep(0.1)  # 模拟处理时间
        logger.info("数据清洗完成", extra={'cleaned_rows': 950})
    
    with LoggingContext(logger, "模型训练", algorithm="RandomForest"):
        time.sleep(0.2)  # 模拟训练时间
        logger.info("训练进度", extra={'epoch': 10, 'accuracy': 0.85})
    
    print("✅ 上下文日志演示完成")


def demo_error_handling():
    """错误处理演示"""
    print("\n=== 错误处理演示 ===")
    
    logger = get_default_logger("error_demo")
    
    # 使用日志装饰器进行错误处理
    @log_trading_operation(logger, "数据获取")
    def fetch_market_data(ticker):
        """模拟数据获取函数"""
        if ticker == "INVALID":
            raise ValueError(f"无效的股票代码: {ticker}")
        
        time.sleep(0.1)
        return {"price": 150.0, "volume": 1000000}
    
    @log_trading_operation(logger, "订单执行")
    def execute_order(action, quantity, price):
        """模拟订单执行函数"""
        if quantity <= 0:
            raise ValueError("订单数量必须大于0")
        
        if price <= 0:
            raise ValueError("价格必须大于0")
        
        time.sleep(0.05)
        return {"order_id": "ORD123", "status": "FILLED"}
    
    # 正常操作
    try:
        data = fetch_market_data("AAPL")
        logger.info("获取数据成功", extra=data)
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
    
    # 异常操作
    try:
        fetch_market_data("INVALID")
    except Exception as e:
        logger.error(f"获取无效数据失败: {e}")
    
    try:
        execute_order("BUY", -100, 150.0)
    except Exception as e:
        logger.error(f"执行订单失败: {e}")
    
    print("✅ 错误处理演示完成")


def demo_environment_specific_logging():
    """环境特定日志演示"""
    print("\n=== 环境特定日志演示 ===")
    
    environments = ["development", "testing", "production"]
    
    for env in environments:
        print(f"\n{env.upper()} 环境:")
        
        # 加载环境特定配置
        config = load_config(environment=env, validate=False)
        
        # 创建环境特定日志器
        logger = get_strategy_logger(f"env_demo_{env}", environment=env)
        
        # 记录环境信息
        logger.info(f"在 {env} 环境中运行", extra={
            'environment': env,
            'live_trading': config.get('LIVE_TRADING', False),
            'log_level': config.get('LOG_LEVEL', 'INFO'),
            'initial_capital': config.get('INIT_CAPITAL', 0)
        })
        
        # 模拟交易操作
        logger.log_trade(
            action="BUY",
            ticker=config.get('TICKER', 'AAPL'),
            quantity=50,
            price=150.0,
            strategy=f"env_demo_{env}"
        )
    
    print("✅ 环境特定日志演示完成")


def demo_custom_filtering():
    """自定义过滤演示"""
    print("\n=== 自定义过滤演示 ===")
    
    from enhanced_logger_setup import TradingTimeFilter, EnhancedLoggerSetup
    
    # 创建带自定义过滤器的日志器
    setup = EnhancedLoggerSetup()
    trading_filter = TradingTimeFilter({
        'NYSE': (9, 16),
        'FOREX': (0, 24)
    })
    
    logger = setup.create_logger(
        name="filtered_demo",
        level="DEBUG",
        custom_filters=[trading_filter]
    )
    
    # 在不同时间测试过滤
    current_hour = time.localtime().tm_hour
    
    logger.debug(f"当前时间: {current_hour}:00")
    logger.info("信息级别日志")
    logger.warning("警告级别日志")
    logger.error("错误级别日志")
    
    print(f"当前时间: {current_hour}:00")
    print("根据交易时间过滤器，某些日志可能被过滤")
    print("✅ 自定义过滤演示完成")


def run_all_demos():
    """运行所有演示"""
    print("🚀 开始增强版日志系统完整演示")
    print("=" * 60)
    
    try:
        demo_basic_logging()
        demo_trading_specific_logging()
        demo_complete_system_logging()
        demo_performance_monitoring()
        demo_context_logging()
        demo_error_handling()
        demo_environment_specific_logging()
        demo_custom_filtering()
        
        print("\n" + "=" * 60)
        print("🎉 所有日志演示完成！")
        print("\n检查 logs/ 目录查看生成的日志文件：")
        print("  - trading.demo_basic.log")
        print("  - trading.RSI_Strategy.log")
        print("  - trading.complete_demo.log")
        print("  - 以及其他演示日志文件...")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_demos()
