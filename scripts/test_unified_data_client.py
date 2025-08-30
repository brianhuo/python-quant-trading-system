"""
统一数据客户端测试脚本
验证基本功能和集成性
"""

import os
import sys
import time
from datetime import datetime, timedelta

# 设置测试API密钥（如果需要）
os.environ['TWELVE_DATA_API_KEY'] = 'demo'  # 使用demo密钥进行测试

try:
    from unified_data_client import UnifiedDataClient, MarketData, DataType
    from data_client_integration import DataClientManager, setup_trading_data
    from enhanced_config_loader import load_config
    from logger_config_integration import get_strategy_logger
    
    print("✅ 所有模块导入成功")
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)


def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    try:
        # 创建客户端
        client = UnifiedDataClient()
        print("✅ 统一数据客户端创建成功")
        
        # 获取状态
        status = client.get_status()
        print(f"✅ 客户端状态: {status}")
        
        # 关闭客户端
        client.close()
        print("✅ 客户端关闭成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False


def test_config_integration():
    """测试配置集成"""
    print("\n=== 测试配置集成 ===")
    
    try:
        # 加载配置
        config = load_config(environment="development", validate=False)
        print("✅ 配置加载成功")
        
        # 创建带配置的客户端
        client = UnifiedDataClient(config=config)
        print("✅ 配置集成客户端创建成功")
        
        # 验证配置参数
        ticker = config.get('TICKER', 'AAPL')
        timeframe = config.get('DATA_TIMEFRAME', '30min')
        print(f"✅ 配置参数: ticker={ticker}, timeframe={timeframe}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ 配置集成测试失败: {e}")
        return False


def test_data_manager():
    """测试数据管理器"""
    print("\n=== 测试数据管理器 ===")
    
    try:
        # 创建数据管理器
        manager = setup_trading_data(environment="development", strategy_name="test")
        print("✅ 数据管理器创建成功")
        
        # 获取性能统计
        stats = manager.get_performance_stats()
        print(f"✅ 性能统计: {stats}")
        
        # 测试当前价格获取（可能失败，因为使用demo key）
        try:
            price = manager.get_current_price("AAPL")
            if price:
                print(f"✅ 当前价格: ${price:.2f}")
            else:
                print("⚠️ 无法获取当前价格（预期，使用demo key）")
        except Exception as e:
            print(f"⚠️ 价格获取失败（预期）: {e}")
        
        manager.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据管理器测试失败: {e}")
        return False


def test_market_data_structure():
    """测试市场数据结构"""
    print("\n=== 测试市场数据结构 ===")
    
    try:
        # 创建测试数据
        test_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            data_type=DataType.HISTORICAL
        )
        
        print("✅ MarketData 对象创建成功")
        
        # 转换为字典
        data_dict = test_data.to_dict()
        print(f"✅ 数据字典转换: {data_dict}")
        
        return True
        
    except Exception as e:
        print(f"❌ 市场数据结构测试失败: {e}")
        return False


def test_logger_integration():
    """测试日志集成"""
    print("\n=== 测试日志集成 ===")
    
    try:
        # 创建策略日志器
        logger = get_strategy_logger("test_data_client", environment="development")
        print("✅ 策略日志器创建成功")
        
        # 测试市场数据日志
        logger.log_market_data(
            ticker="AAPL",
            price=150.0,
            volume=1000000,
            data_type="test"
        )
        print("✅ 市场数据日志记录成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 日志集成测试失败: {e}")
        return False


def test_cache_functionality():
    """测试缓存功能"""
    print("\n=== 测试缓存功能 ===")
    
    try:
        # 创建客户端
        client = UnifiedDataClient()
        
        # 测试缓存方法
        cache_dir = client.cache_dir
        print(f"✅ 缓存目录: {cache_dir}")
        
        # 测试缓存文件名生成
        filename = client._get_cache_filename("AAPL", "1day", "2025-01-01", "2025-01-31")
        print(f"✅ 缓存文件名: {filename}")
        
        # 测试清理缓存
        client.clear_cache("TEST")
        print("✅ 缓存清理功能正常")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ 缓存功能测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始统一数据客户端测试")
    print("=" * 50)
    
    tests = [
        ("基本功能", test_basic_functionality),
        ("配置集成", test_config_integration),
        ("数据管理器", test_data_manager),
        ("市场数据结构", test_market_data_structure),
        ("日志集成", test_logger_integration),
        ("缓存功能", test_cache_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("🎯 测试结果汇总")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<15}: {status}")
        if result:
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 总体结果: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！统一数据客户端可以正常使用！")
    elif passed >= total * 0.8:
        print("⚠️ 大部分测试通过，系统基本可用，但有部分问题需要关注")
    else:
        print("❌ 多个测试失败，需要检查配置和依赖")
    
    return passed, total


if __name__ == "__main__":
    passed, total = run_all_tests()
    
    # 返回适当的退出码
    sys.exit(0 if passed == total else 1)
