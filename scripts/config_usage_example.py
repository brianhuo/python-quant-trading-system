"""
配置加载器使用示例
演示如何使用增强版配置加载器
"""

import os
import sys
from enhanced_config_loader import load_config, get_config_loader, EnhancedConfigLoader


def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 简单加载配置
    config = load_config(environment="development")
    
    print(f"Ticker: {config['TICKER']}")
    print(f"Initial Capital: {config['INIT_CAPITAL']}")
    print(f"Risk per Trade: {config['RISK_PER_TRADE']}")
    print(f"Live Trading: {config['LIVE_TRADING']}")
    print()


def example_environment_specific():
    """环境特定配置示例"""
    print("=== 环境特定配置示例 ===")
    
    environments = ["development", "testing", "production"]
    
    for env in environments:
        try:
            config = load_config(environment=env, validate=False)  # 跳过验证避免API密钥错误
            print(f"{env.capitalize()} Environment:")
            print(f"  Initial Capital: {config['INIT_CAPITAL']}")
            print(f"  Risk per Trade: {config['RISK_PER_TRADE']}")
            print(f"  Live Trading: {config['LIVE_TRADING']}")
            print(f"  Log Level: {config['LOG_LEVEL']}")
            print()
        except Exception as e:
            print(f"Failed to load {env} config: {e}")
            print()


def example_advanced_usage():
    """高级使用示例"""
    print("=== 高级使用示例 ===")
    
    # 创建配置加载器实例
    loader = EnhancedConfigLoader(
        config_dir=".",
        environment="development", 
        enable_security=True
    )
    
    try:
        # 加载配置
        config = loader.load(validate=True)
        
        print("Configuration loaded successfully!")
        print("Key parameters:")
        for key in ['TICKER', 'INIT_CAPITAL', 'RISK_PER_TRADE', 'DATA_TIMEFRAME']:
            print(f"  {key}: {config.get(key)}")
        
        # 保存当前配置
        # loader.save_config(config, "current_config.yaml")
        # print("Configuration saved to current_config.yaml")
        
    except Exception as e:
        print(f"Configuration loading failed: {e}")
    
    print()


def example_validation_demo():
    """配置验证示例"""
    print("=== 配置验证示例 ===")
    
    # 创建无效配置进行演示
    loader = EnhancedConfigLoader(environment="development")
    
    # 模拟无效配置
    invalid_config = {
        "trading": {
            "ticker": "INVALID_TICKER_TOO_LONG",  # 无效股票代码
            "initial_capital": -1000,  # 负数资金
        },
        "risk_management": {
            "risk_per_trade": 1.5,  # 超出范围
        }
    }
    
    try:
        from config_schema import validate_config
        errors = validate_config(invalid_config)
        if errors:
            print("Validation errors found:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("Configuration is valid!")
    except Exception as e:
        print(f"Validation failed: {e}")
    
    print()


def example_security_features():
    """安全功能示例"""
    print("=== 安全功能示例 ===")
    
    from enhanced_config_loader import ConfigSecurityManager
    
    # 创建安全管理器
    security_manager = ConfigSecurityManager()
    
    # 模拟API密钥验证
    test_keys = [
        "valid_api_key_123456789",
        "short",  # 太短
        "invalid-key-with-special-chars!@#",  # 无效字符
        ""  # 空字符串
    ]
    
    print("API Key validation results:")
    for key in test_keys:
        is_valid = security_manager.validate_api_key(key)
        print(f"  '{key}' -> {'Valid' if is_valid else 'Invalid'}")
    
    print()


def example_error_handling():
    """错误处理示例"""
    print("=== 错误处理示例 ===")
    
    # 尝试加载不存在的配置文件
    try:
        loader = EnhancedConfigLoader(
            config_dir="/non/existent/path",
            environment="nonexistent"
        )
        config = loader.load()
    except Exception as e:
        print(f"Expected error when loading from non-existent path: {type(e).__name__}")
    
    # 尝试加载缺少必需环境变量的生产配置
    try:
        # 暂时移除API密钥环境变量
        original_key = os.environ.pop("TWELVE_DATA_API_KEY", None)
        
        config = load_config(environment="production", validate=True)
        
        # 恢复环境变量
        if original_key:
            os.environ["TWELVE_DATA_API_KEY"] = original_key
            
    except Exception as e:
        print(f"Expected error for missing API key: {type(e).__name__}")
        
        # 恢复环境变量
        if 'original_key' in locals() and original_key:
            os.environ["TWELVE_DATA_API_KEY"] = original_key
    
    print()


if __name__ == "__main__":
    print("配置加载器使用示例")
    print("=" * 50)
    print()
    
    try:
        example_basic_usage()
        example_environment_specific()
        example_advanced_usage()
        example_validation_demo()
        example_security_features()
        example_error_handling()
        
        print("所有示例执行完成!")
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        import traceback
        traceback.print_exc()



