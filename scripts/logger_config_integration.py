"""
日志系统与配置系统集成模块
提供统一的日志配置管理
"""

from enhanced_config_loader import load_config, EnhancedConfigLoader
from enhanced_logger_setup import EnhancedLoggerSetup, LogFormat, TradingTimeFilter
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os


class LoggerConfigManager:
    """日志配置管理器"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.config = load_config(environment=environment)
        self.logger_setup = EnhancedLoggerSetup(self._prepare_logger_config())
        
    def _prepare_logger_config(self) -> Dict[str, Any]:
        """准备日志配置"""
        log_config = {
            'log_directory': self.config.get('LOG_DIRECTORY', 'logs'),
            'log_level': self.config.get('LOG_LEVEL', 'INFO'),
            'enable_console': not self.config.get('LIVE_TRADING', False),
            'enable_file': True,
            'enable_performance_monitoring': self.config.get('ENABLE_MODEL_MONITOR', True),
            'max_file_size': self.config.get('LOG_MAX_FILE_SIZE', 10 * 1024 * 1024),
            'backup_count': self.config.get('LOG_BACKUP_COUNT', 5),
            'trading_hours_filter': self.config.get('ENABLE_TRADING_HOURS_FILTER', True)
        }
        
        # 根据环境调整配置
        if self.environment == 'development':
            log_config.update({
                'log_level': 'DEBUG',
                'enable_console': True,
                'log_format': LogFormat.DETAILED
            })
        elif self.environment == 'testing':
            log_config.update({
                'log_level': 'INFO',
                'enable_console': True,
                'log_format': LogFormat.STRUCTURED
            })
        elif self.environment == 'production':
            log_config.update({
                'log_level': 'WARNING',
                'enable_console': False,
                'log_format': LogFormat.JSON
            })
            
        return log_config
    
    def create_strategy_logger(self, strategy_name: str) -> logging.Logger:
        """为策略创建专用日志器"""
        
        # 策略特定配置
        strategy_config = {
            'name': f"strategy.{strategy_name}",
            'level': self.config.get('LOG_LEVEL', 'INFO'),
            'log_format': LogFormat.JSON if self.config.get('LIVE_TRADING') else LogFormat.STRUCTURED,
            'enable_console': not self.config.get('LIVE_TRADING', False),
            'enable_file': True,
            'enable_performance_monitoring': True
        }
        
        # 创建自定义过滤器
        filters = []
        if self.config.get('ENABLE_TRADING_HOURS_FILTER', True):
            trading_filter = TradingTimeFilter()
            filters.append(trading_filter)
        
        logger = self.logger_setup.create_logger(
            name=strategy_config['name'],
            level=strategy_config['level'],
            log_format=strategy_config['log_format'],
            enable_console=strategy_config['enable_console'],
            enable_file=strategy_config['enable_file'],
            enable_performance_monitoring=strategy_config['enable_performance_monitoring'],
            custom_filters=filters
        )
        
        # 添加策略上下文
        logger.strategy_name = strategy_name
        logger.ticker = self.config.get('TICKER', 'UNKNOWN')
        logger.environment = self.environment
        
        # 记录策略启动
        logger.info(f"Strategy {strategy_name} logger initialized", extra={
            'strategy_name': strategy_name,
            'ticker': logger.ticker,
            'environment': self.environment,
            'config': {
                'live_trading': self.config.get('LIVE_TRADING', False),
                'initial_capital': self.config.get('INIT_CAPITAL', 0),
                'risk_per_trade': self.config.get('RISK_PER_TRADE', 0)
            }
        })
        
        return logger
    
    def create_data_logger(self) -> logging.Logger:
        """创建数据处理专用日志器"""
        
        logger = self.logger_setup.create_logger(
            name="data_processor",
            level=self.config.get('LOG_LEVEL', 'INFO'),
            log_format=LogFormat.STRUCTURED,
            enable_console=self.environment != 'production',
            enable_file=True,
            enable_performance_monitoring=True
        )
        
        logger.info("Data processor logger initialized", extra={
            'data_timeframe': self.config.get('DATA_TIMEFRAME', '30min'),
            'history_years': self.config.get('HISTORY_YEARS', 5),
            'ticker': self.config.get('TICKER', 'UNKNOWN')
        })
        
        return logger
    
    def create_model_logger(self) -> logging.Logger:
        """创建模型训练专用日志器"""
        
        logger = self.logger_setup.create_logger(
            name="model_trainer",
            level=self.config.get('LOG_LEVEL', 'INFO'),
            log_format=LogFormat.STRUCTURED,
            enable_console=self.environment != 'production',
            enable_file=True,
            enable_performance_monitoring=True,
            max_file_size=50 * 1024 * 1024  # 模型训练日志较大
        )
        
        logger.info("Model trainer logger initialized", extra={
            'model_update_interval': self.config.get('MODEL_UPDATE_INTERVAL_DAYS', 3),
            'min_samples': self.config.get('MIN_SAMPLES', 50),
            'rolling_window': self.config.get('ROLLING_WINDOW', 98),
            'feature_selection': self.config.get('FEATURE_SELECTION_METHOD', 'shap_dynamic')
        })
        
        return logger
    
    def create_risk_logger(self) -> logging.Logger:
        """创建风险管理专用日志器"""
        
        logger = self.logger_setup.create_logger(
            name="risk_manager",
            level="WARNING",  # 风险日志通常只记录重要信息
            log_format=LogFormat.JSON,
            enable_console=True,  # 风险信息总是显示
            enable_file=True,
            enable_performance_monitoring=True
        )
        
        logger.info("Risk manager logger initialized", extra={
            'risk_per_trade': self.config.get('RISK_PER_TRADE', 0.02),
            'max_trade_pct': self.config.get('MAX_TRADE_PCT', 0.1),
            'max_drawdown': self.config.get('MAX_DRAWDOWN', 0.1),
            'daily_loss_limit': self.config.get('DAILY_LOSS_LIMIT', -0.03),
            'stop_loss_multiplier': self.config.get('STOPLOSS_MULTIPLIER', 1.8),
            'take_profit_multiplier': self.config.get('TAKEPROFIT_MULTIPLIER', 2.5)
        })
        
        return logger
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取所有日志器的性能统计"""
        return self.logger_setup.get_performance_stats()
    
    def setup_complete_logging_system(self) -> Dict[str, logging.Logger]:
        """设置完整的日志系统"""
        
        loggers = {
            'main': self.create_strategy_logger('main'),
            'data': self.create_data_logger(),
            'model': self.create_model_logger(),
            'risk': self.create_risk_logger()
        }
        
        # 记录系统启动
        main_logger = loggers['main']
        main_logger.info("Complete logging system initialized", extra={
            'environment': self.environment,
            'loggers_created': list(loggers.keys()),
            'log_directory': self.config.get('LOG_DIRECTORY', 'logs'),
            'system_config': {
                'live_trading': self.config.get('LIVE_TRADING', False),
                'ticker': self.config.get('TICKER', 'UNKNOWN'),
                'initial_capital': self.config.get('INIT_CAPITAL', 0)
            }
        })
        
        return loggers


# 便捷函数
def setup_trading_logging(environment: str = "development", 
                         strategy_name: str = "main") -> Dict[str, logging.Logger]:
    """设置交易系统日志的便捷函数"""
    
    manager = LoggerConfigManager(environment)
    
    loggers = {
        'strategy': manager.create_strategy_logger(strategy_name),
        'data': manager.create_data_logger(),
        'model': manager.create_model_logger(),
        'risk': manager.create_risk_logger()
    }
    
    return loggers


def get_strategy_logger(strategy_name: str, environment: str = "development") -> logging.Logger:
    """获取策略专用日志器的便捷函数"""
    manager = LoggerConfigManager(environment)
    return manager.create_strategy_logger(strategy_name)


# 日志装饰器
def log_trading_operation(logger: logging.Logger, operation_type: str):
    """交易操作日志装饰器"""
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            logger.info(f"Starting {operation_type}: {func.__name__}", extra={
                'operation_type': operation_type,
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs': list(kwargs.keys())
            })
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(f"Completed {operation_type}: {func.__name__}", extra={
                    'operation_type': operation_type,
                    'function_name': func.__name__,
                    'execution_time': execution_time,
                    'success': True
                })
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(f"Failed {operation_type}: {func.__name__}", extra={
                    'operation_type': operation_type,
                    'function_name': func.__name__,
                    'execution_time': execution_time,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'success': False
                }, exc_info=True)
                
                raise
                
        return wrapper
    return decorator


if __name__ == "__main__":
    # 演示用法
    print("=== 日志系统集成演示 ===")
    
    # 设置完整的日志系统
    loggers = setup_trading_logging(environment="development", strategy_name="demo_strategy")
    
    # 使用不同类型的日志器
    strategy_logger = loggers['strategy']
    data_logger = loggers['data']
    model_logger = loggers['model']
    risk_logger = loggers['risk']
    
    # 策略日志
    strategy_logger.log_strategy("RSI_Strategy", "BUY", 0.8, ticker="AAPL")
    strategy_logger.log_trade("BUY", "AAPL", 100, 150.0, strategy="RSI_Strategy")
    
    # 数据日志
    data_logger.info("Market data updated", extra={'ticker': 'AAPL', 'price': 150.0})
    
    # 模型日志
    model_logger.info("Model training started", extra={'samples': 1000, 'features': 20})
    
    # 风险日志
    risk_logger.warning("Position size limit reached", extra={
        'current_position': 0.08,
        'limit': 0.1,
        'ticker': 'AAPL'
    })
    
    print("日志演示完成！检查 logs/ 目录查看生成的日志文件。")


