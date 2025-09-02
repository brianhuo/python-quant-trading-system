"""
增强版日志系统
集成结构化日志、性能监控、智能过滤、配置集成等企业级功能
"""

import logging
import logging.handlers
import json
import time
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import functools
import traceback
import platform
import psutil


class LogLevel(Enum):
    """日志级别枚举"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


class LogFormat(Enum):
    """日志格式枚举"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class LogMetrics:
    """日志性能指标"""
    timestamp: float
    level: str
    module: str
    function: str
    line_number: int
    execution_time: float
    memory_usage: float
    cpu_usage: float
    thread_id: int
    process_id: int


@dataclass
class TradingLogContext:
    """交易日志上下文"""
    session_id: str
    strategy_name: str
    ticker: str
    action: str
    timestamp: str
    market_state: str
    portfolio_value: Optional[float] = None
    position_size: Optional[float] = None
    price: Optional[float] = None
    pnl: Optional[float] = None


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def __init__(self, format_type: LogFormat = LogFormat.STRUCTURED):
        super().__init__()
        self.format_type = format_type
        
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        
        # 基础信息
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
            "message": record.getMessage()
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # 添加自定义字段
        custom_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 
                          'relativeCreated', 'thread', 'threadName', 'processName', 
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                custom_fields[key] = value
        
        if custom_fields:
            log_data["custom"] = custom_fields
        
        # 根据格式类型输出
        if self.format_type == LogFormat.JSON:
            return json.dumps(log_data, ensure_ascii=False, default=str)
        elif self.format_type == LogFormat.SIMPLE:
            return f"{log_data['timestamp']} | {log_data['level']} | {log_data['message']}"
        elif self.format_type == LogFormat.DETAILED:
            return (f"{log_data['timestamp']} | {log_data['level']} | "
                   f"{log_data['module']}.{log_data['function']}:{log_data['line']} | "
                   f"[PID:{log_data['process']}] [TID:{log_data['thread']}] | "
                   f"{log_data['message']}")
        else:  # STRUCTURED
            return json.dumps(log_data, ensure_ascii=False, default=str, indent=2)


class TradingTimeFilter(logging.Filter):
    """交易时间智能过滤器"""
    
    def __init__(self, trading_hours: Dict[str, tuple] = None):
        super().__init__()
        self.trading_hours = trading_hours or {
            'NYSE': (9, 16),  # 9:00-16:00 EST
            'NASDAQ': (9, 16),
            'FOREX': (0, 24),  # 24小时
        }
        self.current_market = 'NYSE'
    
    def filter(self, record: logging.LogRecord) -> bool:
        """根据交易时间和日志级别过滤"""
        current_hour = datetime.now().hour
        start_hour, end_hour = self.trading_hours.get(self.current_market, (0, 24))
        
        is_trading_hours = start_hour <= current_hour < end_hour
        
        # 交易时间内提高过滤级别
        if is_trading_hours:
            return record.levelno >= logging.WARNING
        else:
            return record.levelno >= logging.DEBUG


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = []
        self.lock = threading.Lock()
        
    def log_performance(self, logger_name: str, level: str, execution_time: float):
        """记录性能指标"""
        with self.lock:
            try:
                metrics = LogMetrics(
                    timestamp=time.time(),
                    level=level,
                    module=logger_name,
                    function="unknown",
                    line_number=0,
                    execution_time=execution_time,
                    memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                    cpu_usage=psutil.cpu_percent(),
                    thread_id=threading.get_ident(),
                    process_id=os.getpid()
                )
                self.metrics.append(metrics)
                
                # 保持最近1000条记录
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
                    
            except Exception:
                pass  # 忽略监控本身的错误
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self.lock:
            if not self.metrics:
                return {}
            
            exec_times = [m.execution_time for m in self.metrics]
            memory_usage = [m.memory_usage for m in self.metrics]
            
            return {
                "total_logs": len(self.metrics),
                "avg_execution_time": sum(exec_times) / len(exec_times),
                "max_execution_time": max(exec_times),
                "avg_memory_usage": sum(memory_usage) / len(memory_usage),
                "max_memory_usage": max(memory_usage),
                "level_distribution": self._get_level_distribution()
            }
    
    def _get_level_distribution(self) -> Dict[str, int]:
        """获取日志级别分布"""
        distribution = {}
        for metric in self.metrics:
            level = metric.level
            distribution[level] = distribution.get(level, 0) + 1
        return distribution


class EnhancedLoggerSetup:
    """增强版日志系统设置器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.loggers = {}
        self.performance_monitor = PerformanceMonitor()
        self.log_dir = Path(self.config.get('log_directory', 'logs'))
        self.log_dir.mkdir(exist_ok=True)
        
    def create_logger(self, 
                     name: str,
                     level: Union[str, int] = "INFO",
                     log_format: LogFormat = LogFormat.STRUCTURED,
                     enable_console: bool = True,
                     enable_file: bool = True,
                     enable_performance_monitoring: bool = True,
                     max_file_size: int = 10 * 1024 * 1024,  # 10MB
                     backup_count: int = 5,
                     custom_filters: List[logging.Filter] = None) -> logging.Logger:
        """
        创建增强版日志器
        
        Args:
            name: 日志器名称
            level: 日志级别
            log_format: 日志格式
            enable_console: 是否启用控制台输出
            enable_file: 是否启用文件输出
            enable_performance_monitoring: 是否启用性能监控
            max_file_size: 最大文件大小
            backup_count: 备份文件数量
            custom_filters: 自定义过滤器列表
            
        Returns:
            配置好的日志器
        """
        
        # 避免重复创建
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()) if isinstance(level, str) else level)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 创建格式化器
        formatter = StructuredFormatter(log_format)
        
        # 控制台处理器
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.DEBUG)
            logger.addHandler(console_handler)
        
        # 文件处理器
        if enable_file:
            log_file = self.log_dir / f"{name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # 错误单独记录
            error_file = self.log_dir / f"{name}_error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            logger.addHandler(error_handler)
        
        # 添加自定义过滤器
        if custom_filters:
            for handler in logger.handlers:
                for filter_obj in custom_filters:
                    handler.addFilter(filter_obj)
        
        # 性能监控装饰器
        if enable_performance_monitoring:
            logger.performance_monitor = self.performance_monitor
            logger.log_with_metrics = self._create_metrics_decorator(logger)
        
        # 交易特定方法
        logger.log_trade = self._create_trade_logger(logger)
        logger.log_strategy = self._create_strategy_logger(logger)
        logger.log_market_data = self._create_market_data_logger(logger)
        
        self.loggers[name] = logger
        return logger
    
    def _create_metrics_decorator(self, logger: logging.Logger):
        """创建性能监控装饰器"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    logger.debug(
                        f"Function {func.__name__} executed successfully",
                        extra={
                            'execution_time': execution_time,
                            'function_name': func.__name__,
                            'args_count': len(args),
                            'kwargs_count': len(kwargs)
                        }
                    )
                    
                    # 记录性能指标
                    self.performance_monitor.log_performance(
                        logger.name, 'DEBUG', execution_time
                    )
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(
                        f"Function {func.__name__} failed: {str(e)}",
                        exc_info=True,
                        extra={
                            'execution_time': execution_time,
                            'function_name': func.__name__,
                            'error_type': type(e).__name__
                        }
                    )
                    
                    self.performance_monitor.log_performance(
                        logger.name, 'ERROR', execution_time
                    )
                    
                    raise
            return wrapper
        return decorator
    
    def _create_trade_logger(self, logger: logging.Logger):
        """创建交易专用日志方法"""
        def log_trade(action: str, ticker: str, quantity: float, price: float, 
                     strategy: str = None, **kwargs):
            trade_context = {
                'action': action,
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'value': quantity * price,
                'strategy': strategy,
                'timestamp': datetime.now().isoformat()
            }
            trade_context.update(kwargs)
            
            logger.info(
                f"Trade executed: {action} {quantity} {ticker} @ ${price:.2f}",
                extra={'trade': trade_context}
            )
        return log_trade
    
    def _create_strategy_logger(self, logger: logging.Logger):
        """创建策略专用日志方法"""
        def log_strategy(strategy_name: str, signal: str, confidence: float, 
                        ticker: str = None, **kwargs):
            strategy_context = {
                'strategy_name': strategy_name,
                'signal': signal,
                'confidence': confidence,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }
            strategy_context.update(kwargs)
            
            logger.info(
                f"Strategy signal: {strategy_name} -> {signal} (confidence: {confidence:.2f})",
                extra={'strategy': strategy_context}
            )
        return log_strategy
    
    def _create_market_data_logger(self, logger: logging.Logger):
        """创建市场数据专用日志方法"""
        def log_market_data(ticker: str, price: float, volume: int = None, 
                           data_type: str = "quote", **kwargs):
            market_context = {
                'ticker': ticker,
                'price': price,
                'volume': volume,
                'data_type': data_type,
                'timestamp': datetime.now().isoformat()
            }
            market_context.update(kwargs)
            
            logger.debug(
                f"Market data: {ticker} ${price:.2f}",
                extra={'market_data': market_context}
            )
        return log_market_data
    
    def get_logger(self, name: str) -> Optional[logging.Logger]:
        """获取已创建的日志器"""
        return self.loggers.get(name)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_monitor.get_stats()
    
    def create_trading_logger(self, strategy_name: str, config: Dict[str, Any] = None) -> logging.Logger:
        """为交易策略创建专用日志器"""
        logger_config = {
            'level': config.get('LOG_LEVEL', 'INFO') if config else 'INFO',
            'enable_console': config.get('ENABLE_CONSOLE_LOG', True) if config else True,
            'enable_file': True,
            'log_format': LogFormat.JSON if config and config.get('LIVE_TRADING') else LogFormat.STRUCTURED
        }
        
        # 添加交易时间过滤器
        trading_filter = TradingTimeFilter()
        
        logger = self.create_logger(
            name=f"trading.{strategy_name}",
            level=logger_config['level'],
            log_format=logger_config['log_format'],
            enable_console=logger_config['enable_console'],
            enable_file=logger_config['enable_file'],
            custom_filters=[trading_filter]
        )
        
        # 添加策略特定上下文
        logger.strategy_name = strategy_name
        logger.session_id = f"{strategy_name}_{int(time.time())}"
        
        return logger


# 与配置系统集成的便捷函数
def setup_logger_from_config(config: Dict[str, Any], strategy_name: str = "main") -> logging.Logger:
    """从配置创建日志器"""
    logger_setup = EnhancedLoggerSetup(config)
    return logger_setup.create_trading_logger(strategy_name, config)


def get_default_logger(name: str = "trading") -> logging.Logger:
    """获取默认配置的日志器"""
    logger_setup = EnhancedLoggerSetup()
    return logger_setup.create_logger(
        name=name,
        level="INFO",
        log_format=LogFormat.STRUCTURED,
        enable_console=True,
        enable_file=True
    )


# 性能监控上下文管理器
class LoggingContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                extra={**self.context, 'execution_time': execution_time}
            )
        else:
            self.logger.error(
                f"Failed {self.operation}: {exc_val}",
                exc_info=True,
                extra={**self.context, 'execution_time': execution_time}
            )


if __name__ == "__main__":
    # 演示用法
    setup = EnhancedLoggerSetup()
    logger = setup.create_logger("demo", level="DEBUG")
    
    # 基础日志
    logger.info("System started")
    
    # 交易日志
    logger.log_trade("BUY", "AAPL", 100, 150.0, strategy="momentum")
    
    # 策略日志
    logger.log_strategy("RSI_Strategy", "SELL", 0.85, ticker="AAPL")
    
    # 市场数据日志
    logger.log_market_data("AAPL", 151.25, volume=1000000)
    
    # 性能监控
    print("Performance Stats:", setup.get_performance_stats())





