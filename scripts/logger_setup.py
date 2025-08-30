# logger_setup.py
import logging
from logging.handlers import RotatingFileHandler
import functools
import time
import os

class DynamicLevelFilter(logging.Filter):
    """
    动态日志级别过滤器：根据交易时段调整日志级别。
    示例：交易时段（9:00-17:00）仅记录 ERROR 级别，非交易时段记录所有级别。
    """
    def filter(self, record):
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour < 17:  # 交易时段
            return record.levelno >= logging.ERROR
        return True

class SafeFormatter(logging.Formatter):
    """
    自定义 Formatter 子类：当缺失 'latency' 字段时，使用默认值 0.000 填充。
    """
    def format(self, record):
        if not hasattr(record, 'latency'):
            record.latency = 0.000  # 默认值
        return super().format(record)

def setup_logging(env='prod'):
    """设置优化后的日志系统，避免重复初始化"""
    logger = logging.getLogger("OptimizedTrading")
    if logger.handlers:  # 避免重复初始化
        return logger

    # 获取环境变量 TRADING_ENV，默认为 'prod'
    env = os.getenv('TRADING_ENV', env)

    # 动态日志级别
    level = logging.DEBUG if env == 'dev' else logging.INFO
    logger.setLevel(level)

    # 日志格式（包含 latency 字段）
    formatter = SafeFormatter(
        "%(asctime)s|%(module)s|%(levelname)s|latency=%(latency).3fs|%(message)s"
    )
    
    # 日志轮转文件处理器
    file_handler = RotatingFileHandler(
        'trading.log', maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # 生产环境禁用控制台输出
    if env == 'prod':
        stream_handler = logging.NullHandler()
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

    # 添加动态级别过滤器（仅生产环境启用）
    if env == 'prod':
        dynamic_filter = DynamicLevelFilter()
        file_handler.addFilter(dynamic_filter)
        stream_handler.addFilter(dynamic_filter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # 异常捕获装饰器
    def exception_logger(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start
                logger.debug(f"{func.__name__} executed successfully", extra={'latency': latency})
                return result
            except Exception as e:
                latency = time.time() - start
                logger.error(f"Error in {func.__name__}: {str(e)}", 
                            exc_info=True, extra={'latency': latency})
                raise
        return wrapper

    logger.exception_logger = exception_logger
    return logger

# 移除自动初始化，改为按需调用
# setup_logging()