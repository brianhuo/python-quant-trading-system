"""
交易系统数据管道 - 第一部分：数据获取和清洗
整合四个核心模块：配置加载器 + 日志系统 + 统一数据客户端 + 数据健康检查器
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

# 导入核心模块
from enhanced_config_loader import load_config
from logger_config_integration import get_strategy_logger
from adaptive_data_client import AdaptiveDataClient
from enhanced_data_health_checker import EnhancedDataHealthChecker, HealthStatus
from data_cleaner import DataCleaner, CleaningConfig, CleaningMethod


@dataclass
class DataPipelineConfig:
    """数据管道配置"""
    # 数据源配置
    symbol: str = "AAPL"
    timeframe: str = "30min"
    limit: int = 1000
    
    # 质量控制配置
    quality_threshold: HealthStatus = HealthStatus.WARNING
    enable_cleaning: bool = True
    cleaning_mode: str = "standard"  # conservative, standard, strict
    
    # 缓存配置
    enable_cache: bool = True
    cache_expiry_hours: int = 1
    
    # 实时数据配置
    enable_realtime: bool = False
    realtime_callback: Optional[Callable] = None


class TradingDataPipeline:
    """
    交易系统数据管道
    
    核心功能：
    1. 配置管理 - 统一配置加载和管理
    2. 日志系统 - 企业级日志记录
    3. 数据获取 - 历史和实时数据统一接口
    4. 数据清洗 - 智能数据质量检查和清洗
    """
    
    def __init__(self, 
                 config_path: str = None,
                 environment: str = "development",
                 pipeline_config: DataPipelineConfig = None):
        """
        初始化交易数据管道
        
        Args:
            config_path: 配置文件路径
            environment: 运行环境 (development/testing/production)
            pipeline_config: 数据管道专用配置
        """
        self.environment = environment
        self.pipeline_config = pipeline_config or DataPipelineConfig()
        
        # 1. 初始化配置系统
        self.config = load_config(environment=environment)
        
        # 2. 初始化日志系统
        self.logger = get_strategy_logger("trading_data_pipeline")
        self.logger.info(f"交易数据管道初始化 - 环境: {environment}")
        
        # 3. 初始化数据客户端
        self.data_client = AdaptiveDataClient(self.config, self.logger)
        
        # 4. 初始化数据健康检查器
        self.health_checker = EnhancedDataHealthChecker(self.config, self.logger)
        
        # 5. 初始化数据清洗器
        self.data_cleaner = self._create_data_cleaner()
        
        # 状态管理
        self.pipeline_stats = {
            "initialized_at": datetime.now(),
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_rows_processed": 0,
            "total_rows_cleaned": 0,
            "average_processing_time": 0.0
        }
        
        self.logger.info("交易数据管道初始化完成")
    
    def _create_data_cleaner(self) -> DataCleaner:
        """根据配置创建数据清洗器"""
        cleaning_configs = {
            "conservative": CleaningConfig(
                missing_value_method=CleaningMethod.INTERPOLATE,
                outlier_method=CleaningMethod.MEDIAN_FILL,
                negative_volume_method=CleaningMethod.ZERO_FILL,
                remove_invalid_ohlc=False,
                standardize_frequency=True,
                target_frequency=self.pipeline_config.timeframe
            ),
            "standard": CleaningConfig(
                missing_value_method=CleaningMethod.INTERPOLATE,
                outlier_method=CleaningMethod.MEDIAN_FILL,
                negative_volume_method=CleaningMethod.ZERO_FILL,
                remove_invalid_ohlc=True,
                standardize_frequency=True,
                target_frequency=self.pipeline_config.timeframe
            ),
            "strict": CleaningConfig(
                missing_value_method=CleaningMethod.DROP,
                outlier_method=CleaningMethod.DROP,
                negative_volume_method=CleaningMethod.DROP,
                remove_invalid_ohlc=True,
                standardize_frequency=True,
                target_frequency=self.pipeline_config.timeframe
            )
        }
        
        config = cleaning_configs.get(self.pipeline_config.cleaning_mode, cleaning_configs["standard"])
        return DataCleaner(config, self.logger)
    
    def get_clean_data(self, 
                      symbol: str = None,
                      timeframe: str = None,
                      limit: int = None,
                      force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        获取清洗后的数据 - 核心方法
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            limit: 数据数量限制
            force_refresh: 强制刷新缓存
            
        Returns:
            (清洗后数据, 处理报告)
        """
        # 使用配置默认值
        symbol = symbol or self.pipeline_config.symbol
        timeframe = timeframe or self.pipeline_config.timeframe
        limit = limit or self.pipeline_config.limit
        
        start_time = datetime.now()
        self.pipeline_stats["total_requests"] += 1
        
        try:
            self.logger.info(f"开始获取数据: {symbol} {timeframe} limit={limit}")
            
            # 步骤1: 获取原始数据
            raw_data = self.data_client.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            if raw_data.empty:
                self.logger.error(f"无法获取 {symbol} 的数据")
                self.pipeline_stats["failed_requests"] += 1
                return pd.DataFrame(), {
                    "status": "failed",
                    "error": "No data available",
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            
            self.logger.info(f"获取原始数据: {raw_data.shape[0]} 行")
            original_rows = raw_data.shape[0]
            
            # 步骤2: 数据质量检查
            health_report = self.health_checker.comprehensive_health_check(
                raw_data,
                clean_data=False,
                save_report=False
            )
            
            self.logger.info(f"数据健康检查完成: {health_report.status.value} - {len(health_report.issues)} 个问题")
            
            # 步骤3: 质量门控
            if health_report.status == HealthStatus.FAILED:
                self.logger.error("数据质量检查失败")
                self.pipeline_stats["failed_requests"] += 1
                return pd.DataFrame(), {
                    "status": "failed",
                    "error": "Data quality check failed",
                    "health_report": health_report.get_summary(),
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            
            # 步骤4: 数据清洗 (如果启用)
            if self.pipeline_config.enable_cleaning:
                cleaned_data, cleaning_log = self.data_cleaner.comprehensive_clean(raw_data)
                self.logger.info(f"数据清洗完成: {raw_data.shape[0]} → {cleaned_data.shape[0]} 行")
            else:
                cleaned_data = raw_data
                cleaning_log = []
                self.logger.info("跳过数据清洗")
            
            # 步骤5: 清洗后质量验证
            final_health_report = self.health_checker.comprehensive_health_check(
                cleaned_data,
                clean_data=False,
                save_report=False
            )
            
            # 步骤6: 最终质量门控
            if final_health_report.status.value > self.pipeline_config.quality_threshold.value:
                self.logger.warning(f"清洗后数据质量仍不达标: {final_health_report.status.value}")
            
            # 更新统计信息
            processing_time = (datetime.now() - start_time).total_seconds()
            self.pipeline_stats["successful_requests"] += 1
            self.pipeline_stats["total_rows_processed"] += original_rows
            self.pipeline_stats["total_rows_cleaned"] += cleaned_data.shape[0]
            
            # 更新平均处理时间
            total_successful = self.pipeline_stats["successful_requests"]
            current_avg = self.pipeline_stats["average_processing_time"]
            self.pipeline_stats["average_processing_time"] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
            
            # 生成处理报告
            processing_report = {
                "status": "success",
                "symbol": symbol,
                "timeframe": timeframe,
                "processing_time_seconds": processing_time,
                "data_flow": {
                    "original_rows": original_rows,
                    "cleaned_rows": cleaned_data.shape[0],
                    "rows_removed": original_rows - cleaned_data.shape[0],
                    "removal_percentage": ((original_rows - cleaned_data.shape[0]) / original_rows) * 100
                },
                "quality_assessment": {
                    "original_status": health_report.status.value,
                    "original_issues": len(health_report.issues),
                    "final_status": final_health_report.status.value,
                    "final_issues": len(final_health_report.issues),
                    "quality_improved": len(health_report.issues) > len(final_health_report.issues)
                },
                "cleaning_summary": {
                    "enabled": self.pipeline_config.enable_cleaning,
                    "mode": self.pipeline_config.cleaning_mode,
                    "operations": len(cleaning_log),
                    "details": cleaning_log[:5] if cleaning_log else []  # 前5个操作
                },
                "data_characteristics": {
                    "timespan": {
                        "start": cleaned_data.index[0].isoformat() if not cleaned_data.empty else None,
                        "end": cleaned_data.index[-1].isoformat() if not cleaned_data.empty else None,
                        "duration": str(cleaned_data.index[-1] - cleaned_data.index[0]) if len(cleaned_data) > 1 else None
                    },
                    "statistics": final_health_report.statistics.get("financial", {})
                }
            }
            
            self.logger.info(f"数据管道处理完成: {processing_report['status']}")
            return cleaned_data, processing_report
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.pipeline_stats["failed_requests"] += 1
            
            error_report = {
                "status": "error",
                "error_message": str(e),
                "error_type": type(e).__name__,
                "symbol": symbol,
                "timeframe": timeframe,
                "processing_time_seconds": processing_time
            }
            
            self.logger.error(f"数据管道处理失败: {e}")
            return pd.DataFrame(), error_report
    
    def get_realtime_data(self, 
                         symbol: str = None,
                         callback: Callable = None) -> bool:
        """
        启动实时数据订阅
        
        Args:
            symbol: 交易对符号
            callback: 数据回调函数
            
        Returns:
            是否成功启动订阅
        """
        symbol = symbol or self.pipeline_config.symbol
        callback = callback or self.pipeline_config.realtime_callback
        
        if not self.pipeline_config.enable_realtime:
            self.logger.warning("实时数据功能未启用")
            return False
        
        if not callback:
            self.logger.error("实时数据需要提供回调函数")
            return False
        
        try:
            # 创建包装回调，加入数据质量检查
            def quality_checked_callback(market_data):
                try:
                    # 这里可以添加实时数据的质量检查逻辑
                    self.logger.debug(f"实时数据: {market_data.symbol} = ${market_data.price}")
                    callback(market_data)
                except Exception as e:
                    self.logger.error(f"实时数据回调错误: {e}")
            
            success = self.data_client.subscribe_realtime(symbol, quality_checked_callback)
            
            if success:
                self.logger.info(f"实时数据订阅成功: {symbol}")
            else:
                self.logger.error(f"实时数据订阅失败: {symbol}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"实时数据订阅异常: {e}")
            return False
    
    def batch_process_symbols(self, 
                            symbols: List[str],
                            timeframe: str = None,
                            limit: int = None) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """
        批量处理多个交易对
        
        Args:
            symbols: 交易对列表
            timeframe: 时间框架
            limit: 数据数量限制
            
        Returns:
            {symbol: (cleaned_data, report)}
        """
        self.logger.info(f"开始批量处理 {len(symbols)} 个交易对")
        
        results = {}
        successful_count = 0
        
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"处理进度: {i}/{len(symbols)} - {symbol}")
            
            try:
                cleaned_data, report = self.get_clean_data(symbol, timeframe, limit)
                results[symbol] = (cleaned_data, report)
                
                if report["status"] == "success":
                    successful_count += 1
                    self.logger.info(f"{symbol}: ✅ 成功 - {cleaned_data.shape[0]} 行")
                else:
                    self.logger.error(f"{symbol}: ❌ 失败 - {report.get('error', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.error(f"{symbol}: ❌ 异常 - {e}")
                results[symbol] = (pd.DataFrame(), {
                    "status": "error",
                    "error_message": str(e),
                    "symbol": symbol
                })
        
        self.logger.info(f"批量处理完成: {successful_count}/{len(symbols)} 成功")
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取管道状态"""
        return {
            "pipeline_info": {
                "environment": self.environment,
                "initialized_at": self.pipeline_stats["initialized_at"].isoformat(),
                "uptime_seconds": (datetime.now() - self.pipeline_stats["initialized_at"]).total_seconds()
            },
            "configuration": {
                "symbol": self.pipeline_config.symbol,
                "timeframe": self.pipeline_config.timeframe,
                "cleaning_mode": self.pipeline_config.cleaning_mode,
                "quality_threshold": self.pipeline_config.quality_threshold.value,
                "cache_enabled": self.pipeline_config.enable_cache,
                "realtime_enabled": self.pipeline_config.enable_realtime
            },
            "statistics": self.pipeline_stats,
            "performance": {
                "success_rate": (self.pipeline_stats["successful_requests"] / max(self.pipeline_stats["total_requests"], 1)) * 100,
                "data_retention_rate": (self.pipeline_stats["total_rows_cleaned"] / max(self.pipeline_stats["total_rows_processed"], 1)) * 100,
                "average_processing_time": self.pipeline_stats["average_processing_time"]
            },
            "component_status": {
                "config_loader": "active",
                "logger": "active", 
                "data_client": self.data_client.get_status(),
                "health_checker": "active",
                "data_cleaner": "active"
            }
        }
    
    def save_pipeline_report(self, file_path: str = None) -> str:
        """保存管道状态报告"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"logs/trading_pipeline_report_{timestamp}.json"
        
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        status = self.get_pipeline_status()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"管道状态报告已保存: {file_path}")
        return file_path
    
    def shutdown(self):
        """优雅关闭管道"""
        self.logger.info("开始关闭交易数据管道...")
        
        try:
            # 关闭数据客户端
            if hasattr(self.data_client, 'close'):
                self.data_client.close()
            
            # 保存最终报告
            self.save_pipeline_report()
            
            self.logger.info("交易数据管道已优雅关闭")
            
        except Exception as e:
            self.logger.error(f"管道关闭过程中出现错误: {e}")


def create_default_pipeline() -> TradingDataPipeline:
    """创建默认配置的数据管道"""
    default_config = DataPipelineConfig(
        symbol="AAPL",
        timeframe="30min",
        limit=1000,
        quality_threshold=HealthStatus.WARNING,
        enable_cleaning=True,
        cleaning_mode="standard",
        enable_cache=True,
        enable_realtime=False
    )
    
    return TradingDataPipeline(
        environment="development",
        pipeline_config=default_config
    )


if __name__ == "__main__":
    # 演示用法
    print("🚀 交易系统数据管道演示")
    print("=" * 70)
    
    # 创建数据管道
    pipeline = create_default_pipeline()
    
    # 获取单个股票数据
    print("\n📊 获取单个股票数据...")
    data, report = pipeline.get_clean_data("AAPL", "30min", 100)
    
    if not data.empty:
        print(f"✅ 成功获取数据: {data.shape[0]} 行")
        print(f"📈 数据时间范围: {data.index[0]} 到 {data.index[-1]}")
        print(f"🔍 处理状态: {report['status']}")
        print(f"⏱️ 处理时间: {report['processing_time_seconds']:.4f} 秒")
        
        print(f"\n📋 数据预览:")
        print(data.head(3).to_string())
    else:
        print(f"❌ 数据获取失败: {report.get('error', 'Unknown error')}")
    
    # 批量处理示例
    print(f"\n📈 批量处理示例...")
    symbols = ["AAPL", "GOOGL"]
    batch_results = pipeline.batch_process_symbols(symbols, "30min", 50)
    
    for symbol, (data, report) in batch_results.items():
        status = "✅" if report["status"] == "success" else "❌"
        rows = data.shape[0] if not data.empty else 0
        print(f"   {symbol}: {status} {rows} 行")
    
    # 显示管道状态
    print(f"\n📊 管道状态:")
    status = pipeline.get_pipeline_status()
    print(f"   成功率: {status['performance']['success_rate']:.1f}%")
    print(f"   数据保留率: {status['performance']['data_retention_rate']:.1f}%")
    print(f"   平均处理时间: {status['performance']['average_processing_time']:.4f} 秒")
    
    # 优雅关闭
    pipeline.shutdown()
    
    print(f"\n🎉 演示完成！您现在有了一个完整的交易系统数据获取和清洗模块！")
