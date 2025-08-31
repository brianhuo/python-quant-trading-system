"""
数据健康检查器集成示例
展示如何在交易系统中使用增强版数据健康检查器
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

from enhanced_data_health_checker import EnhancedDataHealthChecker, HealthStatus
from data_cleaner import DataCleaner, CleaningConfig, CleaningMethod
from adaptive_data_client import AdaptiveDataClient
from enhanced_config_loader import load_config
from logger_config_integration import get_strategy_logger


class TradingDataPipeline:
    """
    交易数据处理管道
    集成数据获取、健康检查、清洗和验证
    """
    
    def __init__(self):
        """初始化数据处理管道"""
        self.config = load_config()
        self.logger = get_strategy_logger("trading_data_pipeline")
        
        # 初始化组件
        self.data_client = AdaptiveDataClient()
        self.health_checker = EnhancedDataHealthChecker()
        
        # 配置清洗策略
        self.cleaning_config = CleaningConfig(
            missing_value_method=CleaningMethod.INTERPOLATE,
            outlier_method=CleaningMethod.MEDIAN_FILL,
            negative_volume_method=CleaningMethod.ZERO_FILL,
            remove_invalid_ohlc=True,
            standardize_frequency=True,
            target_frequency=self.config.get('DATA_TIMEFRAME', '30min')
        )
        self.data_cleaner = DataCleaner(self.cleaning_config)
        
        self.logger.info("交易数据处理管道初始化完成")
    
    def get_clean_data(self, 
                      symbol: str, 
                      timeframe: str = None, 
                      limit: int = 1000,
                      quality_threshold: HealthStatus = HealthStatus.WARNING) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        获取清洗后的交易数据
        
        Args:
            symbol: 股票代码
            timeframe: 时间框架
            limit: 数据数量限制
            quality_threshold: 数据质量阈值
            
        Returns:
            (清洗后数据, 处理报告)
        """
        timeframe = timeframe or self.config.get('DATA_TIMEFRAME', '30min')
        
        self.logger.info(f"开始处理 {symbol} {timeframe} 数据...")
        
        # 1. 获取原始数据
        raw_data = self.data_client.get_historical_data(
            symbol=symbol, 
            timeframe=timeframe, 
            limit=limit
        )
        
        if raw_data.empty:
            self.logger.error(f"无法获取 {symbol} 的数据")
            return pd.DataFrame(), {"error": "No data available"}
        
        self.logger.info(f"获取到 {len(raw_data)} 条原始数据")
        
        # 2. 执行健康检查
        health_report = self.health_checker.comprehensive_health_check(
            raw_data, 
            clean_data=False, 
            save_report=False
        )
        
        # 3. 评估数据质量
        if health_report.status == HealthStatus.FAILED:
            self.logger.error("数据质量检查失败，无法处理")
            return pd.DataFrame(), {
                "error": "Data quality check failed",
                "health_report": health_report.get_summary()
            }
        
        # 4. 数据清洗
        cleaned_data, cleaning_log = self.data_cleaner.comprehensive_clean(raw_data)
        
        # 5. 清洗后验证
        final_health_report = self.health_checker.comprehensive_health_check(
            cleaned_data, 
            clean_data=False, 
            save_report=False
        )
        
        # 6. 质量门控
        if final_health_report.status.value > quality_threshold.value:
            self.logger.warning(f"清洗后数据质量仍不达标: {final_health_report.status.value}")
        
        # 7. 生成处理报告
        processing_report = {
            "symbol": symbol,
            "timeframe": timeframe,
            "original_shape": raw_data.shape,
            "cleaned_shape": cleaned_data.shape,
            "data_reduction": {
                "rows_removed": raw_data.shape[0] - cleaned_data.shape[0],
                "removal_percentage": ((raw_data.shape[0] - cleaned_data.shape[0]) / raw_data.shape[0]) * 100
            },
            "original_health": health_report.get_summary(),
            "final_health": final_health_report.get_summary(),
            "cleaning_summary": self.data_cleaner.get_cleaning_summary(),
            "quality_passed": final_health_report.status.value <= quality_threshold.value
        }
        
        self.logger.info(f"数据处理完成: {raw_data.shape} → {cleaned_data.shape}")
        
        return cleaned_data, processing_report
    
    def batch_process_symbols(self, 
                            symbols: list, 
                            timeframe: str = None, 
                            limit: int = 1000) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """
        批量处理多个股票的数据
        
        Args:
            symbols: 股票代码列表
            timeframe: 时间框架
            limit: 数据数量限制
            
        Returns:
            {symbol: (cleaned_data, report)}
        """
        results = {}
        
        for symbol in symbols:
            try:
                cleaned_data, report = self.get_clean_data(symbol, timeframe, limit)
                results[symbol] = (cleaned_data, report)
                
                # 简要日志
                if not cleaned_data.empty:
                    status = "✅ 成功" if report["quality_passed"] else "⚠️ 警告"
                    self.logger.info(f"{symbol}: {status} - {cleaned_data.shape[0]} 行数据")
                else:
                    self.logger.error(f"{symbol}: ❌ 失败")
                    
            except Exception as e:
                self.logger.error(f"{symbol} 处理失败: {e}")
                results[symbol] = (pd.DataFrame(), {"error": str(e)})
        
        return results
    
    def get_data_quality_summary(self, processing_results: Dict[str, Tuple[pd.DataFrame, Dict]]) -> Dict[str, Any]:
        """
        生成数据质量摘要报告
        
        Args:
            processing_results: 批量处理结果
            
        Returns:
            质量摘要报告
        """
        total_symbols = len(processing_results)
        successful_symbols = 0
        total_original_rows = 0
        total_cleaned_rows = 0
        quality_issues = []
        
        for symbol, (data, report) in processing_results.items():
            if not data.empty and "error" not in report:
                successful_symbols += 1
                total_original_rows += report["original_shape"][0]
                total_cleaned_rows += report["cleaned_shape"][0]
                
                # 收集质量问题
                if not report["quality_passed"]:
                    quality_issues.append({
                        "symbol": symbol,
                        "issues": report["final_health"]["total_issues"],
                        "status": report["final_health"]["overall_status"]
                    })
        
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "total_symbols": total_symbols,
            "successful_symbols": successful_symbols,
            "success_rate": (successful_symbols / total_symbols) * 100 if total_symbols > 0 else 0,
            "data_statistics": {
                "total_original_rows": total_original_rows,
                "total_cleaned_rows": total_cleaned_rows,
                "total_rows_removed": total_original_rows - total_cleaned_rows,
                "overall_data_retention": (total_cleaned_rows / total_original_rows) * 100 if total_original_rows > 0 else 0
            },
            "quality_issues": quality_issues,
            "symbols_with_issues": len(quality_issues)
        }
        
        return summary


def demo_trading_data_pipeline():
    """演示交易数据处理管道"""
    print("🔄 交易数据处理管道演示")
    print("=" * 60)
    
    # 创建数据处理管道
    pipeline = TradingDataPipeline()
    
    # 单个股票处理示例
    print("\n📊 单个股票数据处理演示...")
    cleaned_data, report = pipeline.get_clean_data("AAPL", "30min", 100)
    
    if not cleaned_data.empty:
        print(f"✅ AAPL 数据处理成功:")
        print(f"   原始数据: {report['original_shape'][0]} 行")
        print(f"   清洗后: {report['cleaned_shape'][0]} 行")
        print(f"   数据保留率: {100 - report['data_reduction']['removal_percentage']:.1f}%")
        print(f"   质量状态: {report['final_health']['overall_status']}")
        
        print(f"\n📋 数据预览:")
        print(cleaned_data.head())
    else:
        print(f"❌ AAPL 数据处理失败")
    
    # 批量处理示例
    print(f"\n📈 批量处理演示...")
    symbols = ["AAPL", "GOOGL", "MSFT"]
    batch_results = pipeline.batch_process_symbols(symbols, "30min", 50)
    
    # 质量摘要
    quality_summary = pipeline.get_data_quality_summary(batch_results)
    
    print(f"\n📊 批量处理质量摘要:")
    print(f"   处理股票数: {quality_summary['total_symbols']}")
    print(f"   成功率: {quality_summary['success_rate']:.1f}%")
    print(f"   总数据保留率: {quality_summary['data_statistics']['overall_data_retention']:.1f}%")
    print(f"   有质量问题的股票: {quality_summary['symbols_with_issues']}")
    
    if quality_summary['quality_issues']:
        print(f"\n⚠️ 质量问题详情:")
        for issue in quality_summary['quality_issues']:
            print(f"   {issue['symbol']}: {issue['issues']} 个问题 ({issue['status']})")


def demo_custom_quality_pipeline():
    """演示自定义质量管道"""
    print("\n🛠️ 自定义质量管道演示")
    print("=" * 60)
    
    # 创建自定义清洗配置
    strict_config = CleaningConfig(
        missing_value_method=CleaningMethod.DROP,  # 严格模式：删除缺失值
        outlier_method=CleaningMethod.DROP,        # 严格模式：删除异常值
        remove_invalid_ohlc=True,
        standardize_frequency=True,
        target_frequency="1h"  # 标准化为1小时
    )
    
    # 创建自定义管道
    pipeline = TradingDataPipeline()
    pipeline.data_cleaner = DataCleaner(strict_config)
    
    print("🔧 使用严格清洗模式...")
    cleaned_data, report = pipeline.get_clean_data("AAPL", "30min", 100)
    
    if not cleaned_data.empty:
        print(f"✅ 严格模式处理结果:")
        print(f"   数据保留率: {100 - report['data_reduction']['removal_percentage']:.1f}%")
        print(f"   最终质量: {report['final_health']['overall_status']}")
        print(f"   清洗操作: {report['cleaning_summary']['total_actions']} 次")
    
    print("\n✨ 自定义管道演示完成")


if __name__ == "__main__":
    # 运行演示
    demo_trading_data_pipeline()
    demo_custom_quality_pipeline()
    
    print("\n" + "=" * 60)
    print("🎉 增强版数据健康检查器集成演示完成！")
    print("\n💡 主要特性:")
    print("✅ 自动数据获取与清洗")
    print("✅ 多级质量检查与验证") 
    print("✅ 批量处理与质量监控")
    print("✅ 可配置的清洗策略")
    print("✅ 详细的处理报告")
    print("✅ 企业级错误处理")
    print("\n🚀 现在您有了一个完整的数据质量管理解决方案！")

