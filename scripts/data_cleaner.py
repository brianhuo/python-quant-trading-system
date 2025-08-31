"""
数据清洗工具
提供专业的数据清洗和预处理功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from enhanced_data_health_checker import HealthReport, EnhancedDataHealthChecker
from logger_config_integration import get_strategy_logger


class CleaningMethod(Enum):
    """清洗方法枚举"""
    DROP = "drop"
    INTERPOLATE = "interpolate"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    MEDIAN_FILL = "median_fill"
    MEAN_FILL = "mean_fill"
    ZERO_FILL = "zero_fill"


@dataclass
class CleaningConfig:
    """清洗配置"""
    missing_value_method: CleaningMethod = CleaningMethod.INTERPOLATE
    outlier_method: CleaningMethod = CleaningMethod.MEDIAN_FILL
    negative_volume_method: CleaningMethod = CleaningMethod.ZERO_FILL
    remove_invalid_ohlc: bool = True
    fill_time_gaps: bool = True
    standardize_frequency: bool = True
    target_frequency: str = "30min"


class DataCleaner:
    """
    专业数据清洗器
    与EnhancedDataHealthChecker配合使用
    """
    
    def __init__(self, config: CleaningConfig = None, logger=None):
        """
        初始化数据清洗器
        
        Args:
            config: 清洗配置
            logger: 日志器
        """
        self.config = config or CleaningConfig()
        self.logger = logger or get_strategy_logger("data_cleaner")
        self.cleaning_log = []
        
    def log_action(self, action: str, details: str, affected_rows: int = 0):
        """记录清洗动作"""
        log_entry = {
            'timestamp': datetime.now(),
            'action': action,
            'details': details,
            'affected_rows': affected_rows
        }
        self.cleaning_log.append(log_entry)
        self.logger.info(f"数据清洗: {action} - {details} (影响 {affected_rows} 行)")
    
    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗缺失值
        
        Args:
            df: 输入数据框
            
        Returns:
            清洗后的数据框
        """
        original_shape = df.shape
        
        if self.config.missing_value_method == CleaningMethod.DROP:
            df = df.dropna()
            self.log_action("删除缺失值", "删除包含缺失值的行", original_shape[0] - df.shape[0])
        
        elif self.config.missing_value_method == CleaningMethod.INTERPOLATE:
            # 对价格数据使用线性插值
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        df[col] = df[col].interpolate(method='linear')
                        self.log_action(f"插值填充-{col}", f"线性插值填充缺失值", missing_count)
            
            # 成交量用0填充
            if 'volume' in df.columns:
                volume_missing = df['volume'].isnull().sum()
                if volume_missing > 0:
                    df['volume'] = df['volume'].fillna(0)
                    self.log_action("成交量填充", "用0填充缺失的成交量", volume_missing)
        
        elif self.config.missing_value_method == CleaningMethod.FORWARD_FILL:
            missing_count = df.isnull().sum().sum()
            df = df.fillna(method='ffill')
            self.log_action("向前填充", "使用前一个有效值填充", missing_count)
        
        elif self.config.missing_value_method == CleaningMethod.MEDIAN_FILL:
            for col in df.select_dtypes(include=[np.number]).columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    self.log_action(f"中位数填充-{col}", f"用中位数 {median_val:.4f} 填充", missing_count)
        
        return df
    
    def clean_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗异常值
        
        Args:
            df: 输入数据框
            
        Returns:
            清洗后的数据框
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 使用IQR方法识别异常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                if self.config.outlier_method == CleaningMethod.DROP:
                    df = df[~outliers]
                    self.log_action(f"删除异常值-{col}", f"删除异常值范围外的数据", outlier_count)
                
                elif self.config.outlier_method == CleaningMethod.MEDIAN_FILL:
                    median_val = df[col].median()
                    df.loc[outliers, col] = median_val
                    self.log_action(f"中位数替换-{col}", f"用中位数 {median_val:.4f} 替换异常值", outlier_count)
                
                elif self.config.outlier_method == CleaningMethod.INTERPOLATE:
                    # 将异常值设为NaN然后插值
                    df.loc[outliers, col] = np.nan
                    df[col] = df[col].interpolate()
                    self.log_action(f"插值替换-{col}", f"异常值插值替换", outlier_count)
        
        return df
    
    def clean_price_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗价格逻辑错误
        
        Args:
            df: 输入数据框
            
        Returns:
            清洗后的数据框
        """
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return df
        
        original_len = len(df)
        
        if self.config.remove_invalid_ohlc:
            # 检查OHLC逻辑
            valid_high = (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['high'] >= df['low'])
            valid_low = (df['low'] <= df['open']) & (df['low'] <= df['close']) & (df['low'] <= df['high'])
            valid_prices = (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)
            
            valid_rows = valid_high & valid_low & valid_prices
            df = df[valid_rows]
            
            removed_rows = original_len - len(df)
            if removed_rows > 0:
                self.log_action("删除无效OHLC", "删除价格逻辑错误的行", removed_rows)
        
        return df
    
    def clean_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗成交量异常
        
        Args:
            df: 输入数据框
            
        Returns:
            清洗后的数据框
        """
        if 'volume' not in df.columns:
            return df
        
        # 处理负成交量
        negative_volume = df['volume'] < 0
        negative_count = negative_volume.sum()
        
        if negative_count > 0:
            if self.config.negative_volume_method == CleaningMethod.ZERO_FILL:
                df.loc[negative_volume, 'volume'] = 0
                self.log_action("负成交量处理", "将负成交量设为0", negative_count)
            elif self.config.negative_volume_method == CleaningMethod.DROP:
                df = df[~negative_volume]
                self.log_action("删除负成交量", "删除负成交量的行", negative_count)
        
        return df
    
    def standardize_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据频率
        
        Args:
            df: 输入数据框
            
        Returns:
            标准化后的数据框
        """
        if not isinstance(df.index, pd.DatetimeIndex) or not self.config.standardize_frequency:
            return df
        
        original_len = len(df)
        
        try:
            # 重新采样到目标频率
            df_resampled = df.resample(self.config.target_frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            new_len = len(df_resampled)
            self.log_action("频率标准化", f"重新采样到 {self.config.target_frequency}", 
                          abs(original_len - new_len))
            
            return df_resampled
            
        except Exception as e:
            self.logger.warning(f"频率标准化失败: {e}")
            return df
    
    def fill_time_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        填充时间间隙
        
        Args:
            df: 输入数据框
            
        Returns:
            填充后的数据框
        """
        if not isinstance(df.index, pd.DatetimeIndex) or not self.config.fill_time_gaps:
            return df
        
        original_len = len(df)
        
        try:
            # 创建完整的时间索引
            full_index = pd.date_range(
                start=df.index[0], 
                end=df.index[-1], 
                freq=self.config.target_frequency
            )
            
            # 重新索引并向前填充
            df_filled = df.reindex(full_index).ffill()
            
            filled_rows = len(df_filled) - original_len
            if filled_rows > 0:
                self.log_action("填充时间间隙", f"填充 {filled_rows} 个时间点", filled_rows)
            
            return df_filled
            
        except Exception as e:
            self.logger.warning(f"时间间隙填充失败: {e}")
            return df
    
    def comprehensive_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        执行全面数据清洗
        
        Args:
            df: 输入数据框
            
        Returns:
            清洗后的数据框和清洗日志
        """
        self.logger.info("开始全面数据清洗...")
        self.cleaning_log.clear()
        
        original_shape = df.shape
        self.log_action("开始清洗", f"原始数据形状: {original_shape}", 0)
        
        # 执行清洗步骤
        df = self.clean_missing_values(df)
        df = self.clean_price_logic(df)
        df = self.clean_volume_anomalies(df)
        df = self.clean_outliers(df)
        
        # 时间相关清洗
        if isinstance(df.index, pd.DatetimeIndex):
            df = self.standardize_frequency(df)
            df = self.fill_time_gaps(df)
        
        final_shape = df.shape
        total_removed = original_shape[0] - final_shape[0]
        self.log_action("清洗完成", f"最终数据形状: {final_shape}", total_removed)
        
        self.logger.info(f"数据清洗完成: {original_shape} → {final_shape}")
        
        return df, self.cleaning_log.copy()
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        获取清洗摘要
        
        Returns:
            清洗摘要字典
        """
        if not self.cleaning_log:
            return {}
        
        total_affected = sum(entry['affected_rows'] for entry in self.cleaning_log)
        action_counts = {}
        
        for entry in self.cleaning_log:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_actions': len(self.cleaning_log),
            'total_affected_rows': total_affected,
            'action_breakdown': action_counts,
            'start_time': self.cleaning_log[0]['timestamp'].isoformat() if self.cleaning_log else None,
            'end_time': self.cleaning_log[-1]['timestamp'].isoformat() if self.cleaning_log else None,
            'detailed_log': self.cleaning_log
        }


def create_integrated_data_processor():
    """
    创建集成的数据处理器（健康检查 + 清洗）
    
    Returns:
        集成处理函数
    """
    def process_data(df: pd.DataFrame, 
                    cleaning_config: CleaningConfig = None,
                    save_reports: bool = True) -> Tuple[pd.DataFrame, HealthReport, List[Dict]]:
        """
        集成数据处理流程
        
        Args:
            df: 输入数据
            cleaning_config: 清洗配置
            save_reports: 是否保存报告
            
        Returns:
            (清洗后数据, 健康报告, 清洗日志)
        """
        # 健康检查
        health_checker = EnhancedDataHealthChecker()
        health_report = health_checker.comprehensive_health_check(
            df, clean_data=False, save_report=save_reports
        )
        
        # 数据清洗
        cleaner = DataCleaner(cleaning_config)
        cleaned_df, cleaning_log = cleaner.comprehensive_clean(df)
        
        # 对清洗后数据再次检查
        final_report = health_checker.comprehensive_health_check(
            cleaned_df, clean_data=False, save_report=save_reports
        )
        
        return cleaned_df, final_report, cleaning_log
    
    return process_data


if __name__ == "__main__":
    # 演示数据清洗器
    print("🧹 数据清洗器演示")
    print("=" * 50)
    
    # 创建测试数据
    from mock_data_generator import MockDataGenerator
    
    generator = MockDataGenerator()
    df = generator.generate_historical_data("TEST", "30min", records=100)
    
    # 人工添加问题
    print("📊 添加数据问题用于演示...")
    df.iloc[10:15, 1] = np.nan  # 缺失值
    df.iloc[20, 3] = df.iloc[20, 3] * 20  # 异常值
    df.iloc[30:32, :] = 0  # 零值
    df.loc[df.index[5], 'volume'] = -1000  # 负成交量
    
    print(f"原始数据形状: {df.shape}")
    print(f"缺失值数量: {df.isnull().sum().sum()}")
    
    # 创建清洗配置
    config = CleaningConfig(
        missing_value_method=CleaningMethod.INTERPOLATE,
        outlier_method=CleaningMethod.MEDIAN_FILL,
        negative_volume_method=CleaningMethod.ZERO_FILL,
        remove_invalid_ohlc=True,
        standardize_frequency=True,
        target_frequency="30min"
    )
    
    # 执行清洗
    cleaner = DataCleaner(config)
    cleaned_df, cleaning_log = cleaner.comprehensive_clean(df)
    
    print(f"\n清洗后数据形状: {cleaned_df.shape}")
    print(f"清洗后缺失值: {cleaned_df.isnull().sum().sum()}")
    
    # 显示清洗摘要
    summary = cleaner.get_cleaning_summary()
    print(f"\n🧹 清洗摘要:")
    print(f"总操作数: {summary['total_actions']}")
    print(f"影响行数: {summary['total_affected_rows']}")
    print(f"操作类型: {list(summary['action_breakdown'].keys())}")
    
    print("\n✅ 数据清洗演示完成！")
