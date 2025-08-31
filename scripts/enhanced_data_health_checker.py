"""
增强版数据健康检查器
提供全面的数据质量验证和清洗功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import json
from pathlib import Path

# 导入我们的增强模块
from enhanced_config_loader import load_config
from logger_config_integration import get_strategy_logger


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class IssueType(Enum):
    """问题类型枚举"""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    TIME_CONTINUITY = "time_continuity"
    FREQUENCY_CONSISTENCY = "frequency_consistency"
    DATA_RANGE = "data_range"
    VOLUME_ANOMALY = "volume_anomaly"
    PRICE_ANOMALY = "price_anomaly"
    STATISTICAL = "statistical"


@dataclass
class HealthIssue:
    """健康检查问题记录"""
    issue_type: IssueType
    severity: HealthStatus
    column: str
    description: str
    value: Any = None
    suggestion: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'issue_type': self.issue_type.value,
            'severity': self.severity.value,
            'column': self.column,
            'description': self.description,
            'value': self.value,
            'suggestion': self.suggestion,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class HealthReport:
    """健康检查报告"""
    status: HealthStatus
    issues: List[HealthIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    cleaned_data: Optional[pd.DataFrame] = None
    original_shape: Tuple[int, int] = (0, 0)
    cleaned_shape: Tuple[int, int] = (0, 0)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_issue(self, issue: HealthIssue):
        """添加问题"""
        self.issues.append(issue)
        # 更新整体状态
        if issue.severity == HealthStatus.CRITICAL:
            self.status = HealthStatus.CRITICAL
        elif issue.severity == HealthStatus.WARNING and self.status == HealthStatus.HEALTHY:
            self.status = HealthStatus.WARNING
    
    def get_summary(self) -> Dict[str, Any]:
        """获取报告摘要"""
        issue_counts = {}
        for issue_type in IssueType:
            issue_counts[issue_type.value] = len([i for i in self.issues if i.issue_type == issue_type])
        
        return {
            'overall_status': self.status.value,
            'total_issues': len(self.issues),
            'critical_issues': len([i for i in self.issues if i.severity == HealthStatus.CRITICAL]),
            'warning_issues': len([i for i in self.issues if i.severity == HealthStatus.WARNING]),
            'issue_breakdown': issue_counts,
            'data_reduction': {
                'original_rows': self.original_shape[0],
                'cleaned_rows': self.cleaned_shape[0],
                'rows_removed': self.original_shape[0] - self.cleaned_shape[0],
                'removal_percentage': ((self.original_shape[0] - self.cleaned_shape[0]) / max(self.original_shape[0], 1)) * 100
            },
            'processing_time_seconds': self.processing_time
        }
    
    def to_json(self) -> str:
        """转换为JSON格式"""
        # 转换numpy和pandas类型为Python原生类型
        def convert_numpy_types(obj):
            if obj is None:
                return None
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.astype(object).to_dict()
            elif hasattr(obj, 'dtype'):  # pandas数据类型
                return str(obj)
            elif isinstance(obj, dict):
                return {str(key): convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                try:
                    # 尝试转换为基础类型
                    if hasattr(obj, 'item'):
                        return obj.item()
                    else:
                        return str(obj)
                except:
                    return str(obj)
        
        data = {
            'summary': convert_numpy_types(self.get_summary()),
            'issues': [issue.to_dict() for issue in self.issues],
            'statistics': convert_numpy_types(self.statistics),
            'timestamp': self.timestamp.isoformat()
        }
        return json.dumps(data, indent=2, ensure_ascii=False)


class EnhancedDataHealthChecker:
    """
    增强版数据健康检查器
    
    功能：
    1. 缺失值处理
    2. 异常值检测
    3. 时间连续性验证
    4. 数据频率一致性
    5. 数据清洗
    6. 详细报告生成
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        """
        初始化数据健康检查器
        
        Args:
            config: 配置字典
            logger: 日志器
        """
        self.config = config or load_config()
        self.logger = logger or get_strategy_logger("data_health_checker")
        
        # 从配置中获取阈值
        self.missing_threshold = self.config.get('DATA_MISSING_THRESHOLD', 0.1)
        self.outlier_threshold = self.config.get('DATA_OUTLIER_THRESHOLD', 3.0)
        self.zero_value_threshold = self.config.get('DATA_ZERO_VALUE_THRESHOLD', 0.05)
        
        self.logger.info("数据健康检查器初始化完成")
    
    def check_missing_values(self, df: pd.DataFrame, report: HealthReport) -> pd.DataFrame:
        """
        检查和处理缺失值
        
        Args:
            df: 输入数据框
            report: 健康报告
            
        Returns:
            处理后的数据框
        """
        self.logger.debug("开始缺失值检查...")
        
        missing_stats = df.isnull().sum()
        missing_ratios = df.isnull().mean()
        
        # 记录统计信息
        report.statistics['missing_values'] = {
            'total_missing': int(missing_stats.sum()),
            'missing_by_column': missing_stats.to_dict(),
            'missing_ratios': missing_ratios.to_dict()
        }
        
        # 检查每列的缺失值
        for column in df.columns:
            missing_ratio = missing_ratios[column]
            
            if missing_ratio > self.missing_threshold:
                # 严重缺失
                issue = HealthIssue(
                    issue_type=IssueType.MISSING_VALUES,
                    severity=HealthStatus.CRITICAL,
                    column=column,
                    description=f"列 {column} 缺失值比例过高: {missing_ratio:.2%}",
                    value=missing_ratio,
                    suggestion=f"考虑删除该列或使用插值方法填充"
                )
                report.add_issue(issue)
                
                # 如果是价格数据的关键列，尝试插值
                if column in ['open', 'high', 'low', 'close']:
                    df[column] = df[column].interpolate(method='linear')
                    self.logger.info(f"对价格列 {column} 进行线性插值")
                
            elif missing_ratio > 0:
                # 轻微缺失
                issue = HealthIssue(
                    issue_type=IssueType.MISSING_VALUES,
                    severity=HealthStatus.WARNING,
                    column=column,
                    description=f"列 {column} 存在缺失值: {missing_ratio:.2%}",
                    value=missing_ratio,
                    suggestion="使用向前填充或插值方法处理"
                )
                report.add_issue(issue)
                
                # 处理缺失值
                if column == 'volume':
                    # 成交量用0填充
                    df[column] = df[column].fillna(0)
                elif column in ['open', 'high', 'low', 'close']:
                    # 价格数据向前填充
                    df[column] = df[column].ffill()
                else:
                    # 其他数据插值
                    df[column] = df[column].interpolate()
        
        # 删除仍然有缺失值的行
        original_len = len(df)
        df = df.dropna()
        removed_rows = original_len - len(df)
        
        if removed_rows > 0:
            self.logger.info(f"删除了 {removed_rows} 行包含缺失值的数据")
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, report: HealthReport) -> pd.DataFrame:
        """
        检测和处理异常值
        
        Args:
            df: 输入数据框
            report: 健康报告
            
        Returns:
            处理后的数据框
        """
        self.logger.debug("开始异常值检测...")
        
        outlier_stats = {}
        
        # 检查数值列的异常值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in df.columns:
                # 使用IQR方法检测异常值
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 使用Z-score方法 (手动计算)
                values = df[column].dropna()
                z_scores = np.abs((values - values.mean()) / values.std())
                z_outliers = pd.Series(z_scores > self.outlier_threshold, index=values.index).reindex(df.index, fill_value=False)
                
                # IQR异常值
                iqr_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                
                # 组合检测
                outliers = iqr_outliers | z_outliers
                outlier_count = outliers.sum()
                outlier_ratio = outlier_count / len(df)
                
                outlier_stats[column] = {
                    'count': int(outlier_count),
                    'ratio': float(outlier_ratio),
                    'iqr_bounds': [float(lower_bound), float(upper_bound)],
                    'z_threshold': self.outlier_threshold
                }
                
                if outlier_ratio > 0.05:  # 超过5%为异常值
                    severity = HealthStatus.CRITICAL if outlier_ratio > 0.1 else HealthStatus.WARNING
                    issue = HealthIssue(
                        issue_type=IssueType.OUTLIERS,
                        severity=severity,
                        column=column,
                        description=f"列 {column} 异常值比例: {outlier_ratio:.2%}",
                        value=outlier_ratio,
                        suggestion="检查数据源，考虑使用中位数或插值替换异常值"
                    )
                    report.add_issue(issue)
                
                # 处理异常值
                if column in ['open', 'high', 'low', 'close']:
                    # 价格异常值用中位数替换
                    median_val = df[column].median()
                    df.loc[outliers, column] = median_val
                    if outlier_count > 0:
                        self.logger.info(f"用中位数替换了 {outlier_count} 个价格异常值 ({column})")
                
                elif column == 'volume':
                    # 成交量异常值检查
                    if (df[column] < 0).any():
                        negative_count = (df[column] < 0).sum()
                        issue = HealthIssue(
                            issue_type=IssueType.VOLUME_ANOMALY,
                            severity=HealthStatus.CRITICAL,
                            column=column,
                            description=f"发现 {negative_count} 个负成交量",
                            value=negative_count,
                            suggestion="将负成交量设置为0或删除对应行"
                        )
                        report.add_issue(issue)
                        # 将负成交量设为0
                        df.loc[df[column] < 0, column] = 0
        
        report.statistics['outliers'] = outlier_stats
        return df
    
    def check_time_continuity(self, df: pd.DataFrame, report: HealthReport) -> pd.DataFrame:
        """
        检查时间连续性
        
        Args:
            df: 输入数据框
            report: 健康报告
            
        Returns:
            处理后的数据框
        """
        self.logger.debug("开始时间连续性检查...")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            issue = HealthIssue(
                issue_type=IssueType.TIME_CONTINUITY,
                severity=HealthStatus.WARNING,
                column="index",
                description="索引不是时间类型",
                suggestion="将索引转换为DatetimeIndex"
            )
            report.add_issue(issue)
            return df
        
        # 计算时间间隔
        time_diffs = df.index.to_series().diff()
        
        # 识别时间频率
        mode_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else pd.Timedelta(minutes=30)
        expected_freq = mode_diff
        
        # 检查时间间隙
        large_gaps = time_diffs > expected_freq * 2
        gap_count = large_gaps.sum()
        
        time_stats = {
            'total_periods': len(df),
            'expected_frequency': str(expected_freq),
            'time_gaps': int(gap_count),
            'gap_ratio': float(gap_count / len(df)),
            'first_timestamp': df.index[0].isoformat(),
            'last_timestamp': df.index[-1].isoformat(),
            'total_duration': str(df.index[-1] - df.index[0])
        }
        
        if gap_count > 0:
            gap_ratio = gap_count / len(df)
            severity = HealthStatus.CRITICAL if gap_ratio > 0.05 else HealthStatus.WARNING
            issue = HealthIssue(
                issue_type=IssueType.TIME_CONTINUITY,
                severity=severity,
                column="index",
                description=f"发现 {gap_count} 个时间间隙",
                value=gap_count,
                suggestion="考虑填充缺失的时间点或重新采样数据"
            )
            report.add_issue(issue)
        
        # 检查重复时间戳
        duplicate_times = df.index.duplicated().sum()
        if duplicate_times > 0:
            issue = HealthIssue(
                issue_type=IssueType.TIME_CONTINUITY,
                severity=HealthStatus.WARNING,
                column="index",
                description=f"发现 {duplicate_times} 个重复时间戳",
                value=duplicate_times,
                suggestion="删除或合并重复时间戳的数据"
            )
            report.add_issue(issue)
            # 删除重复时间戳，保留第一个
            df = df[~df.index.duplicated(keep='first')]
        
        time_stats['duplicate_timestamps'] = int(duplicate_times)
        report.statistics['time_continuity'] = time_stats
        
        return df
    
    def check_frequency_consistency(self, df: pd.DataFrame, report: HealthReport) -> pd.DataFrame:
        """
        检查数据频率一致性
        
        Args:
            df: 输入数据框
            report: 健康报告
            
        Returns:
            处理后的数据框
        """
        self.logger.debug("开始频率一致性检查...")
        
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return df
        
        # 计算所有时间间隔
        time_diffs = df.index.to_series().diff().dropna()
        
        # 统计不同的时间间隔
        diff_counts = time_diffs.value_counts()
        most_common_diff = diff_counts.index[0]
        
        # 计算频率一致性
        consistency_ratio = diff_counts.iloc[0] / len(time_diffs)
        
        freq_stats = {
            'most_common_interval': str(most_common_diff),
            'consistency_ratio': float(consistency_ratio),
            'unique_intervals': len(diff_counts),
            'irregular_periods': int(len(time_diffs) - diff_counts.iloc[0])
        }
        
        if consistency_ratio < 0.9:  # 90%的数据应该有相同的频率
            issue = HealthIssue(
                issue_type=IssueType.FREQUENCY_CONSISTENCY,
                severity=HealthStatus.WARNING,
                column="index",
                description=f"数据频率不一致，一致性比例: {consistency_ratio:.2%}",
                value=consistency_ratio,
                suggestion="考虑重新采样到统一频率"
            )
            report.add_issue(issue)
        
        report.statistics['frequency_consistency'] = freq_stats
        return df
    
    def check_data_ranges(self, df: pd.DataFrame, report: HealthReport) -> pd.DataFrame:
        """
        检查数据范围合理性
        
        Args:
            df: 输入数据框
            report: 健康报告
            
        Returns:
            处理后的数据框
        """
        self.logger.debug("开始数据范围检查...")
        
        range_stats = {}
        
        # 检查OHLC逻辑关系
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High应该是最高的
            high_logic = (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['high'] >= df['low'])
            # Low应该是最低的
            low_logic = (df['low'] <= df['open']) & (df['low'] <= df['close']) & (df['low'] <= df['high'])
            
            invalid_high = (~high_logic).sum()
            invalid_low = (~low_logic).sum()
            
            if invalid_high > 0:
                issue = HealthIssue(
                    issue_type=IssueType.PRICE_ANOMALY,
                    severity=HealthStatus.CRITICAL,
                    column="high",
                    description=f"发现 {invalid_high} 个不合理的最高价",
                    value=invalid_high,
                    suggestion="检查数据源，修正OHLC逻辑关系"
                )
                report.add_issue(issue)
            
            if invalid_low > 0:
                issue = HealthIssue(
                    issue_type=IssueType.PRICE_ANOMALY,
                    severity=HealthStatus.CRITICAL,
                    column="low",
                    description=f"发现 {invalid_low} 个不合理的最低价",
                    value=invalid_low,
                    suggestion="检查数据源，修正OHLC逻辑关系"
                )
                report.add_issue(issue)
            
            range_stats['ohlc_logic'] = {
                'invalid_high_count': int(invalid_high),
                'invalid_low_count': int(invalid_low),
                'total_rows': len(df)
            }
        
        # 检查价格为零或负数
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                zero_or_negative = (df[col] <= 0).sum()
                if zero_or_negative > 0:
                    issue = HealthIssue(
                        issue_type=IssueType.PRICE_ANOMALY,
                        severity=HealthStatus.CRITICAL,
                        column=col,
                        description=f"发现 {zero_or_negative} 个零或负价格",
                        value=zero_or_negative,
                        suggestion="删除或修正无效价格数据"
                    )
                    report.add_issue(issue)
                    # 删除无效价格的行
                    df = df[df[col] > 0]
        
        report.statistics['data_ranges'] = range_stats
        return df
    
    def calculate_statistics(self, df: pd.DataFrame, report: HealthReport):
        """
        计算数据统计信息
        
        Args:
            df: 数据框
            report: 健康报告
        """
        stats_dict = {}
        
        # 基础统计
        stats_dict['basic'] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.to_dict()
        }
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_dict['numeric'] = df[numeric_cols].describe().to_dict()
        
        # 时间序列统计
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            stats_dict['time_series'] = {
                'start_date': df.index[0].isoformat(),
                'end_date': df.index[-1].isoformat(),
                'duration': str(df.index[-1] - df.index[0]),
                'frequency': pd.infer_freq(df.index) if len(df) >= 3 else None
            }
        
        # 特定金融指标
        if 'close' in df.columns and len(df) > 1:
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                stats_dict['financial'] = {
                    'volatility': float(returns.std()),
                    'mean_return': float(returns.mean()),
                    'max_return': float(returns.max()),
                    'min_return': float(returns.min()),
                    'skewness': float(returns.skew()),
                    'kurtosis': float(returns.kurtosis())
                }
        
        report.statistics.update(stats_dict)
    
    def comprehensive_health_check(self, 
                                 df: pd.DataFrame, 
                                 clean_data: bool = True,
                                 save_report: bool = True,
                                 report_path: str = None) -> HealthReport:
        """
        执行全面的健康检查
        
        Args:
            df: 输入数据框
            clean_data: 是否清洗数据
            save_report: 是否保存报告
            report_path: 报告保存路径
            
        Returns:
            健康检查报告
        """
        start_time = datetime.now()
        self.logger.info("开始全面数据健康检查...")
        
        # 初始化报告
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            original_shape=df.shape
        )
        
        # 检查空数据
        if df.empty:
            issue = HealthIssue(
                issue_type=IssueType.MISSING_VALUES,
                severity=HealthStatus.CRITICAL,
                column="all",
                description="数据框为空",
                suggestion="检查数据源和获取过程"
            )
            report.add_issue(issue)
            report.status = HealthStatus.FAILED
            return report
        
        # 如果需要清洗数据，创建副本
        if clean_data:
            cleaned_df = df.copy()
            
            # 执行各项检查和清洗
            cleaned_df = self.check_missing_values(cleaned_df, report)
            cleaned_df = self.detect_outliers(cleaned_df, report)
            cleaned_df = self.check_time_continuity(cleaned_df, report)
            cleaned_df = self.check_frequency_consistency(cleaned_df, report)
            cleaned_df = self.check_data_ranges(cleaned_df, report)
            
            report.cleaned_data = cleaned_df
            report.cleaned_shape = cleaned_df.shape
        else:
            # 只检查不清洗
            self.check_missing_values(df, report)
            self.detect_outliers(df, report)
            self.check_time_continuity(df, report)
            self.check_frequency_consistency(df, report)
            self.check_data_ranges(df, report)
            
            report.cleaned_shape = df.shape
        
        # 计算统计信息
        self.calculate_statistics(df, report)
        
        # 计算处理时间
        end_time = datetime.now()
        report.processing_time = (end_time - start_time).total_seconds()
        
        # 保存报告
        if save_report:
            self.save_report(report, report_path)
        
        self.logger.info(f"数据健康检查完成，状态: {report.status.value}, 处理时间: {report.processing_time:.2f}秒")
        
        return report
    
    def save_report(self, report: HealthReport, file_path: str = None):
        """
        保存健康检查报告
        
        Args:
            report: 健康报告
            file_path: 保存路径
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"scripts/logs/data_health_report_{timestamp}.json"
        
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存报告
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        
        self.logger.info(f"健康检查报告已保存到: {file_path}")
    
    def print_report_summary(self, report: HealthReport):
        """
        打印报告摘要
        
        Args:
            report: 健康报告
        """
        summary = report.get_summary()
        
        print("\n" + "=" * 60)
        print("📊 数据健康检查报告摘要")
        print("=" * 60)
        print(f"🎯 整体状态: {summary['overall_status'].upper()}")
        print(f"📈 数据维度: {summary['data_reduction']['original_rows']} 行 → {summary['data_reduction']['cleaned_rows']} 行")
        print(f"🗑️  删除数据: {summary['data_reduction']['rows_removed']} 行 ({summary['data_reduction']['removal_percentage']:.1f}%)")
        print(f"⏱️  处理时间: {summary['processing_time_seconds']:.2f} 秒")
        print(f"⚠️  总问题数: {summary['total_issues']} (严重: {summary['critical_issues']}, 警告: {summary['warning_issues']})")
        
        if report.issues:
            print("\n🔍 主要问题:")
            for issue in report.issues[:5]:  # 显示前5个问题
                status_emoji = "🚨" if issue.severity == HealthStatus.CRITICAL else "⚠️"
                print(f"  {status_emoji} {issue.description}")
                if issue.suggestion:
                    print(f"     💡 建议: {issue.suggestion}")
        
        print("=" * 60)


if __name__ == "__main__":
    # 演示用法
    from unified_data_client import UnifiedDataClient
    
    print("🔍 增强版数据健康检查器演示")
    print("=" * 50)
    
    # 创建检查器
    checker = EnhancedDataHealthChecker()
    
    # 获取测试数据
    try:
        data_client = UnifiedDataClient()
        df = data_client.get_historical_data("AAPL", "30min", limit=100)
        
        print(f"📊 获取测试数据: {df.shape[0]} 行 x {df.shape[1]} 列")
        
        # 执行健康检查 (暂时不保存报告)
        report = checker.comprehensive_health_check(df, clean_data=True, save_report=False)
        
        # 显示结果
        checker.print_report_summary(report)
        
        # 保存清洗后的数据
        if report.cleaned_data is not None:
            print(f"\n✅ 清洗后数据形状: {report.cleaned_data.shape}")
            print("📋 清洗后数据预览:")
            print(report.cleaned_data.head())
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        
        # 使用模拟数据演示
        print("\n🎭 使用模拟数据演示...")
        from mock_data_generator import MockDataGenerator
        
        generator = MockDataGenerator()
        df = generator.generate_historical_data("DEMO", "30min", records=50)
        
        # 人工添加一些问题用于演示
        df.iloc[5:8, 1] = np.nan  # 添加缺失值
        df.iloc[10, 3] = df.iloc[10, 3] * 10  # 添加异常值
        
        print(f"📊 模拟数据: {df.shape[0]} 行 x {df.shape[1]} 列")
        
        # 执行健康检查 (暂时不保存报告)
        report = checker.comprehensive_health_check(df, clean_data=True, save_report=False)
        
        # 显示结果
        checker.print_report_summary(report)
