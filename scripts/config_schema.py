"""
配置模式验证模块
定义配置参数的类型、范围和验证规则
"""

from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field
from enum import Enum
import re


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TimeFrame(Enum):
    """时间框架枚举"""
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    HOUR_1 = "1h"
    DAY_1 = "1day"


class FeatureSelectionMethod(Enum):
    """特征选择方法枚举"""
    SHAP_DYNAMIC = "shap_dynamic"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    CHI2 = "chi2"


@dataclass
class TradingConfig:
    """交易配置"""
    ticker: str
    exchange: str
    currency: str
    initial_capital: float
    benchmark: str = "SPY"
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not re.match(r'^[A-Z]{1,6}$', self.ticker):
            raise ValueError("ticker must be 1-6 uppercase letters")
        if self.initial_capital < 1000:
            raise ValueError("initial_capital should be at least 1000")


@dataclass
class DataConfig:
    """数据配置"""
    timeframe: str
    history_years: int
    train_days: int
    volatility_window: int
    trade_frequency_minutes: int
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.timeframe not in [tf.value for tf in TimeFrame]:
            raise ValueError(f"timeframe must be one of {[tf.value for tf in TimeFrame]}")
        if not 1 <= self.history_years <= 10:
            raise ValueError("history_years must be between 1 and 10")
        if not 30 <= self.train_days <= 5000:
            raise ValueError("train_days must be between 30 and 5000")
        if not 5 <= self.volatility_window <= 252:
            raise ValueError("volatility_window must be between 5 and 252")


@dataclass
class ModelConfig:
    """模型配置"""
    update_interval_days: int
    min_samples: int
    rolling_window: int
    feature_selection_method: str
    enable_model_monitor: bool = True
    walkforward_test_window: int = 120
    walkforward_step: int = 120
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if not 1 <= self.update_interval_days <= 30:
            raise ValueError("update_interval_days must be between 1 and 30")
        if not 10 <= self.min_samples <= 1000:
            raise ValueError("min_samples must be between 10 and 1000")
        if not 10 <= self.rolling_window <= 500:
            raise ValueError("rolling_window must be between 10 and 500")
        if self.feature_selection_method not in [fsm.value for fsm in FeatureSelectionMethod]:
            raise ValueError(f"feature_selection_method must be one of {[fsm.value for fsm in FeatureSelectionMethod]}")


@dataclass
class RiskManagementConfig:
    """风险管理配置"""
    risk_per_trade: float
    max_trade_percentage: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    max_drawdown: float
    daily_loss_limit: float
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if not 0.001 <= self.risk_per_trade <= 0.1:
            raise ValueError("risk_per_trade must be between 0.001 and 0.1")
        if not 0.01 <= self.max_trade_percentage <= 1.0:
            raise ValueError("max_trade_percentage must be between 0.01 and 1.0")
        if not 0.5 <= self.stop_loss_multiplier <= 5.0:
            raise ValueError("stop_loss_multiplier must be between 0.5 and 5.0")
        if not 0.5 <= self.take_profit_multiplier <= 10.0:
            raise ValueError("take_profit_multiplier must be between 0.5 and 10.0")
        if not 0.01 <= self.max_drawdown <= 0.5:
            raise ValueError("max_drawdown must be between 0.01 and 0.5")
        if not -0.5 <= self.daily_loss_limit <= 0:
            raise ValueError("daily_loss_limit must be between -0.5 and 0")


@dataclass
class CostsConfig:
    """交易成本配置"""
    transaction_cost: float
    slippage: float
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if not 0 <= self.transaction_cost <= 0.01:
            raise ValueError("transaction_cost must be between 0 and 0.01")
        if not 0 <= self.slippage <= 0.01:
            raise ValueError("slippage must be between 0 and 0.01")


@dataclass
class AdaptiveThresholdConfig:
    """自适应阈值配置"""
    initial_threshold: float
    transition_covariance: float
    observation_covariance: float
    confirm_signal_k: int
    atr_threshold: float
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if not 0.1 <= self.initial_threshold <= 1.0:
            raise ValueError("initial_threshold must be between 0.1 and 1.0")
        if not 0.001 <= self.transition_covariance <= 1.0:
            raise ValueError("transition_covariance must be between 0.001 and 1.0")
        if not 0.1 <= self.observation_covariance <= 10.0:
            raise ValueError("observation_covariance must be between 0.1 and 10.0")
        if not 1 <= self.confirm_signal_k <= 10:
            raise ValueError("confirm_signal_k must be between 1 and 10")
        if not 0.001 <= self.atr_threshold <= 0.1:
            raise ValueError("atr_threshold must be between 0.001 and 0.1")


@dataclass
class SystemConfig:
    """系统配置"""
    live_trading: bool = False
    max_retries: int = 5
    enable_crash_recovery: bool = True
    log_level: str = "INFO"
    use_best_params: bool = True
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.log_level not in [level.value for level in LogLevel]:
            raise ValueError(f"log_level must be one of {[level.value for level in LogLevel]}")
        if not 1 <= self.max_retries <= 20:
            raise ValueError("max_retries must be between 1 and 20")


@dataclass
class ApiConfig:
    """API配置"""
    twelve_data_api_key: Optional[str] = None
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        if self.twelve_data_api_key is not None and len(self.twelve_data_api_key.strip()) == 0:
            raise ValueError("twelve_data_api_key cannot be empty string")


@dataclass
class TradingSystemConfig:
    """完整的交易系统配置"""
    trading: TradingConfig
    data: DataConfig
    model: ModelConfig
    risk_management: RiskManagementConfig
    costs: CostsConfig
    adaptive_threshold: AdaptiveThresholdConfig
    system: SystemConfig
    api: ApiConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TradingSystemConfig':
        """从字典创建配置对象"""
        return cls(
            trading=TradingConfig(**config_dict.get('trading', {})),
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            risk_management=RiskManagementConfig(**config_dict.get('risk_management', {})),
            costs=CostsConfig(**config_dict.get('costs', {})),
            adaptive_threshold=AdaptiveThresholdConfig(**config_dict.get('adaptive_threshold', {})),
            system=SystemConfig(**config_dict.get('system', {})),
            api=ApiConfig(**config_dict.get('api', {}))
        )
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """转换为扁平字典格式（兼容旧版本）"""
        return {
            # Trading
            "TICKER": self.trading.ticker,
            "EXCHANGE": self.trading.exchange,
            "CURRENCY": self.trading.currency,
            "INIT_CAPITAL": self.trading.initial_capital,
            "BENCHMARK": self.trading.benchmark,
            
            # Data
            "DATA_TIMEFRAME": self.data.timeframe,
            "HISTORY_YEARS": self.data.history_years,
            "TRAIN_DAYS": self.data.train_days,
            "VOLATILITY_WINDOW": self.data.volatility_window,
            "TRADE_FREQ_MIN": self.data.trade_frequency_minutes,
            
            # Model
            "MODEL_UPDATE_INTERVAL_DAYS": self.model.update_interval_days,
            "MIN_SAMPLES": self.model.min_samples,
            "ROLLING_WINDOW": self.model.rolling_window,
            "FEATURE_SELECTION_METHOD": self.model.feature_selection_method,
            "ENABLE_MODEL_MONITOR": self.model.enable_model_monitor,
            "WALKFORWARD_TEST_WINDOW": self.model.walkforward_test_window,
            "WALKFORWARD_STEP": self.model.walkforward_step,
            
            # Risk Management
            "RISK_PER_TRADE": self.risk_management.risk_per_trade,
            "MAX_TRADE_PCT": self.risk_management.max_trade_percentage,
            "STOPLOSS_MULTIPLIER": self.risk_management.stop_loss_multiplier,
            "TAKEPROFIT_MULTIPLIER": self.risk_management.take_profit_multiplier,
            "MAX_DRAWDOWN": self.risk_management.max_drawdown,
            "DAILY_LOSS_LIMIT": self.risk_management.daily_loss_limit,
            
            # Costs
            "TRANSACTION_COST": self.costs.transaction_cost,
            "SLIPPAGE": self.costs.slippage,
            
            # Adaptive Threshold
            "ADAPTIVE_THRESHOLD_INIT": self.adaptive_threshold.initial_threshold,
            "ADAPTIVE_THRESHOLD_TRANS_COV": self.adaptive_threshold.transition_covariance,
            "ADAPTIVE_THRESHOLD_OBS_COV": self.adaptive_threshold.observation_covariance,
            "CONFIRM_SIGNAL_K": self.adaptive_threshold.confirm_signal_k,
            "ATR_THRESHOLD": self.adaptive_threshold.atr_threshold,
            
            # System
            "LIVE_TRADING": self.system.live_trading,
            "MAX_RETRIES": self.system.max_retries,
            "ENABLE_CRASH_RECOVERY": self.system.enable_crash_recovery,
            "LOG_LEVEL": self.system.log_level,
            "USE_BEST_PARAMS": self.system.use_best_params,
            
            # API
            "TWELVE_DATA_API_KEY": self.api.twelve_data_api_key
        }


# 验证函数
def validate_config(config_dict: Dict[str, Any]) -> List[str]:
    """验证配置字典，返回错误信息列表"""
    errors = []
    
    try:
        TradingSystemConfig.from_dict(config_dict)
    except Exception as e:
        errors.append(str(e))
    
    return errors

