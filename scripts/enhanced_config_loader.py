"""
增强版配置加载器
支持YAML/JSON多格式、类型验证、安全管理、多环境配置
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import hashlib
import base64

from config_schema import TradingSystemConfig, validate_config


class ConfigSecurityManager:
    """配置安全管理器"""
    
    def __init__(self, key_file: str = ".config_key"):
        self.key_file = key_file
        self._fernet = None
        self._setup_encryption()
    
    def _setup_encryption(self):
        """设置加密"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            os.chmod(self.key_file, 0o600)  # 仅所有者可读写
        
        self._fernet = Fernet(key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        return self._fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        return self._fernet.decrypt(encrypted_data.encode()).decode()
    
    def validate_api_key(self, api_key: str) -> bool:
        """验证API密钥格式"""
        if not api_key:
            return False
        # 基本长度和字符检查
        if len(api_key) < 16 or not all(c.isalnum() or c in '-_' for c in api_key):
            return False
        return True


class EnhancedConfigLoader:
    """增强版配置加载器"""
    
    # 默认配置
    _DEFAULT_CONFIG = {
        "trading": {
            "ticker": "AAPL",
            "exchange": "ARCA", 
            "currency": "USD",
            "initial_capital": 100000.0,
            "benchmark": "SPY"
        },
        "data": {
            "timeframe": "30min",
            "history_years": 5,
            "train_days": 1260,
            "volatility_window": 60,
            "trade_frequency_minutes": 60
        },
        "model": {
            "update_interval_days": 3,
            "min_samples": 50,
            "rolling_window": 98,
            "feature_selection_method": "shap_dynamic",
            "enable_model_monitor": True,
            "walkforward_test_window": 120,
            "walkforward_step": 120
        },
        "risk_management": {
            "risk_per_trade": 0.02,
            "max_trade_percentage": 0.1,
            "stop_loss_multiplier": 1.8,
            "take_profit_multiplier": 2.5,
            "max_drawdown": 0.1,
            "daily_loss_limit": -0.03
        },
        "costs": {
            "transaction_cost": 0.0005,
            "slippage": 0.001
        },
        "adaptive_threshold": {
            "initial_threshold": 0.75,
            "transition_covariance": 0.01,
            "observation_covariance": 1.0,
            "confirm_signal_k": 3,
            "atr_threshold": 0.015
        },
        "system": {
            "live_trading": False,
            "max_retries": 5,
            "enable_crash_recovery": True,
            "log_level": "INFO",
            "use_best_params": True
        },
        "api": {
            "twelve_data_api_key": None
        }
    }
    
    def __init__(self, 
                 config_dir: str = ".",
                 environment: str = "development",
                 enable_security: bool = True):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
            environment: 环境名称 (development, testing, production)
            enable_security: 是否启用安全功能
        """
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.enable_security = enable_security
        self.logger = self._setup_logger()
        
        if enable_security:
            self.security_manager = ConfigSecurityManager()
        else:
            self.security_manager = None
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("ConfigLoader")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _find_config_file(self) -> Optional[Path]:
        """查找配置文件"""
        # 环境特定配置文件优先级
        possible_files = [
            f"config.{self.environment}.yaml",
            f"config.{self.environment}.yml", 
            f"config.{self.environment}.json",
            "config.yaml",
            "config.yml",
            "config.json"
        ]
        
        for filename in possible_files:
            config_path = self.config_dir / filename
            if config_path.exists():
                self.logger.info(f"Found config file: {config_path}")
                return config_path
        
        return None
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_path}: {e}")
            raise
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """加载环境变量"""
        load_dotenv()
        
        env_config = {}
        
        # API密钥
        api_key = os.getenv("TWELVE_DATA_API_KEY")
        if api_key:
            if self.security_manager and self.security_manager.validate_api_key(api_key):
                env_config["api"] = {"twelve_data_api_key": api_key}
            else:
                self.logger.warning("Invalid API key format in environment variables")
        
        # 其他环境变量
        env_mappings = {
            "TRADING_TICKER": ("trading", "ticker"),
            "TRADING_INITIAL_CAPITAL": ("trading", "initial_capital", float),
            "DATA_TIMEFRAME": ("data", "timeframe"),
            "RISK_PER_TRADE": ("risk_management", "risk_per_trade", float),
            "LOG_LEVEL": ("system", "log_level"),
            "LIVE_TRADING": ("system", "live_trading", lambda x: x.lower() == 'true')
        }
        
        for env_var, mapping in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section, key = mapping[:2]
                converter = mapping[2] if len(mapping) > 2 else str
                
                if section not in env_config:
                    env_config[section] = {}
                
                try:
                    env_config[section][key] = converter(value)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to convert {env_var}={value}: {e}")
        
        return env_config
    
    def _merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并多个配置字典"""
        result = {}
        
        for config in configs:
            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = value
        
        return result
    
    def _load_best_params(self) -> Dict[str, Any]:
        """加载最佳参数文件"""
        best_params_file = self.config_dir / "best_params.json"
        
        if not best_params_file.exists():
            return {}
        
        try:
            with open(best_params_file, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
                self.logger.info("Loaded best_params.json")
                return best_params
        except Exception as e:
            self.logger.warning(f"Failed to load best_params.json: {e}")
            return {}
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置"""
        errors = validate_config(config)
        if errors:
            error_msg = "Configuration validation failed:\\n" + "\\n".join(errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def load(self, validate: bool = True) -> Dict[str, Any]:
        """
        加载配置
        
        Args:
            validate: 是否验证配置
            
        Returns:
            配置字典
        """
        self.logger.info(f"Loading configuration for environment: {self.environment}")
        
        # 1. 从默认配置开始
        config = self._DEFAULT_CONFIG.copy()
        
        # 2. 加载配置文件
        config_file = self._find_config_file()
        if config_file:
            file_config = self._load_config_file(config_file)
            config = self._merge_configs(config, file_config)
        else:
            self.logger.warning("No config file found, using defaults")
        
        # 3. 加载环境变量
        env_config = self._load_environment_variables()
        if env_config:
            config = self._merge_configs(config, env_config)
        
        # 4. 加载最佳参数（如果启用）
        if config.get("system", {}).get("use_best_params", True):
            best_params = self._load_best_params()
            if best_params:
                # 将扁平的best_params转换为分层结构
                structured_best_params = self._convert_flat_to_structured(best_params)
                config = self._merge_configs(config, structured_best_params)
        
        # 5. 验证配置
        if validate:
            try:
                self._validate_config(config)
                self.logger.info("Configuration validation passed")
            except ValueError as e:
                if self.environment == "production":
                    raise
                else:
                    self.logger.warning(f"Configuration validation failed (non-production): {e}")
        
        # 6. 转换为扁平格式（向后兼容）
        flat_config = TradingSystemConfig.from_dict(config).to_flat_dict()
        
        self.logger.info("Configuration loaded successfully")
        return flat_config
    
    def _convert_flat_to_structured(self, flat_config: Dict[str, Any]) -> Dict[str, Any]:
        """将扁平配置转换为分层结构"""
        structured = {}
        
        # 映射规则
        mappings = {
            "TICKER": ("trading", "ticker"),
            "EXCHANGE": ("trading", "exchange"), 
            "CURRENCY": ("trading", "currency"),
            "INIT_CAPITAL": ("trading", "initial_capital"),
            "BENCHMARK": ("trading", "benchmark"),
            
            "DATA_TIMEFRAME": ("data", "timeframe"),
            "HISTORY_YEARS": ("data", "history_years"),
            "TRAIN_DAYS": ("data", "train_days"),
            "VOLATILITY_WINDOW": ("data", "volatility_window"),
            "TRADE_FREQ_MIN": ("data", "trade_frequency_minutes"),
            
            "MODEL_UPDATE_INTERVAL_DAYS": ("model", "update_interval_days"),
            "MIN_SAMPLES": ("model", "min_samples"),
            "ROLLING_WINDOW": ("model", "rolling_window"),
            "FEATURE_SELECTION_METHOD": ("model", "feature_selection_method"),
            
            "RISK_PER_TRADE": ("risk_management", "risk_per_trade"),
            "MAX_TRADE_PCT": ("risk_management", "max_trade_percentage"),
            "STOPLOSS_MULTIPLIER": ("risk_management", "stop_loss_multiplier"),
            "TAKEPROFIT_MULTIPLIER": ("risk_management", "take_profit_multiplier"),
            "MAX_DRAWDOWN": ("risk_management", "max_drawdown"),
            "DAILY_LOSS_LIMIT": ("risk_management", "daily_loss_limit"),
            
            "TRANSACTION_COST": ("costs", "transaction_cost"),
            "SLIPPAGE": ("costs", "slippage"),
            
            "ADAPTIVE_THRESHOLD_INIT": ("adaptive_threshold", "initial_threshold"),
            "ADAPTIVE_THRESHOLD_TRANS_COV": ("adaptive_threshold", "transition_covariance"),
            "ADAPTIVE_THRESHOLD_OBS_COV": ("adaptive_threshold", "observation_covariance"),
            "CONFIRM_SIGNAL_K": ("adaptive_threshold", "confirm_signal_k"),
            "ATR_THRESHOLD": ("adaptive_threshold", "atr_threshold"),
            
            "LIVE_TRADING": ("system", "live_trading"),
            "MAX_RETRIES": ("system", "max_retries"),
            "ENABLE_CRASH_RECOVERY": ("system", "enable_crash_recovery"),
            "LOG_LEVEL": ("system", "log_level"),
            "USE_BEST_PARAMS": ("system", "use_best_params"),
            
            "TWELVE_DATA_API_KEY": ("api", "twelve_data_api_key")
        }
        
        for flat_key, value in flat_config.items():
            if flat_key in mappings:
                section, key = mappings[flat_key]
                if section not in structured:
                    structured[section] = {}
                structured[section][key] = value
        
        return structured
    
    def save_config(self, config: Dict[str, Any], filename: str = None) -> None:
        """保存配置到文件"""
        if filename is None:
            filename = f"config.{self.environment}.yaml"
        
        config_path = self.config_dir / filename
        
        # 转换为分层结构
        if all(key.isupper() for key in config.keys()):
            # 扁平格式，需要转换
            structured_config = self._convert_flat_to_structured(config)
        else:
            structured_config = config
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(structured_config, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise


# 便捷函数
def load_config(environment: str = "development", 
                config_dir: str = ".",
                validate: bool = True) -> Dict[str, Any]:
    """加载配置的便捷函数"""
    loader = EnhancedConfigLoader(
        config_dir=config_dir,
        environment=environment
    )
    return loader.load(validate=validate)


def get_config_loader(environment: str = "development", 
                     config_dir: str = ".") -> EnhancedConfigLoader:
    """获取配置加载器实例"""
    return EnhancedConfigLoader(
        config_dir=config_dir,
        environment=environment
    )


if __name__ == "__main__":
    # 示例使用
    try:
        config = load_config(environment="development")
        print("✅ Configuration loaded successfully!")
        print(f"Ticker: {config.get('TICKER')}")
        print(f"Environment: development")
        print(f"Live Trading: {config.get('LIVE_TRADING')}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")

