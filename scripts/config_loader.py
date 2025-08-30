import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

class ConfigLoader:
    """配置加载器（增强版，支持 .env 环境变量与 best_params.json 自动合并）"""
    _DEFAULT_CONFIG = {
        # 核心参数
        "TICKER": "AAPL",
        "EXCHANGE": "ARCA",
        "CURRENCY": "USD",
        "INIT_CAPITAL": 100000.0,

        # 数据与时间设置
        "DATA_TIMEFRAME": "30min",
        "HISTORY_YEARS": 5,
        "TRAIN_DAYS": 1260,
        "VOLATILITY_WINDOW": 60,
        "TRADE_FREQ_MIN": 60,

        # 模型与窗口配置
        "MODEL_UPDATE_INTERVAL_DAYS": 3,
        "MIN_SAMPLES": 50,
        "ROLLING_WINDOW": 98,
        "FEATURE_SELECTION_METHOD": "shap_dynamic",

        # 风控设置
        "RISK_PER_TRADE": 0.02,
        "MAX_TRADE_PCT": 0.1,
        "STOPLOSS_MULTIPLIER": 1.8,
        "TAKEPROFIT_MULTIPLIER": 2.5,
        "MAX_DRAWDOWN": 0.1,
        "DAILY_LOSS_LIMIT": -0.03,

        # 成本模型
        "TRANSACTION_COST": 0.0005,
        "SLIPPAGE": 0.001,

        # 自适应阈值模块
        "ADAPTIVE_THRESHOLD_INIT": 0.75,
        "ADAPTIVE_THRESHOLD_TRANS_COV": 0.01,
        "ADAPTIVE_THRESHOLD_OBS_COV": 1.0,
        "CONFIRM_SIGNAL_K": 3,
        "ATR_THRESHOLD": 0.015,

        # 实盘与恢复控制
        "LIVE_TRADING": False,
        "BENCHMARK": "SPY",
        "WALKFORWARD_TEST_WINDOW": 120,
        "WALKFORWARD_STEP": 120,
        "MAX_RETRIES": 5,
        "ENABLE_MODEL_MONITOR": True,
        "ENABLE_CRASH_RECOVERY": True,
        "LOG_LEVEL": "INFO",

        # 控制是否启用 best_params.json 合并
        "USE_BEST_PARAMS": True,

        # API密钥（可从 .env 覆盖）
        "TWELVE_DATA_API_KEY": "your_default_api_key"
    }

    _REQUIRED_KEYS = [
        "TICKER", "EXCHANGE", "CURRENCY", "INIT_CAPITAL",
        "DATA_TIMEFRAME", "TRAIN_DAYS", "VOLATILITY_WINDOW",
        "RISK_PER_TRADE", "STOPLOSS_MULTIPLIER", "TAKEPROFIT_MULTIPLIER",
        "MAX_DRAWDOWN", "DAILY_LOSS_LIMIT", "MAX_TRADE_PCT",
        "MODEL_UPDATE_INTERVAL_DAYS", "TWELVE_DATA_API_KEY"
    ]

    def __init__(self, config_file: str = "config.json", best_params_file: str = "best_params.json"):
        self.config_file = config_file
        self.best_params_file = best_params_file

    def load(self) -> Dict[str, Any]:
        load_dotenv()
        config = self._DEFAULT_CONFIG.copy()

        # 优先使用 .env 中的 TWELVE_DATA_API_KEY
        env_api_key = os.getenv("TWELVE_DATA_API_KEY")
        if env_api_key:
            config["TWELVE_DATA_API_KEY"] = env_api_key

        # 加载主配置文件
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                print(f"[ConfigLoader] ⚠️ 加载 config.json 失败: {str(e)}，使用默认配置")

        # 合并 best_params.json
        if config.get("USE_BEST_PARAMS", False) and os.path.exists(self.best_params_file):
            try:
                with open(self.best_params_file, 'r') as f:
                    best_params = json.load(f)
                    config.update(best_params)
                    print(f"[ConfigLoader] ✅ best_params.json 已合并到配置")
            except Exception as e:
                print(f"[ConfigLoader] ⚠️ 加载 best_params.json 失败: {str(e)}，跳过合并")

        # 检查必需字段
        for key in self._REQUIRED_KEYS:
            if key not in config:
                print(f"[ConfigLoader] ⚠️ 缺少配置项 '{key}'，使用默认值: {self._DEFAULT_CONFIG[key]}")
                config[key] = self._DEFAULT_CONFIG[key]

        return config

# ✅ 模块外部调用接口
def load_config() -> Dict[str, Any]:
    return ConfigLoader().load()

if __name__ == "__main__":
    config = load_config()
    print("当前配置：")
    print(json.dumps(config, indent=2))