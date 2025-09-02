import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
from data_preprocessor import DataPreprocessor
from data_health_checker import DataHealthChecker
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
from logger_setup import setup_logging
from tensorflow.keras.preprocessing.sequence import pad_sequences

class AAPLPositionManager:
    def __init__(self, account_balance=50000, risk_per_trade=0.01, max_position_pct=0.1, logger=None):
        self.balance = account_balance
        self.risk = risk_per_trade
        self.max_pct = max_position_pct
        self.total_assets = account_balance
        self.avg_volatility = 0.02
        self.logger = logger
        self.historical_win_rate = 0.5
        self.avg_winning_trade = 0.02
        self.avg_losing_trade = -0.01

    def update_volatility(self, atr=None, price=None):
        if price is None or price < 1e-3:  # 防止除零错误
            return
        # 添加安全阈值
        safe_price = max(price, 1e-3)
    
        if atr is not None:
            current_vol = atr / safe_price
            # 添加波动率平滑和边界控制
            self.avg_volatility = 0.7 * self.avg_volatility + 0.3 * max(min(current_vol, 0.5), 0.01)

    def calculate(self, entry_price, atr=None, atr_multiplier=2, confidence=1.0, volatility=None):
        if entry_price <= 1e-6:
            if self.logger:
                self.logger.warning(f"无效的入场价格: {entry_price}")
            return 0
        if not (0.1 < entry_price < 10000):
            self.logger.error(f"价格异常: {entry_price}")
            return 0
        if volatility is None:
            volatility = self.avg_volatility
        win_rate = self.historical_win_rate
        avg_win = self.avg_winning_trade
        avg_loss = self.avg_losing_trade
        b = avg_win / abs(avg_loss) if avg_loss != 0 else 1
        f_base = (b * win_rate - (1 - win_rate)) / b if b != 0 else 0
        vol_factor = max(0.5, 1 - volatility / 0.1)
        conf_factor = max(0.7, confidence)
        kelly_fraction = max(0.01, min(f_base * vol_factor * conf_factor, self.max_pct))
        position_size = self.balance * kelly_fraction
        shares = int(position_size / entry_price)
        return shares

    def check_intraday_volatility(self, current_volatility):
        if current_volatility > 3 * self.avg_volatility:
            self.logger.warning(f"异常波动! 状态快照: position={self.current_position}")
            return False
        return True

    def apply_dynamic_exit(self, position, current_price, atr):
        """实现1:2风险回报比+移动回撤止盈"""
        # 多头仓位
        if position.position_type == 'LONG':
            risk = position.entry_price - position.stop_loss
            # 1:2风险回报比止盈
            take_profit = position.entry_price + 2 * risk
       
            # 移动回撤止盈 (盈利>5%后启动)
            if current_price > position.peak_price:
                position.peak_price = current_price
       
            if (position.peak_price - position.entry_price) / position.entry_price > 0.05:
                trailing_stop = position.peak_price * 0.99
                position.stop_loss = max(position.stop_loss, trailing_stop)
           
            return take_profit, position.stop_loss
   
        # 空头仓位
        elif position.position_type == 'SHORT':
            risk = position.stop_loss - position.entry_price
            # 1:2风险回报比止盈 (对称实现)
            take_profit = position.entry_price - 2 * risk
       
            # 移动回撤止盈
            if current_price < position.peak_price:
                position.peak_price = current_price
       
            if (position.entry_price - position.peak_price) / position.entry_price > 0.05:
                trailing_stop = position.peak_price * 1.01
                position.stop_loss = min(position.stop_loss, trailing_stop)
           
            return take_profit, position.stop_loss
   
        return None, None

class Backtester:
    # 定义全局保护特征列表
    PROTECTED_FEATURES = ['close', 'open', 'high', 'low', 'volume', 'market_state']
   
    # 简化的信号验证器
    def validate_signal(self, row, prediction):
        """简化的信号验证逻辑"""
        # 基础风险检查
        volatility = row.get('volatility', 0)
        if volatility > 0.08:  # 波动率过高时暂停交易
            return 'HOLD'
            
        # 流动性检查
        volume = row.get('volume', 0)
        if volume < 1e6:  # 成交量过低时暂停交易
            return 'HOLD'
            
        # 根据预测生成信号
        if prediction == 2:  # 看涨
            return 'BUY'
        elif prediction == 0:  # 看跌
            return 'SHORT'
        else:  # 中性
            return 'HOLD'

    def __init__(self, model_path, config_path, feature_path, timeframe='30min', logger=None):
        self.logger = logger or setup_logging()
        self.pipeline = self._load_pipeline(model_path)
        self.model = self.pipeline['model']
        self.scaler = self.pipeline.get('scaler', None)
        self.selected_features = self.pipeline.get('feature_selector', [])
        self.config = self._load_config(config_path)
        self.feature_path = feature_path
        self.data_health_checker = DataHealthChecker(logger=self.logger)
        self.data_preprocessor = DataPreprocessor(logger=self.logger)
        self.window_size = self.config.get('window_size', 60)
        self.stop_method = self.config.get('stop_method', 'fixed')
        self.atr_multiplier = self.config.get('atr_multiplier', 2)
        self.last_trade_time = None
        self.min_trade_interval = self._get_interval(timeframe)
        self.max_drawdown = 0.2
        self.atr_cache = {}
        self.current_position = 0 # 统一头寸跟踪
       
        # 新增交易频率控制参数
        self.MAX_DAILY_TRADES = 3
        self.HIGH_VOL_LIMIT = 1 # 波动率>5%时的交易上限
        self.daily_trade_count = 0
        self.last_trade_date = None
      
        # 添加滑点模型
        self.SLIPPAGE_MODEL = {
            'normal': {'mean': 0.0003, 'std': 0.0001},
            'high_vol': {'mean': 0.0012, 'std': 0.0003}
        }
      
        # 成本追踪初始化
        self.total_slippage_cost = 0
        self.total_commission_cost = 0
        self.commission_rate = 0.0005 # 统一佣金率
       
        # 简化的配置
        self.risk_threshold = self.config.get('risk_threshold', 0.08)
   
    # ... 类的其他方法保持不变 ...

    class Position:
        def __init__(self, entry_price, shares, position_type, atr):
            self.entry_price = entry_price
            self.shares = shares
            self.position_type = position_type
            self.atr = atr
            # 添加止损和止盈跟踪
            self.stop_loss = None
            self.take_profit = None
            self.peak_price = entry_price # 用于追踪最高价（多头）或最低价（空头）
           
            # 根据头寸类型初始化止损
            if position_type == 'LONG':
                self.stop_loss = entry_price - 2 * atr if atr else entry_price * 0.95
            elif position_type == 'SHORT':
                self.stop_loss = entry_price + 2 * atr if atr else entry_price * 1.05

    def _get_interval(self, tf):
        intervals = {'5min': 5, '15min': 15, '30min': 30, '1h': 60}
        return intervals.get(tf, 30)

    def _load_pipeline(self, pipeline_path):
        try:
            self.logger.info(f"Loading model pipeline: {pipeline_path}")
            return joblib.load(pipeline_path)
        except Exception as e:
            self.logger.error(f"Pipeline loading failed: {str(e)}")
            raise

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Config loading failed: {str(e)}")
            return {}

    def load_and_check_features(self):
        try:
            self.logger.info(f"Loading feature data: {self.feature_path}")
            features = pd.read_feather(self.feature_path)
            if 'datetime' in features.columns and not isinstance(features.index, pd.DatetimeIndex):
                features = features.set_index('datetime', drop=False)
            ny_tz = 'America/New_York'
            if features.index.tz is None:
                features.index = features.index.tz_localize(ny_tz, ambiguous='NaT', nonexistent='shift_forward')
            else:
                features.index = features.index.tz_convert(ny_tz)
            features = features[features.index.notnull()]
            if 'datetime' in features.columns:
                features['datetime'] = features.index
            self.logger.info(f"特征数据时间范围: {features.index.min()} 到 {features.index.max()}")
            required_columns = ['close', 'market_state']
            missing_cols = [col for col in required_columns if col not in features.columns]
            if missing_cols:
                self.logger.error(f"缺失关键列: {missing_cols}")
                raise ValueError(f"特征数据缺失必需列: {missing_cols}")
            for col in ['close']:
                na_count = features[col].isna().sum()
                if na_count > 0:
                    self.logger.warning(f"列 '{col}' 包含 {na_count} 个缺失值")
                    features[col] = features[col].ffill().bfill()
            if features['close'].isna().any() or features['close'].min() <= 0:
                self.logger.error("close列数据无效")
                raise ValueError("价格数据问题")
            health_report = self.data_health_checker.data_health_check(
                features.drop(columns=['market_state'], errors='ignore'),
                raise_on_error=False
            )
            if health_report['issues']:
                self.logger.warning("Data health issues: " + "; ".join(health_report['issues']))
            else:
                self.logger.info("Data health check passed")
            if "时间序列不连续" in health_report['issues']:
                self.logger.warning("处理时间序列不连续问题...")
                features = features.asfreq('30min', method='ffill')
                self.logger.info(f"重新采样后的数据形状: {features.shape}")
            inf_mask = np.isinf(features.select_dtypes(include=np.number))
            if inf_mask.any().any():
                self.logger.warning(f"发现 {inf_mask.sum().sum()} 个INF值，正在清理")
                features = features.replace([np.inf, -np.inf], np.nan)
            # 在特征重建前检查保护特征
            for feature in self.PROTECTED_FEATURES:
                if feature not in features.columns:
                    self.logger.warning(f"关键特征 {feature} 缺失，尝试重建")
                    reconstructed = False
          
                    if feature == 'close':
                        if 'last' in features.columns:
                            features['close'] = features['last']
                            reconstructed = True
                        elif 'price' in features.columns:
                            features['close'] = features['price']
                            reconstructed = True
          
                    # 添加其他关键特征的恢复逻辑...
                    elif feature == 'open' and 'close' in features.columns:
                        features['open'] = features['close'].shift(1).ffill()
                        reconstructed = True
                    elif feature == 'high' and 'close' in features.columns:
                        features['high'] = features['close']
                        reconstructed = True
                    elif feature == 'low' and 'close' in features.columns:
                        features['low'] = features['close']
                        reconstructed = True
                    elif feature == 'volume':
                        features['volume'] = 0
                        reconstructed = True
                    elif feature == 'market_state':
                        features['market_state'] = 1 # Neutral
                        reconstructed = True
          
                    if not reconstructed:
                        self.logger.critical(f"无法重建关键特征: {feature}")
                        features[feature] = 0 # 安全回退
            zero_cols = features.columns[(features == 0).all()]
            if len(zero_cols) > 0:
                self.logger.warning(f"全零列: {zero_cols}，尝试重新计算")
                for col in zero_cols:
                    if 'volatility' in col and 'close' in features.columns: # 确保close存在
                        features[col] = features['close'].pct_change().rolling(20).std()
            critical_features = ['close', 'volume', 'rsi']
            for feature in critical_features:
                if feature in features.columns:
                    na_ratio = features[feature].isna().sum() / len(features)
                    if na_ratio > 0.3:
                        self.logger.warning(f"特征 {feature} 缺失值超过30%，尝试重新计算")
                        if feature == 'volatility':
                            features[feature] = features['close'].pct_change().rolling(20, min_periods=1).std().fillna(0)
                        elif feature == 'returns_1':
                            features[feature] = features['close'].pct_change(1).fillna(0)
            if 'atr' not in features.columns:
                self.logger.warning("动态计算ATR")
                high, low, close = features['high'], features['low'], features['close']
                tr = np.maximum(high - low, 
                                np.abs(high - close.shift()), 
                                np.abs(low - close.shift()))
                features['atr'] = tr.rolling(14).mean().fillna(method='bfill')
            features['atr'] = features['atr'].clip(lower=0.01)
            numeric_features = features.select_dtypes(include=np.number)
            corr_matrix = numeric_features.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            # 在删除操作前添加检查
            protected_to_preserve = [f for f in self.PROTECTED_FEATURES if f in features.columns]
            to_drop = [col for col in to_drop if col not in protected_to_preserve]
            features = features.drop(columns=to_drop)
            self.logger.info(f"Removed redundant features: {to_drop}")
            stds = numeric_features.std()
            low_std_cols = stds[stds < 0.001].index
            low_std_cols = [col for col in low_std_cols if col not in protected_to_preserve]
            features = features.drop(columns=low_std_cols)
            self.logger.info(f"Removed low std features: {low_std_cols}")
            # 增强的完整性检查
            missing_protected = [f for f in self.PROTECTED_FEATURES if f not in features.columns]
            if missing_protected:
                self.logger.critical(f"关键特征缺失: {missing_protected}")
                raise ValueError("关键价格特征缺失")
            diffs = features.index.to_series().diff().dropna()
            expected_interval = pd.Timedelta(minutes=30)
            irregular = diffs != expected_interval
            if irregular.any():
                self.logger.warning(f"Found {irregular.sum()} irregular intervals")
                features = features.asfreq('30min', method='ffill')
            return features
        except Exception as e:
            self.logger.error(f"Feature data processing failed: {str(e)}")
            raise

    def prepare_data(self, features):
        try:
            raw_features = features.copy()
            raw_features = raw_features.replace([np.inf, -np.inf], np.nan)
            nan_report = raw_features.isna().sum()
            self.logger.warning(f"NaN分布: {nan_report[nan_report>0].to_dict()}")
            raw_features = raw_features.ffill().bfill().fillna(0)
            if 'datetime' in raw_features.columns:
                raw_features = raw_features.drop(columns=['datetime'])
            if 'volatility' not in raw_features.columns:
                if 'close' in raw_features.columns:
                    self.logger.warning("动态计算波动率特征")
                    raw_features['volatility'] = raw_features['close'].pct_change().rolling(20, min_periods=1).std().fillna(0)
                    self.logger.info("波动率特征计算完成") # 只在成功时记录
                else:
                    self.logger.error("无法计算波动率特征：close列缺失，使用默认值0")
                    raw_features['volatility'] = 0
            if 'returns_1' not in raw_features.columns:
                self.logger.warning("动态计算1期收益率特征")
                raw_features['returns_1'] = raw_features['close'].pct_change(1).fillna(0)
                self.logger.info("1期收益率特征计算完成")
            if 'returns_3' not in raw_features.columns:
                self.logger.warning("动态计算3期收益率特征")
                raw_features['returns_3'] = raw_features['close'].pct_change(3).fillna(0)
                self.logger.info("3期收益率特征计算完成")
            if 'returns_6' not in raw_features.columns:
                self.logger.warning("动态计算6期收益率特征")
                raw_features['returns_6'] = raw_features['close'].pct_change(6).fillna(0)
                self.logger.info("6期收益率特征计算完成")
            if 'volume_osc' not in raw_features.columns and 'volume' in raw_features.columns:
                self.logger.warning("动态计算成交量震荡指标")
                vol_mean = raw_features['volume'].rolling(20, min_periods=1).mean()
                vol_std = raw_features['volume'].rolling(20, min_periods=1).std().replace(0, 1)
                raw_features['volume_osc'] = (raw_features['volume'] - vol_mean) / vol_std
                raw_features['volume_osc'] = raw_features['volume_osc'].fillna(0)
                self.logger.info("成交量震荡指标计算完成")
            if 'macd_hist' not in raw_features.columns and 'close' in raw_features.columns:
                self.logger.warning("动态计算MACD特征")
                try:
                    ema12 = raw_features['close'].ewm(span=12, adjust=False).mean()
                    ema26 = raw_features['close'].ewm(span=26, adjust=False).mean()
                    macd = ema12 - ema26
                    macd_signal = macd.ewm(span=9, adjust=False).mean()
                    raw_features['macd_hist'] = macd - macd_signal
                    raw_features['macd_hist'] = raw_features['macd_hist'].fillna(0)
                    self.logger.info("MACD特征计算成功")
                except Exception as e:
                    self.logger.error(f"计算MACD特征失败: {str(e)}")
                    try:
                        ema12 = raw_features['close'].rolling(12).mean()
                        ema26 = raw_features['close'].rolling(26).mean()
                        raw_features['macd_hist'] = ema12 - ema26
                        self.logger.warning("使用简单方法计算MACD特征")
                    except:
                        raw_features['macd_hist'] = 0
                        self.logger.error("无法计算MACD特征，使用默认值0")
            # 简化技术指标计算 - 只保留必要的RSI和ATR
            if 'RSI' not in raw_features.columns and 'close' in raw_features.columns:
                self.logger.info("计算RSI指标")
                delta = raw_features['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                raw_features['RSI'] = 100 - (100 / (1 + rs))
                raw_features['RSI'] = raw_features['RSI'].fillna(50)  # 用中性值填充
            numeric_features = raw_features.select_dtypes(include=np.number)
            corr_matrix = numeric_features.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            # 在删除操作前添加检查
            protected_to_preserve = [f for f in self.PROTECTED_FEATURES if f in raw_features.columns]
            to_drop = [col for col in to_drop if col not in protected_to_preserve]
            raw_features = raw_features.drop(columns=to_drop)
            stds = numeric_features.std()
            low_std_cols = stds[stds < 0.001].index
            low_std_cols = [col for col in low_std_cols if col not in protected_to_preserve]
            # 价格特征的特殊处理
            price_features = ['close', 'open', 'high', 'low']
            for col in price_features:
                if col in raw_features.columns:
                    # 即使标准差低也不删除
                    low_std_cols = [c for c in low_std_cols if c != col]
            raw_features = raw_features.drop(columns=low_std_cols)
            if self.selected_features:
                available_features = [col for col in self.selected_features if col in raw_features.columns]
                if len(available_features) < len(self.selected_features):
                    missing = set(self.selected_features) - set(available_features)
                    self.logger.warning(f"Missing selected features: {missing}. Using available ones.")
                required_cols = list(set(available_features + self.PROTECTED_FEATURES))
                features = raw_features[required_cols]
            essential_missing = [col for col in ['close', 'market_state'] if col not in features.columns]
            if essential_missing:
                self.logger.error(f"缺失关键特征: {essential_missing}")
                return np.array([]), raw_features
            numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 5:
                self.logger.error("有效数值特征不足！")
                return np.array([]), features
            min_required = self.window_size + 30
            if len(features) < min_required:
                self.logger.error(f"数据量不足! 需要至少 {min_required} 行数据，当前只有 {len(features)} 行")
                return np.array([]), raw_features
            features = features.ffill().bfill().fillna(0)
            X = []
            total_windows = len(features) - self.window_size + 1
            skipped = 0
            for i in range(total_windows):
                window = features.iloc[i:i + self.window_size]
                try:
                    window_values = window.values
                    # 简化的有效性检查
                    if np.isfinite(window_values).sum() / window_values.size > 0.7:
                        X.append(window_values)
                    else:
                        skipped += 1
                except:
                    skipped += 1
            X = np.array(X)
            if skipped > 0:
                self.logger.warning(f"Skipped {skipped} invalid windows (NaN >30% or non-finite)")
            if X.size == 0:
                self.logger.error("创建的滚动数据集为空!")
                return np.array([]), raw_features
            if self.scaler:
                n_samples, n_timesteps, n_features = X.shape
                X_2d = X.reshape(n_samples * n_timesteps, n_features)
                X_scaled = self.scaler.transform(X_2d)
                X = X_scaled.reshape(n_samples, n_timesteps, n_features)
            expected_features = len(features.columns)
            if len(X.shape) < 3 or X.shape[-1] != expected_features:
                self.logger.error(f"特征维度不匹配! 预期: {expected_features}, 实际: {X.shape[-1] if len(X.shape) >= 3 else 'N/A'}")
                return np.array([]), raw_features
            self.logger.info(f"数据准备完成, 形状: {X.shape}")
            # 最终检查
            if 'close' not in features.columns:
                self.logger.critical("数据准备完成后'close'列丢失!")
                raise ValueError("关键特征缺失")
            return X, features
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            return np.array([]), features

    def predict_with_confidence(self, sequences):
        """增强预测方法，返回信号和置信度"""
        try:
            if sequences.size == 0:
                return np.array([]), np.array([])
           
            if isinstance(self.model, tf.keras.Model):
                predictions = self.model.predict(sequences, verbose=0)
                confidence = np.max(predictions, axis=1)
            elif hasattr(self.model, 'predict_proba'):
                # 支持概率预测的模型（如LightGBM）
                n_samples, n_timesteps, n_features = sequences.shape
                sequences_2d = sequences.reshape(n_samples, n_timesteps * n_features)
                proba = self.model.predict_proba(sequences_2d)
                confidence = np.max(proba, axis=1)
                predictions = np.argmax(proba, axis=1)
            else:
                # 不支持概率的模型使用预测置信度
                n_samples, n_timesteps, n_features = sequences.shape
                sequences_2d = sequences.reshape(n_samples, n_timesteps * n_features)
                predictions = self.model.predict(sequences_2d)
                # 使用预测值本身作为置信度代理
                confidence = np.abs(predictions - 0.5) * 2 # 假设预测值在0-1之间
            return predictions, confidence
        except Exception as e:
            self.logger.error(f"置信度预测失败: {str(e)}")
            return np.array([]), np.array([])

    def calculate_slippage(self, price, row=None):
        """统一滑点计算逻辑"""
        if pd.isna(price) or price <= 0:
            return 0.001
       
        # 确定滑点类型
        volatility = row.get('volatility', 0) if row is not None else 0
        if pd.isna(volatility) or volatility <= 0:
            # 使用position_manager的avg_volatility作为后备值
            if hasattr(self, 'position_manager'):
                volatility = self.position_manager.avg_volatility
            else:
                volatility = 0.02  # 默认值
        slippage_type = 'high_vol' if volatility > 0.05 else 'normal'
       
        # 从模型获取参数
        model_params = self.SLIPPAGE_MODEL.get(slippage_type, {'mean': 0.0005, 'std': 0.0002})
        return np.random.normal(model_params['mean'], model_params['std'])

    def execute_trade(self, action, price, shares, is_forced=False):
        """统一交易执行和成本记录"""
        # 计算成本和更新头寸
        trade_value = price * shares
       
        # 统一成本计算
        commission = max(trade_value * self.commission_rate, 1)
        self.total_commission_cost += commission
       
        # 更新头寸
        if action == "BUY":
            self.current_position += shares
        elif action == "SELL":
            self.current_position = max(0, self.current_position - shares)
        elif action == "SHORT":
            self.current_position -= shares
        elif action == "COVER":
            self.current_position = min(0, self.current_position + shares)
       
        # 只有主动交易才计数（强制平仓不计入每日限制）
        if not is_forced:
            self.daily_trade_count += 1

    def backtest_strategy(self, features, predictions, initial_balance=50000):
        self.logger.info("Starting strategy backtest...")
        if predictions.size == 0:
            self.logger.error("预测结果为空，使用中性市场状态")
            predictions = np.ones(len(features))
        if len(features) != len(predictions):
            min_length = min(len(features), len(predictions))
            features = features.iloc[-min_length:]
            predictions = predictions[-min_length:]
        features = features.copy()
        features['prediction'] = predictions
        balance = initial_balance
        self.current_position = 0
        trades = []
        portfolio_values = []
        active_position = None # 替换current_position等分散变量
        trade_price = None
        position_manager = AAPLPositionManager(account_balance=balance, logger=self.logger)
        self.position_manager = position_manager
        peak_value = initial_balance
        self.atr_cache = {idx: row['atr'] for idx, row in features.iterrows() if 'atr' in row and not pd.isna(row['atr'])}
        # 在循环开始前确保日期变量初始化
        if self.last_trade_date is None:
            self.last_trade_date = features.index[0].date()
        total_rows = len(features)
        for i, (idx, row) in enumerate(features.iterrows()):
            if i % 500 == 0 or i == total_rows - 1:  # 减少日志频率
                self.logger.info(f"回测进度: {i+1}/{total_rows} ({(i+1)/total_rows:.1%})")
            current_time = pd.Timestamp(idx)
            current_price = row['close']
            if pd.isna(current_price) or not (1e-6 < current_price < 10000):
                portfolio_values.append(balance + (active_position.shares * current_price if active_position and active_position.position_type == 'LONG' else -active_position.shares * current_price if active_position else 0))
                continue
            # 日期边界处理 - 确保每日重置
            current_date = current_time.date()
          
            if current_date != self.last_trade_date:
                self.daily_trade_count = 0
                self.last_trade_date = current_date
          
            # 波动率有效性检查
            volatility = row.get('volatility', 0)
            if pd.isna(volatility) or volatility <= 0:
                volatility = position_manager.avg_volatility
          
            # 动态交易限制
            vol_limit = self.HIGH_VOL_LIMIT if volatility > 0.05 else self.MAX_DAILY_TRADES
            if self.daily_trade_count >= vol_limit:
                # 即使跳过交易，仍需要记录组合价值
                portfolio_values.append(balance + (active_position.shares * current_price if active_position and active_position.position_type == 'LONG' else -active_position.shares * current_price if active_position else 0))
                continue
            atr = self.atr_cache.get(current_time)
            current_vol = atr / current_price if atr else position_manager.avg_volatility
            if not position_manager.check_intraday_volatility(current_vol):
                self.logger.info(f"状态同步: position={self.current_position}, cash={balance}")
                portfolio_values.append(balance + (active_position.shares * current_price if active_position and active_position.position_type == 'LONG' else -active_position.shares * current_price if active_position else 0))
                continue
            # 简化的信号生成
            signal = self.generate_trading_signal(row, active_position, current_time)
            
            # 应用风险过滤
            if signal in ['BUY', 'SHORT']:
                signal = self.validate_signal(row, int(row['prediction']))
            if signal in ['BUY', 'SELL', 'SHORT', 'COVER'] and self.last_trade_time:
                if (current_time - self.last_trade_time).total_seconds() / 60 < self.min_trade_interval:
                    signal = 'HOLD'

            confidence = 1.0 if row['prediction'] == 2 else 0.8 if row['prediction'] == 1 else 0.6
            if signal == 'BUY' and active_position is None and balance > 0:
                slippage = self.calculate_slippage(current_price, row=row)
                executed_price = current_price * (1 + slippage)
                shares = position_manager.calculate(executed_price, atr, confidence=confidence, volatility=volatility)
                if shares > 0:
                    position_size = shares * executed_price
                    if position_size > balance:
                        shares = int(balance / executed_price)
                        position_size = shares * executed_price
                    commission = max(position_size * self.commission_rate, 1)
                    slippage_cost = position_size * slippage
                    self.total_slippage_cost += slippage_cost
                    self.total_commission_cost += commission
                    balance -= position_size + commission
                    active_position = self.Position(
                        entry_price=executed_price,
                        shares=shares,
                        position_type='LONG',
                        atr=atr
                    )
                    trade_price = executed_price
                    trades.append(('BUY', idx, executed_price, shares))
                    self.last_trade_time = current_time
                    self.logger.info(f"BUY@{executed_price:.2f}, Shares:{shares}")
            elif signal == 'SELL' and active_position and active_position.position_type == 'LONG':
                # 1. 保存仓位信息
                shares = active_position.shares
                pos_type = active_position.position_type
                # 2. 执行平仓和清空
                slippage = self.calculate_slippage(current_price, row=row)
                executed_price = current_price * (1 - slippage)
                sell_value = shares * executed_price
                commission = max(sell_value * self.commission_rate, 1)
                slippage_cost = sell_value * slippage
                self.total_slippage_cost += slippage_cost
                self.total_commission_cost += commission
                balance += sell_value - commission
                trades.append(('SELL', idx, executed_price, shares))
                self.execute_trade('SELL', executed_price, shares)
                # 3. 记录日志（使用保存的变量）
                self.logger.info(f"SELL@{executed_price:.2f}, Shares:{shares}")
                # 4. 最后设置 None
                active_position = None
                trade_price = None
                self.last_trade_time = current_time
            elif signal == 'SHORT' and active_position is None and balance > 0:
                slippage = self.calculate_slippage(current_price, row=row)
                executed_price = current_price * (1 - slippage)
                shares = position_manager.calculate(executed_price, atr, confidence=confidence, volatility=volatility)
                if shares > 0:
                    max_shares_value = balance * 0.5
                    max_shares = int(max_shares_value / executed_price)
                    shares = min(shares, max_shares)
                    position_size = shares * executed_price
                    commission = max(position_size * self.commission_rate, 1)
                    slippage_cost = position_size * slippage
                    self.total_slippage_cost += slippage_cost
                    self.total_commission_cost += commission
                    balance += position_size - commission
                    active_position = self.Position(
                        entry_price=executed_price,
                        shares=shares, # shares为正数
                        position_type='SHORT',
                        atr=atr
                    )
                    trade_price = executed_price
                    trades.append(('SHORT', idx, executed_price, shares))
                    self.last_trade_time = current_time
                    self.logger.info(f"SHORT@{executed_price:.2f}, Shares:{shares}")
            elif signal == 'COVER' and active_position and active_position.position_type == 'SHORT':
                # 1. 保存仓位信息
                shares = active_position.shares
                pos_type = active_position.position_type
                # 2. 执行平仓和清空
                slippage = self.calculate_slippage(current_price, row=row)
                executed_price = current_price * (1 + slippage)
                buy_value = shares * executed_price
                commission = max(buy_value * self.commission_rate, 1)
                slippage_cost = buy_value * slippage
                self.total_slippage_cost += slippage_cost
                self.total_commission_cost += commission
                balance -= buy_value + commission
                trades.append(('COVER', idx, executed_price, shares))
                self.execute_trade('COVER', executed_price, shares)
                # 3. 记录日志（使用保存的变量）
                self.logger.info(f"COVER@{executed_price:.2f}, Shares:{shares}")
                # 4. 最后设置 None
                active_position = None
                trade_price = None
                self.last_trade_time = current_time

            current_value = balance + (active_position.shares * current_price if active_position and active_position.position_type == 'LONG' else -active_position.shares * current_price if active_position else 0)
            if active_position:
                take_profit, stop_loss = position_manager.apply_dynamic_exit(
                    active_position, 
                    current_price,
                    atr
                )
                active_position.stop_loss = stop_loss
                
                # 检查止损
                if (active_position.position_type == 'LONG' and current_price <= stop_loss) or \
                   (active_position.position_type == 'SHORT' and current_price >= stop_loss):
                    # 先保存关键信息
                    shares = active_position.shares
                    pos_type = active_position.position_type
                    slippage = self.calculate_slippage(current_price, row=row)
                    if pos_type == 'LONG':
                        executed_price = current_price * (1 - slippage)
                        sell_value = shares * executed_price
                        commission = max(sell_value * self.commission_rate, 1)
                        slippage_cost = sell_value * slippage
                        self.total_slippage_cost += slippage_cost
                        self.total_commission_cost += commission
                        balance += sell_value - commission

                        trades.append(('SELL', idx, executed_price, shares))
                        self.execute_trade('SELL', executed_price, shares, is_forced=True)
                    else:
                        executed_price = current_price * (1 + slippage)
                        buy_value = shares * executed_price
                        commission = max(buy_value * self.commission_rate, 1)
                        slippage_cost = buy_value * slippage
                        self.total_slippage_cost += slippage_cost
                        self.total_commission_cost += commission
                        balance -= buy_value + commission

                        trades.append(('COVER', idx, executed_price, shares))
                        self.execute_trade('COVER', executed_price, shares, is_forced=True)
                    action = 'SELL' if pos_type == 'LONG' else 'COVER'
                    self.logger.info(f"{action}@{executed_price:.2f}, Shares:{shares} (Stop Loss)")
                    active_position = None
                    trade_price = None
                    current_value = balance
                    # 跳过后续止盈检查
                    continue # 直接进入下一循环
                
                # 检查止盈
                if (active_position.position_type == 'LONG' and current_price >= take_profit) or \
                   (active_position.position_type == 'SHORT' and current_price <= take_profit):
                    # 先保存关键信息
                    shares = active_position.shares
                    pos_type = active_position.position_type
                    slippage = self.calculate_slippage(current_price, row=row)
                    if pos_type == 'LONG':
                        executed_price = current_price * (1 - slippage)
                        sell_value = shares * executed_price
                        commission = max(sell_value * self.commission_rate, 1)
                        slippage_cost = sell_value * slippage
                        self.total_slippage_cost += slippage_cost
                        self.total_commission_cost += commission
                        balance += sell_value - commission

                        trades.append(('SELL', idx, executed_price, shares))
                        self.execute_trade('SELL', executed_price, shares)
                    else:
                        executed_price = current_price * (1 + slippage)
                        buy_value = shares * executed_price
                        commission = max(buy_value * self.commission_rate, 1)
                        slippage_cost = buy_value * slippage
                        self.total_slippage_cost += slippage_cost
                        self.total_commission_cost += commission
                        balance -= buy_value + commission

                        trades.append(('COVER', idx, executed_price, shares))
                        self.execute_trade('COVER', executed_price, shares)
                    action = 'SELL' if pos_type == 'LONG' else 'COVER'
                    self.logger.info(f"{action}@{executed_price:.2f}, Shares:{shares} (Take Profit)")
                    current_value = balance
                    active_position = None
                    continue # 确保跳过后续检查

            peak_value = max(peak_value, current_value)
            drawdown = (peak_value - current_value) / peak_value
            if drawdown > self.max_drawdown and active_position:
                # 先保存关键信息
                shares = active_position.shares
                pos_type = active_position.position_type
                slippage = self.calculate_slippage(current_price, row=row)
                if pos_type == 'LONG':
                    executed_price = current_price * (1 - slippage)
                    value = shares * executed_price
                    commission = max(value * self.commission_rate, 1)
                    slippage_cost = value * slippage
                    self.total_slippage_cost += slippage_cost
                    self.total_commission_cost += commission
                    balance += value - commission
                    default_metadata = {
                        'pyramid_modified': False,
                        'pyramid_rejected': False,
                        'original_signal': None
                    }
                    trades.append(('SELL', idx, executed_price, shares, default_metadata))
                    self.execute_trade('SELL', executed_price, shares, is_forced=True)
                else:
                    executed_price = current_price * (1 + slippage)
                    value = shares * executed_price
                    commission = max(value * self.commission_rate, 1)
                    slippage_cost = value * slippage
                    self.total_slippage_cost += slippage_cost
                    self.total_commission_cost += commission
                    balance -= value + commission
                    default_metadata = {
                        'pyramid_modified': False,
                        'pyramid_rejected': False,
                        'original_signal': None
                    }
                    trades.append(('COVER', idx, executed_price, shares, default_metadata))
                    self.execute_trade('COVER', executed_price, shares, is_forced=True)
                action = 'SELL' if pos_type == 'LONG' else 'COVER'
                self.logger.info(f"{action}@{executed_price:.2f}, Shares:{shares}")
                active_position = None
                trade_price = None
                current_value = balance
                peak_value = current_value

            position_manager.balance = balance
            position_manager.total_assets = current_value
            portfolio_values.append(current_value)

        if active_position:
            last_row = features[features['close'].between(1e-6, 10000)].iloc[-1]
            if not isinstance(last_row, pd.Series):
                self.logger.error("无效的最后一行数据，无法平仓")
                return results, trades
            last_price = last_row['close']
            atr = self.atr_cache.get(last_row.name)
            # 在获取后立即添加，
            shares = active_position.shares
            pos_type = active_position.position_type
            # 添加日志记录
            self.logger.info(f"Final closing position: {pos_type}, Shares:{shares}")
            slippage = self.calculate_slippage(last_price, row=last_row)
            if pos_type == 'LONG':
                executed_price = last_price * (1 - slippage)
                value = shares * executed_price
                commission = max(value * self.commission_rate, 1)
                slippage_cost = value * slippage
                self.total_slippage_cost += slippage_cost
                self.total_commission_cost += commission
                balance += value - commission
                trades.append(('SELL', last_row.name, executed_price, shares))
                self.execute_trade('SELL', executed_price, shares, is_forced=True)
                self.logger.info(f"SELL@{executed_price:.2f}, Shares:{shares} (Final Close)")
            else:
                executed_price = last_price * (1 + slippage)
                value = shares * executed_price
                commission = max(value * self.commission_rate, 1)
                slippage_cost = value * slippage
                self.total_slippage_cost += slippage_cost
                self.total_commission_cost += commission
                balance -= value + commission
                trades.append(('COVER', last_row.name, executed_price, shares))
                self.execute_trade('COVER', executed_price, shares, is_forced=True)
                self.logger.info(f"COVER@{executed_price:.2f}, Shares:{shares} (Final Close)")
            active_position = None
            portfolio_values[-1] = balance

        results = pd.DataFrame(
            {'portfolio_value': portfolio_values, 'prediction': features['prediction']},
            index=features.index
        )
        self.logger.info(f"Backtest complete, final value: ${portfolio_values[-1]:.2f}")
        return results, trades

    def generate_trading_signal(self, row, active_position, current_time):
        # 使用active_position代替current_position
        if active_position:
            position_type = active_position.position_type
            # 基于position_type生成信号
            if active_position and self.last_trade_time and (current_time - self.last_trade_time) > pd.Timedelta(hours=4):
                return 'SELL' if position_type == 'LONG' else 'COVER'
            trade_price = active_position.entry_price if active_position else None
            if position_type == 'LONG' and trade_price and (row['close'] / trade_price - 1) > 0.03:
                return 'SELL'
            if position_type == 'SHORT' and trade_price and (trade_price / row['close'] - 1) > 0.03:
                return 'COVER'
            atr = row.get('atr', 0) or 0
            if position_type == 'LONG' and trade_price and atr and row['close'] > 1e-6:
                if row['close'] - trade_price > 2 * atr:
                    return 'SELL'
            elif position_type == 'SHORT' and trade_price and atr and row['close'] > 1e-6:
                if trade_price - row['close'] > 2 * atr:
                    return 'COVER'
        else:
            prediction = int(row['prediction'])
            atr = row.get('atr', 0) or 0
            rsi = row.get('rsi', 50) or 50
            if atr and not pd.isna(atr) and row['close'] > 1e-6 and (atr / row['close']) > 0.08:
                return 'HOLD'
            if prediction == 2 or (prediction == 1 and rsi and 40 < rsi < 60):
                return 'BUY'
            if prediction == 0:
                return 'SHORT'
        return 'HOLD'



    def plot_results(self, results):
        try:
            plt.figure(figsize=(14, 7))
            plt.subplot(2, 1, 1)
            results['portfolio_value'].plot(title='Portfolio Value')
            plt.ylabel('Value ($)')
            plt.subplot(2, 1, 2)
            results['prediction'].plot(title='Market State Predictions', marker='o', linestyle='')
            plt.yticks([0, 1, 2], ['Bear', 'Neutral', 'Bull'])
            plt.tight_layout()
            plt.savefig('backtest_results.png')
            self.logger.info("Results chart saved as backtest_results.png")
            return True
        except Exception as e:
            self.logger.error(f"Plotting failed: {str(e)}")
            return False

    def run_backtest(self):
        self.current_position = 0
        start_time = datetime.now()
        self.logger.info(f"===== Starting backtest {start_time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        try:
            features = self.load_and_check_features()
            sequences, processed_features = self.prepare_data(features)
            self.logger.info(f"最终使用的特征列: {processed_features.columns.tolist()}")
            if sequences.size == 0:
                # ...
                self.logger.error("无法准备数据，使用中性市场状态进行回测")
                predictions, confidences = np.array([]), np.array([])
            else:
                predictions, confidences = self.predict_with_confidence(sequences) # 修改这里
                if predictions.size == 0:
                    # ...
                    self.logger.error("预测结果为空，使用中性市场状态")
                    predictions = np.ones(len(processed_features))
                    confidences = np.zeros(len(processed_features))
           
            # 简化处理
            if len(predictions) != len(processed_features):
                min_length = min(len(predictions), len(processed_features))
                self.logger.warning(f"预测结果与特征数据长度不一致，调整到最小长度: {min_length}")
                predictions = predictions[:min_length]
                processed_features = processed_features.iloc[:min_length]
            results, trades = self.backtest_strategy(processed_features, predictions)
            results.to_csv('backtest_results.csv')
            pd.DataFrame(trades, columns=['action', 'timestamp', 'price', 'shares']).to_csv('backtest_trades.csv', index=False)
            if not results.empty:
                self.plot_results(results)
            if sequences.size > 0:
                self.logger.info("滚动数据集非空")
            if 'atr' in processed_features.columns and not processed_features['atr'].isna().all():
                self.logger.info("ATR列存在且有效")
            actions = set(t[0] for t in trades)
            if {'BUY', 'SELL', 'SHORT', 'COVER'}.issubset(actions):
                self.logger.info("交易信号包含多空双向操作")
            if self.current_position == 0:  # 使用实际记录头寸的变量
                self.logger.info("最终仓位自动平仓")
            report = self.generate_report(results, trades, start_time, processed_features)
            return report
        except Exception as e:
            self.logger.exception("Critical error during backtest")
            return {"status": "error", "message": str(e)}
        finally:
            duration = datetime.now() - start_time
            self.logger.info(f"Backtest complete, duration: {duration}")

    def generate_report(self, results, trades, start_time, features):
        """生成简化的回测报告，符合预期输出格式"""
        try:
            # 计算交易对
            trade_pairs = []
            open_positions = {'LONG': [], 'SHORT': []}
            
            for trade in trades:
                action, timestamp, price, shares = trade[:4]  # 只取前4个元素，忽略metadata
               
                if action in ['BUY', 'SHORT']:
                    commission = max(price * shares * self.commission_rate, 1)
                    pos_type = 'LONG' if action == 'BUY' else 'SHORT'
                    open_positions[pos_type].append({
                        'timestamp': timestamp, 
                        'price': price, 
                        'shares': shares, 
                        'commission': commission
                    })
                elif action in ['SELL', 'COVER']:
                    pos_type = 'LONG' if action == 'SELL' else 'SHORT'
                    shares_to_close = shares
                    
                    while shares_to_close > 0 and open_positions[pos_type]:
                        pos = open_positions[pos_type][0]
                        shares_from_pos = min(pos['shares'], shares_to_close)
                        buy_commission = pos['commission'] * (shares_from_pos / pos['shares'])
                        sell_commission = max(price * shares_from_pos * self.commission_rate, 1)
                        
                        profit = (price - pos['price']) * shares_from_pos * (1 if pos_type == 'LONG' else -1) - (buy_commission + sell_commission)
                        trade_pairs.append({
                            'entry': (pos['timestamp'], pos['price'], shares_from_pos),
                            'exit': (timestamp, price, shares_from_pos),
                            'profit': profit,
                            'holding_period': (timestamp - pos['timestamp']).total_seconds() / 3600
                        })
                        
                        shares_to_close -= shares_from_pos
                        pos['shares'] -= shares_from_pos
                        if pos['shares'] == 0:
                            open_positions[pos_type].pop(0)

            # 处理未平仓的仓位
            if any(open_positions.values()):
                final_price = features['close'].iloc[-1]
                for pos_type in ['LONG', 'SHORT']:
                    for pos in open_positions[pos_type]:
                        profit = (final_price - pos['price']) * pos['shares'] * (1 if pos_type == 'LONG' else -1) - pos['commission']
                        trade_pairs.append({
                            'entry': (pos['timestamp'], pos['price'], pos['shares']),
                            'exit': (results.index[-1], final_price, pos['shares']),
                            'profit': profit,
                            'holding_period': (results.index[-1] - pos['timestamp']).total_seconds() / 3600
                        })

            # 计算关键指标
            winning_trades = sum(1 for pair in trade_pairs if pair['profit'] > 0)
            total_trades = len(trade_pairs)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # 计算年化收益率
            final_value = results['portfolio_value'].iloc[-1]
            initial_balance = results['portfolio_value'].iloc[0]
            total_days = (results.index[-1] - results.index[0]).days
            
            if total_days > 0 and initial_balance > 0:
                annual_return = (final_value / initial_balance) ** (252 / total_days) - 1
            else:
                annual_return = 0.0
            
            # 计算最大回撤
            portfolio_values = results['portfolio_value']
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            # 计算夏普比率
            returns = results['portfolio_value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # 返回符合预期格式的报告
            return {
                'annual_return': round(annual_return, 4),
                'max_drawdown': round(max_drawdown, 4), 
                'win_rate': round(win_rate, 4),
                'sharpe_ratio': round(sharpe_ratio, 4),
                
                # 额外的详细信息（可选）
                'details': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'final_value': final_value,
                    'initial_balance': initial_balance,
                    'total_return': (final_value - initial_balance) / initial_balance,
                    'total_days': total_days,
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_slippage_cost': self.total_slippage_cost,
                    'total_commission_cost': self.total_commission_cost
                }
            }
        except Exception as e:
            self.logger.error(f"生成报告时出错: {str(e)}")
            return {
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'error': str(e)
            }

if __name__ == "__main__":
    logger = setup_logging()
    try:
        backtester = Backtester(
            model_path='trading_pipeline.pkl',
            config_path='config.json',
            feature_path='features_selected_aapl_30min.feather',
            logger=logger
        )
        report = backtester.run_backtest()
        
        # 显示主要结果
        logger.info("=" * 50)
        logger.info("回测结果摘要:")
        logger.info("=" * 50)
        logger.info(f"年化收益率: {report['annual_return']:.2%}")
        logger.info(f"最大回撤: {report['max_drawdown']:.2%}")
        logger.info(f"胜率: {report['win_rate']:.2%}")
        logger.info(f"夏普比率: {report['sharpe_ratio']:.2f}")
        logger.info("=" * 50)
        
        # 显示详细信息（如果需要）
        if 'details' in report:
            details = report['details']
            logger.info(f"总交易次数: {details['total_trades']}")
            logger.info(f"胜利交易: {details['winning_trades']}")
            logger.info(f"最终价值: ${details['final_value']:.2f}")
            logger.info(f"总回报: {details['total_return']:.2%}")
        
        logger.info(f"\n完整报告:\n{json.dumps(report, indent=2)}")
    except Exception as e:
        logger.error(f"Backtest execution failed: {str(e)}")
        sys.exit(1)