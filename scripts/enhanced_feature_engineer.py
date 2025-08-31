import numpy as np
import pandas as pd
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func

try:
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.trend import MACD
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    # 创建dummy类避免导入错误
    class RSIIndicator:
        def __init__(self, *args, **kwargs): pass
    class BollingerBands:
        def __init__(self, *args, **kwargs): pass
        def bollinger_hband(self): return pd.Series()
        def bollinger_lband(self): return pd.Series()
    class AverageTrueRange:
        def __init__(self, *args, **kwargs): pass
        def average_true_range(self): return pd.Series()
    class MACD:
        def __init__(self, *args, **kwargs): pass
        def macd_diff(self): return pd.Series()

try:
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler
    from sklearn.linear_model import Lasso
    from sklearn.feature_selection import SelectFromModel
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # 创建简化的替代实现
    class PolynomialFeatures:
        def __init__(self, *args, **kwargs): pass
        def fit_transform(self, X): return X
        def get_feature_names_out(self, features): return features
    class SimpleImputer:
        def __init__(self, *args, **kwargs): pass
        def fit_transform(self, X): return X
    class Lasso:
        def __init__(self, *args, **kwargs): pass
        def fit(self, X, y): pass
    class SelectFromModel:
        def __init__(self, *args, **kwargs): pass
        def get_support(self): return [True] * 10

import datetime
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv(): pass

import yaml
import re
import time
from functools import lru_cache
from typing import Union, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# load_dotenv() 已在上面调用

@njit
def calculate_slope(y):
    n = y.size
    if n < 2:
        return 0.0
    sum_x, sum_y, sum_xy, sum_x2 = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        x_val, y_val = i, y[i]
        sum_x += x_val
        sum_y += y_val
        sum_xy += x_val * y_val
        sum_x2 += x_val ** 2
    denominator = n * sum_x2 - sum_x ** 2
    if denominator == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denominator

class EnhancedFeatureEngineer:
    def __init__(self, logger=None, parameter_tuner=None):
        self.base_features = [
            'rsi', 'atr', 'bb_width', 'macd_hist', 'vwap_gap',
            'volume_ma_ratio', 'adx', 'ema_short'
        ]
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.available_features = None
        self.logger = logger
        self.volatility_lookback = 30
        self.parameter_tuner = parameter_tuner
        self.protected_patterns = self.load_protected_patterns()
        
        # 缓存机制
        self.feature_cache = {}
        self.last_data_hash = None
        self.realtime_state = {}
        
        # 性能监控
        self.performance_stats = {
            'batch_times': [],
            'realtime_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }

    def load_protected_patterns(self, config_path='feature_config.yaml'):
        """从 YAML 配置文件加载受保护的特征模式并添加基础受保护列"""
        protected_features = [
            '^datetime$', '^open$', '^high$', '^low$', '^close$', '^volume$',
            '^market_state$', '^scaled_rsi$', '^scaled_macd_hist$'
        ]
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            additional_protected = config.get('protected_features', [])
            protected_features.extend(additional_protected)
        except Exception as e:
            if self.logger:
                self.logger.error(f"加载受保护特征 {config_path} 失败: {str(e)}")
        return [re.compile(p) for p in protected_features]

    def is_protected(self, feature):
        """检查特征是否匹配受保护模式"""
        return any(pattern.match(feature) for pattern in self.protected_patterns)

    def _get_volatility_level(self, close_series):
        """计算动态波动水平，避免未来数据泄漏"""
        returns = close_series.pct_change()
        expanded_vol = returns.expanding(min_periods=30).std().shift(1)
        last_vol = expanded_vol.iloc[-1]
        if last_vol > 0.02:
            return "high"
        elif last_vol > 0.01:
            return "medium"
        return "low"

    def calculate_dynamic_rsi(self, df, close_col='close'):
        """手动计算 RSI，提升健壮性"""
        try:
            delta = df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # 避免除零
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.clip(0, 100)  # 边界处理
            return rsi
        except Exception as e:
            if self.logger:
                self.logger.error(f"RSI 计算失败: {str(e)}")
            return pd.Series(50, index=df.index)  # 返回中性值

    def calculate_dynamic_macd(self, df, close_col='close'):
        vol_level = self._get_volatility_level(df[close_col])
        params = (self.parameter_tuner.get_optimal_macd_params(vol_level) if self.parameter_tuner
                  else {"high": {"fast": 8, "slow": 17, "signal": 7},
                        "medium": {"fast": 12, "slow": 26, "signal": 9},
                        "low": {"fast": 15, "slow": 30, "signal": 12}}.get(vol_level, {"fast": 12, "slow": 26, "signal": 9}))
        try:
            macd = MACD(df[close_col], window_fast=params['fast'], window_slow=params['slow'],
                        window_sign=params['signal'], fillna=True).macd_diff()
            macd = macd.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            if self.logger:
                self.logger.info(f"生成动态 MACD(fast={params['fast']}, slow={params['slow']})")
            return macd
        except Exception as e:
            if self.logger:
                self.logger.error(f"MACD 计算失败: {str(e)}")
            return pd.Series(0, index=df.index)

    def calculate_adx(self, high, low, close, window=14):
        """更健壮的 ADX 计算"""
        try:
            # 手动计算 ADX 避免库问题
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # 计算真实波幅
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # 计算方向指标
            plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
            minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
            
            # 计算方向指数
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
            adx = dx.ewm(alpha=1/window).mean()
            return adx.fillna(0)  # 填充 NaN
        except Exception as e:
            if self.logger:
                self.logger.error(f"ADX 计算失败: {str(e)}")
            return pd.Series(0, index=high.index)  # 返回全 0

    def robust_nan_fixer(self, df, features, logger=None):
        """健壮的 NaN 值处理器"""
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0.0
                if logger:
                    logger.warning(f"创建缺失特征列: {feature}")
                continue
            non_nan_ratio = df[feature].notna().mean()
            if non_nan_ratio == 0:
                df[feature] = 0.0
                if logger:
                    logger.warning(f"特征 {feature} 全为 NaN，已用 0 填充")
            elif non_nan_ratio < 0.8:
                try:
                    median_val = df[feature].median(skipna=True)
                    df[feature] = df[feature].fillna(median_val if not np.isnan(median_val) else 0)
                except:
                    df[feature] = df[feature].fillna(0)
                if logger:
                    logger.warning(f"特征 {feature} 有 {(1 - non_nan_ratio) * 100:.1f}% NaN 值，已用中位数填充")
            else:
                df[feature] = df[feature].ffill().bfill()
        df[features] = df[features].fillna(0)
        return df

    def data_health_report(self, df, stage):
        """生成数据健康报告"""
        report = {
            "stage": stage,
            "sample_count": len(df),
            "nan_report": df.isna().sum().to_dict(),
            "zero_report": (df == 0).sum().to_dict(),
            "inf_report": (df.replace([np.inf, -np.inf], np.nan).isna()).sum().to_dict()
        }
        if self.logger:
            self.logger.info(f"数据健康报告 ({stage}): 样本量: {report['sample_count']}, NaN 值: {report['nan_report']}")
        return report

    def build_features(self, df: pd.DataFrame, is_training: bool = False):
        health_reports = [self.data_health_report(df, "原始输入")]

        if df.empty or not {'open', 'high', 'low', 'close', 'volume'}.issubset(df.columns):
            if self.logger:
                self.logger.error("无效的输入数据或缺少必需的列")
            return pd.DataFrame()

        # 数据清洗
        df_clean = df.replace(0, np.nan).dropna(subset=['open', 'high', 'low', 'close', 'volume'], how='any')
        df_clean = df_clean.ffill().bfill()
        if df_clean.empty:
            if self.logger:
                self.logger.error("清洗后数据为空")
            return pd.DataFrame()

        if df_clean.index.duplicated().any():
            df_clean = df_clean[~df_clean.index.duplicated(keep='first')].sort_index()

        # 确保基础特征列存在
        for feature in self.base_features:
            if feature not in df_clean.columns:
                df_clean[feature] = np.nan

        # 计算技术指标
        df_clean['rsi'] = self.calculate_dynamic_rsi(df_clean)
        df_clean['macd_hist'] = self.calculate_dynamic_macd(df_clean)
        df_clean['adx'] = self.calculate_adx(df_clean['high'], df_clean['low'], df_clean['close'])

        try:
            high_low = df_clean['high'] - df_clean['low']
            high_close = np.abs(df_clean['high'] - df_clean['close'].shift(1))
            low_close = np.abs(df_clean['low'] - df_clean['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df_clean['atr'] = true_range.rolling(14).mean()
        except Exception as e:
            if self.logger:
                self.logger.error(f"ATR 计算失败: {str(e)}")
            df_clean['atr'] = 0

        try:
            bb = BollingerBands(close=df_clean['close'], window=20)
            df_clean['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df_clean['close']
        except Exception as e:
            if self.logger:
                self.logger.error(f"布林带计算失败: {str(e)}")
            df_clean['bb_width'] = 0

        try:
            vwap_numerator = (df_clean['close'] * df_clean['volume']).cumsum()
            vwap_denominator = df_clean['volume'].cumsum().replace(0, 1e-10)
            df_clean['vwap'] = vwap_numerator / vwap_denominator
            df_clean['vwap_gap'] = (df_clean['close'] - df_clean['vwap']) / df_clean['vwap']
        except Exception as e:
            if self.logger:
                self.logger.error(f"VWAP 计算失败: {str(e)}")
            df_clean['vwap_gap'] = 0

        try:
            df_clean['ema_short'] = df_clean['close'].ewm(span=10, adjust=False).mean()
        except Exception as e:
            if self.logger:
                self.logger.error(f"EMA 计算失败: {str(e)}")
            df_clean['ema_short'] = 0

        try:
            vol_ma = df_clean['volume'].rolling(5).mean().replace(0, 1e-10)
            df_clean['volume_ma_ratio'] = df_clean['volume'] / vol_ma
        except Exception as e:
            if self.logger:
                self.logger.error(f"成交量比率计算失败: {str(e)}")
            df_clean['volume_ma_ratio'] = 0

        # 计算市场状态
        try:
            close_series = df_clean['close'].copy()
            vol_level = self._get_volatility_level(close_series)
            atr_window = {"high": 10, "medium": 14, "low": 20}.get(vol_level, 14)
            atr_series = AverageTrueRange(df_clean['high'], df_clean['low'], df_clean['close'], window=atr_window).average_true_range()
            dynamic_atr = atr_series.shift(1).rolling(30).mean()
            rolling_mean_5 = close_series.rolling(5, min_periods=1).mean().shift(1)
            rolling_mean_20 = close_series.rolling(20, min_periods=1).mean().shift(1)
            slope_5 = rolling_mean_5.rolling(5).apply(lambda x: calculate_slope(x.values), raw=False).shift(1)
            threshold = 0.3
            condition1 = (rolling_mean_5 > rolling_mean_20 + threshold * dynamic_atr) & (slope_5 > 0.05)
            condition2 = (rolling_mean_5 < rolling_mean_20 - threshold * dynamic_atr) & (slope_5 < -0.05)
            df_clean['market_state'] = np.select([condition1, condition2], [2, 0], default=1)
            df_clean['market_state'] = df_clean['market_state'].fillna(1).clip(0, 2)  # 修复 NaN
            if self.logger:
                self.logger.info(f"市场状态分布: \n{df_clean['market_state'].value_counts(normalize=True)}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"市场状态生成失败: {str(e)}")
            df_clean['market_state'] = 1

        # NaN 处理
        df_clean = self.robust_nan_fixer(df_clean, self.base_features + ['market_state'], self.logger)

        # 多项式特征生成
        valid_features = [f for f in self.base_features if f in df_clean.columns]
        
        # 确保所有特征都存在且无NaN值
        for feature in valid_features:
            if feature not in df_clean.columns:
                df_clean[feature] = 0.0
            else:
                # 填充NaN值
                df_clean[feature] = df_clean[feature].fillna(0.0)
                # 处理无穷值
                df_clean[feature] = df_clean[feature].replace([np.inf, -np.inf], 0.0)
        
        # 提取特征数据并确保数值类型
        feature_data = df_clean[valid_features].astype(float)
        
        try:
            if SKLEARN_AVAILABLE and len(feature_data) > 0:
                # 应用多项式变换
                X_poly = self.poly.fit_transform(feature_data.values)
                poly_feature_names = [f'poly_{name.replace(" ", "_")}' for name in self.poly.get_feature_names_out(valid_features)]
                df_poly = pd.DataFrame(X_poly, columns=poly_feature_names, index=df_clean.index)
                
                # 再次清理多项式特征中的NaN和无穷值
                df_poly = df_poly.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                # 如果sklearn不可用，创建简化版本
                df_poly = pd.DataFrame(index=df_clean.index)
                for feature in valid_features:
                    df_poly[f'poly_{feature}'] = df_clean[feature] ** 2
                df_poly = df_poly.fillna(0.0)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"多项式特征生成失败，使用简化版本: {str(e)}")
            # 创建简化的多项式特征
            df_poly = pd.DataFrame(index=df_clean.index)
            for feature in valid_features:
                if feature in df_clean.columns:
                    df_poly[f'poly_{feature}'] = df_clean[feature].fillna(0) ** 2
            df_poly = df_poly.fillna(0.0)

        # Lasso 特征选择
        all_features = df_poly.columns.tolist()
        if is_training:
            try:
                y = df_clean['close'].shift(-1) - df_clean['close']
                y = y.dropna()
                X = df_poly.loc[y.index]
                lasso = Lasso(alpha=0.005)
                lasso.fit(X, y)
                selector = SelectFromModel(lasso, prefit=True)
                selected_features = [col for col, selected in zip(df_poly.columns, selector.get_support()) if selected]
                protected_features = [f for f in all_features if self.is_protected(f)]
                self.available_features = list(set(selected_features + protected_features))
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Lasso 特征选择失败: {str(e)}")
                self.available_features = all_features
        else:
            self.available_features = all_features

        # 构建最终特征矩阵
        def build_final_feature_matrix(df_clean, df_poly, available_features):
            """构建包含基础特征和多项式特征的最终矩阵"""
            base_features_df = df_clean[self.base_features].copy()
            poly_features_df = df_poly[available_features].copy()
            df_final = pd.concat([base_features_df, poly_features_df], axis=1)
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in price_cols:
                if col not in df_final.columns and col in df_clean.columns:
                    df_final[col] = df_clean[col]
            if 'market_state' not in df_final.columns:
                df_final['market_state'] = df_clean['market_state']
            return df_final

        df_final = build_final_feature_matrix(df_clean, df_poly, self.available_features)

        # 验证特征
        def validate_features(df, base_features):
            """更灵活的特征验证"""
            must_have = ['close', 'market_state']
            for col in must_have:
                if col not in df.columns:
                    if self.logger:
                        self.logger.error(f"缺失关键列: {col}")
                    if col == 'close':
                        return pd.DataFrame()
                    elif col == 'market_state':
                        df['market_state'] = 1
            if 'market_state' in df.columns:
                df['market_state'] = df['market_state'].fillna(1).astype(int)
            missing_base = [f for f in base_features if f not in df.columns]
            if missing_base:
                if self.logger:
                    self.logger.warning(f"缺失基础特征: {missing_base}")
                for feature in missing_base:
                    df[feature] = 0.0
            return df

        df_final = validate_features(df_final, self.base_features)
        if df_final.empty:
            if self.logger:
                self.logger.error("特征验证失败，返回空 DataFrame")
            return pd.DataFrame()

        if self.logger:
            self.logger.info(f"最终特征矩阵形状: {df_final.shape}, 列: {df_final.columns.tolist()}")
        return df_final

    def create_features(self, data: pd.DataFrame, mode: str = 'batch') -> pd.DataFrame:
        """
        优化的特征生成入口方法
        
        Parameters:
        -----------
        data : pd.DataFrame
            输入数据（清洗后的数据）
        mode : str
            处理模式 - 'batch' 用于历史数据完整计算，'realtime' 用于实时增量计算
            
        Returns:
        --------
        pd.DataFrame
            特征数据集
        """
        start_time = time.time()
        
        try:
            if mode == 'realtime':
                result = self._fast_features(data)
                self.performance_stats['realtime_times'].append(time.time() - start_time)
            else:
                result = self._full_features(data)
                self.performance_stats['batch_times'].append(time.time() - start_time)
            
            # 记录性能统计
            if self.logger:
                avg_time = (self.performance_stats[f'{mode}_times'][-1] 
                           if self.performance_stats[f'{mode}_times'] else 0)
                self.logger.info(f"{mode}模式特征生成完成，耗时: {avg_time:.3f}秒")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"特征生成失败 ({mode}模式): {str(e)}")
            return pd.DataFrame()
    
    def _fast_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """实时模式：优化计算实时特征"""
        if data.empty:
            return pd.DataFrame()
        
        # 检查缓存
        data_hash = self._get_data_hash(data)
        cache_key = f"realtime_{data_hash}"
        
        if cache_key in self.feature_cache:
            self.performance_stats['cache_hits'] += 1
            if self.logger:
                self.logger.debug("使用缓存的实时特征")
            return self.feature_cache[cache_key].copy()
        
        self.performance_stats['cache_misses'] += 1
        
        # 智能窗口大小选择
        if len(data) <= 50:
            # 小数据集，使用全部数据
            window_size = len(data)
        elif len(data) <= 200:
            # 中等数据集，使用大部分数据
            window_size = min(80, len(data))
        else:
            # 大数据集，只使用最新数据
            window_size = min(60, len(data))
        
        recent_data = data.tail(window_size).copy()
        
        # 快速特征计算（简化版本）
        features = self._calculate_fast_features(recent_data)
        
        # 缓存结果（限制缓存大小）
        if len(self.feature_cache) > 10:  # 限制缓存条目数量
            # 删除最旧的缓存项
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
        
        self.feature_cache[cache_key] = features
        self.last_data_hash = data_hash
        
        return features
    
    def _full_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """批处理模式：完整历史特征计算"""
        if data.empty:
            return pd.DataFrame()
        
        # 检查缓存
        data_hash = self._get_data_hash(data)
        if data_hash == self.last_data_hash and 'batch_features' in self.feature_cache:
            self.performance_stats['cache_hits'] += 1
            if self.logger:
                self.logger.debug("使用缓存的批处理特征")
            return self.feature_cache['batch_features'].copy()
        
        self.performance_stats['cache_misses'] += 1
        
        # 完整特征计算
        features = self.build_features(data, is_training=True)
        
        # 缓存结果
        self.feature_cache['batch_features'] = features
        self.last_data_hash = data_hash
        
        return features
    
    def _get_data_hash(self, data: pd.DataFrame) -> str:
        """计算数据哈希用于缓存检查"""
        try:
            # 使用数据形状、最后几行数据和时间戳的组合作为哈希
            recent_data = data.tail(5) if len(data) > 5 else data
            shape_str = f"{data.shape[0]}x{data.shape[1]}"
            
            # 获取最后几行的关键数据
            if not recent_data.empty and 'close' in recent_data.columns:
                last_values = recent_data['close'].values[-3:] if len(recent_data) >= 3 else recent_data['close'].values
                values_str = '_'.join([f"{v:.4f}" for v in last_values])
                return f"{shape_str}_{values_str}"
            else:
                return f"{shape_str}_{time.time()}"
        except Exception:
            return str(time.time())  # 如果哈希失败，使用时间戳
    

    
    def _calculate_fast_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """快速特征计算（简化版本）"""
        try:
            df = data.copy()
            
            # 只计算最关键的特征，使用更小的窗口
            df['rsi'] = self._fast_rsi(df['close'], window=7)  # 减小窗口
            df['macd_hist'] = self._fast_macd(df['close'])
            df['atr'] = self._fast_atr(df['high'], df['low'], df['close'], window=7)
            df['ema_short'] = df['close'].ewm(span=5).mean()  # 更快的EMA
            
            # 简单的波动率和成交量指标
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(5, min_periods=1).mean()
            df['price_change'] = df['close'].pct_change()
            
            # 市场状态简化版本
            df['market_state'] = self._simple_market_state(df['close'])
            
            # 填充缺失值
            df = df.ffill().fillna(0)
            
            return df.tail(1)  # 只返回最新行
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"快速特征计算失败: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_full_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """完整特征计算"""
        return self.build_features(data, is_training=True)
    
    def _fast_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """快速RSI计算"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()
            
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except Exception:
            return pd.Series(50, index=prices.index)
    
    def _fast_macd(self, prices: pd.Series, fast: int = 8, slow: int = 17) -> pd.Series:
        """快速MACD计算"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            return ema_fast - ema_slow
        except Exception:
            return pd.Series(0, index=prices.index)
    
    def _fast_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """快速ATR计算"""
        try:
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window, min_periods=1).mean()
        except Exception:
            return pd.Series(0, index=high.index)
    
    def _simple_market_state(self, prices: pd.Series) -> pd.Series:
        """简化的市场状态计算"""
        try:
            sma_short = prices.rolling(5, min_periods=1).mean()
            sma_long = prices.rolling(20, min_periods=1).mean()
            
            conditions = [
                sma_short > sma_long * 1.002,  # 上涨趋势
                sma_short < sma_long * 0.998   # 下跌趋势
            ]
            choices = [2, 0]  # 2=上涨, 0=下跌, 1=横盘
            
            return pd.Series(np.select(conditions, choices, default=1), index=prices.index)
        except Exception:
            return pd.Series(1, index=prices.index)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        
        if stats['batch_times']:
            stats['avg_batch_time'] = np.mean(stats['batch_times'])
            stats['max_batch_time'] = np.max(stats['batch_times'])
        
        if stats['realtime_times']:
            stats['avg_realtime_time'] = np.mean(stats['realtime_times'])
            stats['max_realtime_time'] = np.max(stats['realtime_times'])
        
        stats['cache_hit_rate'] = (stats['cache_hits'] / 
                                  max(stats['cache_hits'] + stats['cache_misses'], 1))
        
        return stats
    
    def clear_cache(self):
        """清除缓存"""
        self.feature_cache.clear()
        self.last_data_hash = None
        if self.logger:
            self.logger.info("特征缓存已清除")

if __name__ == "__main__":
    from logger_setup import setup_logging
    from twelve_data_client import TwelveDataClient
    from parameter_tuner import ParameterTuner

    logger = setup_logging()
    tuner = ParameterTuner()
    fe = EnhancedFeatureEngineer(logger=logger, parameter_tuner=tuner)

    api_key = os.getenv("TWELVE_DATA_API_KEY")
    client = TwelveDataClient(api_key=api_key)
    latest_trading_day = datetime.date.today().strftime("%Y-%m-%d")
    df = client.get_historical_data(
        symbol="AAPL",
        interval="30min",
        start_date="2020-01-01",
        end_date=latest_trading_day
    )
    features = fe.build_features(df, is_training=True)
    if not features.empty:
        features.reset_index().to_feather("features_aapl_30min.feather")
        logger.info(f"特征保存成功，样本量: {len(features)}")
    else:
        logger.error("特征工程失败，输出为空！")