import numpy as np
import pandas as pd
from numba import njit
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import datetime
import os
from dotenv import load_dotenv
import yaml
import re

load_dotenv()  # 从 .env 文件加载环境变量

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
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        base_features_clean = imputer.fit_transform(df_clean[valid_features])
        X_poly = self.poly.fit_transform(base_features_clean)
        poly_feature_names = [f'poly_{name.replace(" ", "_")}' for name in self.poly.get_feature_names_out(valid_features)]
        df_poly = pd.DataFrame(X_poly, columns=poly_feature_names, index=df_clean.index)

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