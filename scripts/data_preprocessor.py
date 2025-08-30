import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import logging
import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pytz import timezone
import joblib

def safe_setup_logging():
    """确保日志系统至少输出到控制台"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
    return logger

def clean_extreme_values(df):
    """使用 IQR 处理极端值"""
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            q1, q3 = df[col].quantile(0.05), df[col].quantile(0.95)
            iqr = q3 - q1
            df[col] = df[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)
    return df

class DataPreprocessor:
    def __init__(self, logger=None):
        self.logger = logger or safe_setup_logging()
        self.feature_cols = None
        self.volatility_window = 30
        self.window_size = None
        self.scaler = RobustScaler()
        self.class_weights = None

    def _calculate_volatility(self, close_series: pd.Series) -> pd.Series:
        """计算价格波动率"""
        returns = close_series.pct_change().dropna()
        if len(returns) < self.volatility_window:
            return pd.Series(np.nan, index=close_series.index)
        volatility = returns.rolling(
            window=self.volatility_window, 
            min_periods=max(10, int(self.volatility_window * 0.3))
        ).std()
        return volatility.reindex(close_series.index)

    def _get_dynamic_window(self, volatility: float, base_window: int) -> int:
        """根据波动率返回动态窗口大小"""
        if np.isnan(volatility) or volatility <= 0:
            return base_window
        if volatility > 0.03:
            return int(base_window * 0.5)
        elif volatility > 0.015:
            return int(base_window * 0.75)
        else:
            return base_window

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行特征工程"""
        # 基础特征
        df['adx_above_25'] = (df['adx'] > 25).astype(int)
        df['atr_ratio'] = df['atr'] / df['close'].rolling(20, min_periods=5).mean()
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        # 动量特征
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['volume_change_5'] = df['volume'].pct_change(5)
        
        # 波动率特征
        df['volatility'] = self._calculate_volatility(df['close'])
        df['volatility_rank'] = df['volatility'].rolling(50).rank(pct=True)
        
        # 清理无穷值和缺失值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().dropna()
        
        # 添加时间特征使用索引
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        self.logger.info(f"特征工程完成，新增特征: {set(df.columns) - {'close', 'market_state'}}")
        return df

    def create_rolling_dataset(self, features: pd.DataFrame, base_window: int = 60, min_samples: int = 30):
        """创建滚动窗口数据集，防止数据泄漏"""
        try:
            # 验证必需列
            required_columns = ['close', 'market_state']
            for col in required_columns:
                if col not in features.columns:
                    self.logger.error(f"缺少必需列: {col}")
                    return np.array([]), np.array([]), None
            
            # 时间序列排序
            features = features.sort_index()
            
            # 波动率计算
            volatility_series = self._calculate_volatility(features['close'])
            if volatility_series.isna().any():
                volatility_series = volatility_series.fillna(0)
                self.logger.warning("波动率序列包含 NaN，已填充为 0")

            sequences, labels, timestamps = [], [], []
            feature_cols = features.columns.drop('market_state')
            
            # 创建滚动窗口
            for i in range(base_window, len(features) - 1):
                vol = volatility_series.iloc[i]
                dynamic_window = self._get_dynamic_window(vol, base_window)
                dynamic_window = max(min_samples, min(dynamic_window, i))
                
                # 提取序列和标签
                sequence = features.iloc[i - dynamic_window:i][feature_cols].values
                if len(sequence) >= min_samples:
                    sequences.append(sequence)
                    labels.append(features.iloc[i + 1]['market_state'])
                    timestamps.append(features.index[i + 1])
            
            if not sequences:
                self.logger.error("未生成有效序列")
                return np.array([]), np.array([]), None
            
            # 填充序列
            max_len = max(len(seq) for seq in sequences)
            X = pad_sequences(sequences, maxlen=max_len, dtype='float32', padding='post', value=0)
            y = np.array(labels, dtype=np.int32)
            
            # 保存元数据
            self.feature_cols = feature_cols
            self.window_size = max_len
            self.logger.info(f"生成序列数: {len(sequences)}, 填充后形状: {X.shape}")
            self.logger.info(f"标签分布: {pd.Series(y).value_counts()}")

            return X, y, timestamps
        
        except Exception as e:
            self.logger.exception(f"创建滚动数据集失败: {str(e)}")
            return np.array([]), np.array([]), None

    def normalize_data(self, X):
        """标准化3D时间序列数据"""
        original_shape = X.shape
        # 转换为2D (samples * timesteps, features)
        X_2d = X.reshape(-1, original_shape[-1])
        
        # 处理无穷值和缺失值
        imp = SimpleImputer(strategy='mean')
        X_2d = imp.fit_transform(X_2d)
        
        # 标准化
        X_normalized = self.scaler.fit_transform(X_2d)
        
        # 恢复原始形状
        return X_normalized.reshape(original_shape)

    def handle_class_imbalance(self, y):
        """计算类别权重处理不平衡"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = dict(zip(classes, weights))
        self.logger.info(f"计算类别权重: {self.class_weights}")
        return self.class_weights

    def split_dataset(self, X, y, timestamps, test_size=0.2):
        """按时间顺序拆分数据集"""
        # 按时间排序确保拆分正确
        sorted_indices = np.argsort(timestamps)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # 时间顺序拆分
        split_idx = int(len(X_sorted) * (1 - test_size))
        X_train, X_test = X_sorted[:split_idx], X_sorted[split_idx:]
        y_train, y_test = y_sorted[:split_idx], y_sorted[split_idx:]
        
        self.logger.info(f"数据集拆分 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        self.logger.info(f"训练集时间范围: {timestamps[0]} 到 {timestamps[split_idx-1]}")
        self.logger.info(f"测试集时间范围: {timestamps[split_idx]} 到 {timestamps[-1]}")
        
        return X_train, X_test, y_train, y_test

    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir="preprocessed_data"):
        """保存预处理后的数据集"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        
        # 保存元数据
        joblib.dump({
            'feature_cols': self.feature_cols,
            'window_size': self.window_size,
            'class_weights': self.class_weights,
            'scaler': self.scaler
        }, os.path.join(output_dir, "metadata.pkl"))
        
        self.logger.info(f"预处理数据保存至: {output_dir}")
        return output_dir

if __name__ == "__main__":
    try:
        logger = safe_setup_logging()
        logger.info("===== 数据预处理开始 =====")
        
        dp = DataPreprocessor(logger=logger)
        feature_file = "features_aapl_30min.feather"

        if not os.path.exists(feature_file):
            logger.error(f"特征文件未找到: {feature_file}")
            sys.exit(1)
        
        features = pd.read_feather(feature_file)
        
        # 时区处理
        ny_tz = timezone('America/New_York')
        if 'datetime' in features.columns:
            features = features.set_index('datetime', drop=True)
        elif not isinstance(features.index, pd.DatetimeIndex):
            features['datetime'] = pd.date_range(start='2020-01-01', periods=len(features), freq='30min')
            features = features.set_index('datetime', drop=True)
        
        if features.index.tz is None:
            features.index = features.index.tz_localize(ny_tz)
        else:
            features.index = features.index.tz_convert(ny_tz)

        # 数据验证
        if features.index.duplicated().any():
            logger.warning("检测到重复的时间戳，正在删除...")
            features = features[~features.index.duplicated(keep='first')]
        if not features.index.is_monotonic_increasing:
            logger.warning("时间序列未按升序排列，正在排序...")
            features = features.sort_index()

        # 特征工程
        features = clean_extreme_values(features)
        features = dp.feature_engineering(features)

        # 创建滚动窗口数据集
        X, y, timestamps = dp.create_rolling_dataset(features, base_window=60, min_samples=30)
        
        if X.size == 0 or y.size == 0:
            logger.error("未能生成有效数据集，退出")
            sys.exit(1)

        # 标准化数据
        X_normalized = dp.normalize_data(X)
        logger.info(f"标准化后数据形状: {X_normalized.shape}")

        # 处理类别不平衡
        dp.handle_class_imbalance(y)

        # 拆分数据集
        X_train, X_test, y_train, y_test = dp.split_dataset(
            X_normalized, y, timestamps, test_size=0.2
        )

        # 保存预处理结果
        dp.save_preprocessed_data(X_train, X_test, y_train, y_test)
        
        logger.info("===== 数据预处理完成 =====")
    
    except Exception as e:
        logger.error(f"预处理异常: {str(e)}")
        logger.exception("异常详情:")
        sys.exit(1)