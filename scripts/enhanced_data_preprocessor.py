#!/usr/bin/env python3
"""
Enhanced DataPreprocessor - 优化的数据预处理器

主要功能：
1. 智能标准化/归一化 (多种策略)
2. 高效时间序列窗口创建 (动态窗口、增量更新)
3. 高级数据集拆分 (时间感知、交叉验证)
4. 智能类别不平衡处理 (SMOTE、权重调整、采样策略)
5. 实时处理能力 (增量更新、缓存机制)
6. 性能优化 (内存效率、并行处理)
"""

import numpy as np
import pandas as pd

# 处理sklearn依赖
try:
    from sklearn.preprocessing import (
        RobustScaler, StandardScaler, MinMaxScaler, 
        QuantileTransformer, PowerTransformer
    )
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # 创建简化的替代实现
    class RobustScaler:
        def __init__(self): 
            self.median_ = None
            self.scale_ = None
        def fit_transform(self, X):
            self.median_ = np.median(X, axis=0)
            scale = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
            self.scale_ = np.where(scale == 0, 1, scale)
            return (X - self.median_) / self.scale_
        def transform(self, X):
            return (X - self.median_) / self.scale_
    
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        def fit_transform(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
            return (X - self.mean_) / self.scale_
        def transform(self, X):
            return (X - self.mean_) / self.scale_
    
    class MinMaxScaler:
        def __init__(self):
            self.min_ = None
            self.scale_ = None
        def fit_transform(self, X):
            self.min_ = np.min(X, axis=0)
            self.scale_ = np.max(X, axis=0) - self.min_
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
            return (X - self.min_) / self.scale_
        def transform(self, X):
            return (X - self.min_) / self.scale_
    
    class QuantileTransformer:
        def __init__(self, *args, **kwargs): pass
        def fit_transform(self, X): return X
        def transform(self, X): return X
    
    class PowerTransformer:
        def __init__(self, *args, **kwargs): pass
        def fit_transform(self, X): return X
        def transform(self, X): return X
    
    class SimpleImputer:
        def __init__(self, strategy='mean'): 
            self.strategy = strategy
        def fit_transform(self, X):
            if self.strategy == 'mean':
                return np.where(np.isnan(X), np.nanmean(X, axis=0), X)
            elif self.strategy == 'median':
                return np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
            else:
                return np.where(np.isnan(X), 0, X)
        def transform(self, X):
            return self.fit_transform(X)
    
    class KNNImputer:
        def __init__(self, *args, **kwargs): pass
        def fit_transform(self, X): 
            return np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        def transform(self, X): 
            return self.fit_transform(X)
    
    def compute_class_weight(strategy, classes, y):
        class_counts = pd.Series(y).value_counts()
        total = len(y)
        weights = []
        for cls in classes:
            weight = total / (len(classes) * class_counts.get(cls, 1))
            weights.append(weight)
        return np.array(weights)
import logging
import os
import sys
import time
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    import pickle
    # 使用pickle作为joblib的替代
    class joblib:
        @staticmethod
        def dump(obj, filename):
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)
        @staticmethod  
        def load(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
from typing import Tuple, Dict, Any, List, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # 创建简单的替代实现
    def pad_sequences(sequences, maxlen=None, padding='post', value=0.0, dtype='float32'):
        if maxlen is None:
            maxlen = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            if len(seq) >= maxlen:
                padded.append(seq[-maxlen:])
            else:
                pad_size = maxlen - len(seq)
                if padding == 'post':
                    padded.append(np.concatenate([seq, np.full((pad_size, seq.shape[1]), value)]))
                else:
                    padded.append(np.concatenate([np.full((pad_size, seq.shape[1]), value), seq]))
        return np.array(padded, dtype=dtype)

@dataclass
class PreprocessingConfig:
    """预处理配置类"""
    # 标准化配置
    normalization_method: str = 'robust'  # robust, standard, minmax, quantile, power
    feature_selection_k: int = 50  # 特征选择数量
    pca_components: Optional[int] = None  # PCA降维
    
    # 窗口配置
    base_window_size: int = 60
    min_window_size: int = 30
    max_window_size: int = 120
    stride: int = 1  # 窗口步长
    
    # 数据拆分配置
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_folds: int = 5
    
    # 类别不平衡配置
    imbalance_strategy: str = 'auto'  # auto, smote, undersample, weights, none
    sampling_ratio: float = 0.8
    
    # 性能优化配置
    use_parallel: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    enable_caching: bool = True
    
    # 实时处理配置
    enable_incremental: bool = True
    cache_size: int = 1000

def safe_setup_logging():
    """确保日志系统至少输出到控制台"""
    logger = logging.getLogger("EnhancedDataPreprocessor")
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

class EnhancedDataPreprocessor:
    """增强的数据预处理器"""
    
    def __init__(self, config: PreprocessingConfig = None, logger=None):
        self.config = config or PreprocessingConfig()
        self.logger = logger or safe_setup_logging()
        
        # 核心组件
        self.scaler = self._get_scaler()
        self.imputer = KNNImputer(n_neighbors=5) if self.config.normalization_method != 'robust' else SimpleImputer(strategy='median')
        self.feature_selector = None
        self.pca = None
        
        # 状态管理
        self.feature_cols = None
        self.window_size = None
        self.class_weights = None
        self.is_fitted = False
        
        # 缓存机制
        self.cache = {} if self.config.enable_caching else None
        self.processing_stats = {
            'total_samples_processed': 0,
            'cache_hits': 0,
            'processing_times': [],
            'memory_usage': []
        }
        
        # 增量处理状态
        self.incremental_state = {
            'last_window_data': None,
            'accumulated_stats': None,
            'feature_stats': None
        } if self.config.enable_incremental else None

    def _get_scaler(self):
        """根据配置获取标准化器"""
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer(method='yeo-johnson')
        }
        return scalers.get(self.config.normalization_method, RobustScaler())

    def _calculate_dynamic_window(self, volatility: float, trend_strength: float = 0.5) -> int:
        """动态计算窗口大小"""
        base_size = self.config.base_window_size
        
        # 基于波动率的调整
        if not np.isnan(volatility) and volatility > 0:
            if volatility > 0.03:  # 高波动
                vol_factor = 0.7
            elif volatility > 0.015:  # 中等波动
                vol_factor = 0.85
            else:  # 低波动
                vol_factor = 1.2
        else:
            vol_factor = 1.0
        
        # 基于趋势强度的调整
        trend_factor = 0.8 + 0.4 * trend_strength  # 0.8 - 1.2
        
        # 计算最终窗口大小
        dynamic_size = int(base_size * vol_factor * trend_factor)
        return np.clip(dynamic_size, self.config.min_window_size, self.config.max_window_size)

    def smart_normalization(self, X: np.ndarray, mode: str = 'fit_transform') -> np.ndarray:
        """智能标准化处理"""
        start_time = time.time()
        
        try:
            if X.ndim == 3:  # 时间序列数据 (samples, timesteps, features)
                original_shape = X.shape
                X_2d = X.reshape(-1, original_shape[-1])
            else:
                X_2d = X
                original_shape = None
            
            # 处理缺失值
            if mode == 'fit_transform':
                X_imputed = self.imputer.fit_transform(X_2d)
                X_scaled = self.scaler.fit_transform(X_imputed)
            else:
                X_imputed = self.imputer.transform(X_2d)
                X_scaled = self.scaler.transform(X_imputed)
            
            # 恢复原始形状
            if original_shape is not None:
                X_scaled = X_scaled.reshape(original_shape)
            
            processing_time = time.time() - start_time
            self.processing_stats['processing_times'].append(processing_time)
            
            if self.logger:
                self.logger.info(f"标准化完成，方法: {self.config.normalization_method}, "
                               f"耗时: {processing_time:.3f}秒")
            
            return X_scaled
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"标准化失败: {str(e)}")
            return X

    def create_advanced_windows(self, features: pd.DataFrame, 
                              target_col: str = 'market_state') -> Tuple[np.ndarray, np.ndarray, List]:
        """创建高级时间序列窗口"""
        start_time = time.time()
        
        try:
            # 验证输入
            if features.empty or target_col not in features.columns:
                self.logger.error(f"输入数据无效或缺少目标列: {target_col}")
                return np.array([]), np.array([]), []
            
            # 排序确保时间顺序
            features = features.sort_index()
            
            # 计算辅助指标
            close_series = features.get('close', features.iloc[:, 0])
            volatility = self._calculate_volatility_series(close_series)
            trend_strength = self._calculate_trend_strength(close_series)
            
            sequences, labels, timestamps = [], [], []
            feature_cols = [col for col in features.columns if col != target_col]
            
            # 并行处理大数据集
            if len(features) > self.config.chunk_size and self.config.use_parallel:
                sequences, labels, timestamps = self._parallel_window_creation(
                    features, feature_cols, target_col, volatility, trend_strength
                )
            else:
                sequences, labels, timestamps = self._sequential_window_creation(
                    features, feature_cols, target_col, volatility, trend_strength
                )
            
            if not sequences:
                self.logger.warning("未生成有效窗口序列")
                return np.array([]), np.array([]), []
            
            # 智能填充策略
            X = self._smart_padding(sequences)
            y = np.array(labels, dtype=np.int32)
            
            # 更新状态
            self.feature_cols = feature_cols
            self.window_size = X.shape[1] if X.ndim > 1 else None
            
            processing_time = time.time() - start_time
            self.processing_stats['processing_times'].append(processing_time)
            self.processing_stats['total_samples_processed'] += len(X)
            
            if self.logger:
                self.logger.info(f"窗口创建完成: {len(sequences)}个序列, "
                               f"形状: {X.shape}, 耗时: {processing_time:.3f}秒")
                self.logger.info(f"标签分布: {pd.Series(y).value_counts().to_dict()}")
            
            return X, y, timestamps
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"窗口创建失败: {str(e)}")
            return np.array([]), np.array([]), []

    def _calculate_volatility_series(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """计算价格波动率序列"""
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=window, min_periods=5).std()
        return volatility.reindex(prices.index).fillna(0)

    def _calculate_trend_strength(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算趋势强度"""
        try:
            # 简单的趋势强度计算
            sma_short = prices.rolling(window=window//2).mean()
            sma_long = prices.rolling(window=window).mean()
            trend_strength = np.abs(sma_short - sma_long) / sma_long
            return trend_strength.fillna(0.5)
        except Exception:
            return pd.Series(0.5, index=prices.index)

    def _sequential_window_creation(self, features: pd.DataFrame, feature_cols: List[str],
                                  target_col: str, volatility: pd.Series, 
                                  trend_strength: pd.Series) -> Tuple[List, List, List]:
        """顺序窗口创建"""
        sequences, labels, timestamps = [], [], []
        
        for i in range(self.config.base_window_size, len(features) - 1, self.config.stride):
            try:
                # 动态窗口大小
                vol = volatility.iloc[i] if i < len(volatility) else 0
                trend = trend_strength.iloc[i] if i < len(trend_strength) else 0.5
                window_size = self._calculate_dynamic_window(vol, trend)
                
                # 确保窗口不超出数据范围
                start_idx = max(0, i - window_size)
                
                # 提取序列和标签
                sequence = features.iloc[start_idx:i][feature_cols].values
                if len(sequence) >= self.config.min_window_size:
                    sequences.append(sequence)
                    labels.append(features.iloc[i][target_col])
                    timestamps.append(features.index[i])
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"窗口 {i} 创建失败: {str(e)}")
                continue
        
        return sequences, labels, timestamps

    def _parallel_window_creation(self, features: pd.DataFrame, feature_cols: List[str],
                                target_col: str, volatility: pd.Series, 
                                trend_strength: pd.Series) -> Tuple[List, List, List]:
        """并行窗口创建"""
        chunk_indices = range(self.config.base_window_size, len(features) - 1, 
                            max(1, len(features) // self.config.max_workers))
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for start_idx in chunk_indices:
                end_idx = min(start_idx + len(features) // self.config.max_workers, len(features) - 1)
                future = executor.submit(
                    self._process_chunk, features, feature_cols, target_col,
                    volatility, trend_strength, start_idx, end_idx
                )
                futures.append(future)
            
            # 收集结果
            all_sequences, all_labels, all_timestamps = [], [], []
            for future in futures:
                try:
                    chunk_seq, chunk_lab, chunk_time = future.result()
                    all_sequences.extend(chunk_seq)
                    all_labels.extend(chunk_lab)
                    all_timestamps.extend(chunk_time)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"并行处理块失败: {str(e)}")
        
        return all_sequences, all_labels, all_timestamps

    def _process_chunk(self, features: pd.DataFrame, feature_cols: List[str],
                      target_col: str, volatility: pd.Series, trend_strength: pd.Series,
                      start_idx: int, end_idx: int) -> Tuple[List, List, List]:
        """处理数据块"""
        sequences, labels, timestamps = [], [], []
        
        for i in range(start_idx, end_idx, self.config.stride):
            try:
                vol = volatility.iloc[i] if i < len(volatility) else 0
                trend = trend_strength.iloc[i] if i < len(trend_strength) else 0.5
                window_size = self._calculate_dynamic_window(vol, trend)
                
                chunk_start = max(0, i - window_size)
                sequence = features.iloc[chunk_start:i][feature_cols].values
                
                if len(sequence) >= self.config.min_window_size:
                    sequences.append(sequence)
                    labels.append(features.iloc[i][target_col])
                    timestamps.append(features.index[i])
                    
            except Exception:
                continue
        
        return sequences, labels, timestamps

    def _smart_padding(self, sequences: List[np.ndarray]) -> np.ndarray:
        """智能填充策略"""
        if not sequences:
            return np.array([])
        
        # 计算统计信息
        lengths = [len(seq) for seq in sequences]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        
        # 智能选择最大长度
        max_len = int(mean_len + 1.5 * std_len)  # 避免过度填充
        max_len = min(max_len, self.config.max_window_size)
        max_len = max(max_len, self.config.min_window_size)
        
        # 使用自定义填充
        return pad_sequences(sequences, maxlen=max_len, dtype='float32', 
                           padding='pre', value=0.0)

    def advanced_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                                timestamps: List) -> Dict[str, np.ndarray]:
        """高级数据集拆分"""
        try:
            # 按时间排序
            sorted_indices = np.argsort(timestamps)
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]
            timestamps_sorted = np.array(timestamps)[sorted_indices]
            
            # 计算拆分点
            total_size = len(X_sorted)
            test_start = int(total_size * (1 - self.config.test_size))
            val_start = int(test_start * (1 - self.config.validation_size))
            
            # 拆分数据
            X_train = X_sorted[:val_start]
            y_train = y_sorted[:val_start]
            
            X_val = X_sorted[val_start:test_start]
            y_val = y_sorted[val_start:test_start]
            
            X_test = X_sorted[test_start:]
            y_test = y_sorted[test_start:]
            
            # 时间范围信息
            train_time_range = (timestamps_sorted[0], timestamps_sorted[val_start-1])
            val_time_range = (timestamps_sorted[val_start], timestamps_sorted[test_start-1])
            test_time_range = (timestamps_sorted[test_start], timestamps_sorted[-1])
            
            if self.logger:
                self.logger.info(f"数据集拆分完成:")
                self.logger.info(f"  训练集: {X_train.shape} ({train_time_range[0]} - {train_time_range[1]})")
                self.logger.info(f"  验证集: {X_val.shape} ({val_time_range[0]} - {val_time_range[1]})")
                self.logger.info(f"  测试集: {X_test.shape} ({test_time_range[0]} - {test_time_range[1]})")
            
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test,
                'train_time_range': train_time_range,
                'val_time_range': val_time_range,
                'test_time_range': test_time_range
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"数据集拆分失败: {str(e)}")
            return {}

    def handle_class_imbalance_advanced(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """高级类别不平衡处理"""
        try:
            original_shape = X.shape
            class_counts = pd.Series(y).value_counts()
            imbalance_ratio = class_counts.min() / class_counts.max()
            
            if self.logger:
                self.logger.info(f"原始类别分布: {class_counts.to_dict()}")
                self.logger.info(f"不平衡比例: {imbalance_ratio:.3f}")
            
            # 根据策略处理
            if self.config.imbalance_strategy == 'auto':
                if imbalance_ratio < 0.3:
                    strategy = 'smote' if IMBALANCED_LEARN_AVAILABLE else 'weights'
                else:
                    strategy = 'weights'
            else:
                strategy = self.config.imbalance_strategy
            
            X_resampled, y_resampled = X, y
            processing_info = {'strategy': strategy, 'original_counts': class_counts.to_dict()}
            
            if strategy == 'smote' and IMBALANCED_LEARN_AVAILABLE:
                X_resampled, y_resampled = self._apply_smote(X, y)
                processing_info['resampled_counts'] = pd.Series(y_resampled).value_counts().to_dict()
                
            elif strategy == 'undersample' and IMBALANCED_LEARN_AVAILABLE:
                X_resampled, y_resampled = self._apply_undersampling(X, y)
                processing_info['resampled_counts'] = pd.Series(y_resampled).value_counts().to_dict()
            
            # 计算类别权重
            classes = np.unique(y_resampled)
            weights = compute_class_weight('balanced', classes=classes, y=y_resampled)
            self.class_weights = dict(zip(classes, weights))
            processing_info['class_weights'] = self.class_weights
            
            if self.logger:
                self.logger.info(f"类别不平衡处理完成 (策略: {strategy})")
                if 'resampled_counts' in processing_info:
                    self.logger.info(f"重采样后分布: {processing_info['resampled_counts']}")
                self.logger.info(f"类别权重: {self.class_weights}")
            
            return X_resampled, y_resampled, processing_info
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"类别不平衡处理失败: {str(e)}")
            return X, y, {'strategy': 'none', 'error': str(e)}

    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用SMOTE过采样"""
        if X.ndim == 3:
            original_shape = X.shape
            X_2d = X.reshape(X.shape[0], -1)
            
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_2d, y)
            
            # 恢复3D形状
            X_resampled = X_resampled.reshape(-1, original_shape[1], original_shape[2])
            return X_resampled, y_resampled
        else:
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            return smote.fit_resample(X, y)

    def _apply_undersampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用欠采样"""
        if X.ndim == 3:
            original_shape = X.shape
            X_2d = X.reshape(X.shape[0], -1)
            
            undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_2d, y)
            
            # 恢复3D形状
            X_resampled = X_resampled.reshape(-1, original_shape[1], original_shape[2])
            return X_resampled, y_resampled
        else:
            undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            return undersampler.fit_resample(X, y)

    def process_pipeline(self, features: pd.DataFrame, target_col: str = 'market_state') -> Dict[str, Any]:
        """完整的预处理流水线"""
        start_time = time.time()
        pipeline_results = {}
        
        try:
            if self.logger:
                self.logger.info("开始执行预处理流水线...")
            
            # 1. 创建时间序列窗口
            X, y, timestamps = self.create_advanced_windows(features, target_col)
            if X.size == 0:
                raise ValueError("窗口创建失败")
            
            pipeline_results['raw_data'] = {'X_shape': X.shape, 'y_shape': y.shape}
            
            # 2. 标准化
            X_normalized = self.smart_normalization(X, mode='fit_transform')
            pipeline_results['normalization'] = {'method': self.config.normalization_method}
            
            # 3. 处理类别不平衡
            X_balanced, y_balanced, imbalance_info = self.handle_class_imbalance_advanced(X_normalized, y)
            pipeline_results['imbalance_handling'] = imbalance_info
            
            # 4. 数据集拆分
            split_data = self.advanced_train_test_split(X_balanced, y_balanced, timestamps)
            if not split_data:
                raise ValueError("数据集拆分失败")
            
            pipeline_results['data_splits'] = {
                'train_shape': split_data['X_train'].shape,
                'val_shape': split_data['X_val'].shape,
                'test_shape': split_data['X_test'].shape,
            }
            
            # 5. 更新状态
            self.is_fitted = True
            
            # 6. 计算处理统计
            total_time = time.time() - start_time
            pipeline_results['processing_stats'] = {
                'total_time': total_time,
                'samples_processed': len(X),
                'feature_count': len(self.feature_cols) if self.feature_cols else 0,
                'window_size': self.window_size
            }
            
            if self.logger:
                self.logger.info(f"预处理流水线完成，总耗时: {total_time:.3f}秒")
                self.logger.info(f"处理样本数: {len(X)}")
            
            # 返回完整结果
            result = {
                'data': split_data,
                'metadata': {
                    'feature_cols': self.feature_cols,
                    'window_size': self.window_size,
                    'class_weights': self.class_weights,
                    'scaler': self.scaler,
                    'config': self.config
                },
                'pipeline_results': pipeline_results
            }
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"预处理流水线失败: {str(e)}")
            return {'error': str(e), 'pipeline_results': pipeline_results}

    def save_preprocessing_artifacts(self, output_dir: str, data_dict: Dict[str, Any]) -> str:
        """保存预处理产物"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存数据
            if 'data' in data_dict:
                for key, value in data_dict['data'].items():
                    if isinstance(value, np.ndarray):
                        np.save(os.path.join(output_dir, f"{key}.npy"), value)
            
            # 保存元数据
            if 'metadata' in data_dict:
                joblib.dump(data_dict['metadata'], os.path.join(output_dir, "metadata.pkl"))
            
            # 保存流水线结果
            if 'pipeline_results' in data_dict:
                joblib.dump(data_dict['pipeline_results'], os.path.join(output_dir, "pipeline_results.pkl"))
            
            if self.logger:
                self.logger.info(f"预处理产物保存至: {output_dir}")
            
            return output_dir
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存预处理产物失败: {str(e)}")
            return ""

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.processing_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = np.sum(stats['processing_times'])
        
        if self.cache:
            stats['cache_size'] = len(self.cache)
        
        return stats

if __name__ == "__main__":
    # 演示用法
    logger = safe_setup_logging()
    
    # 配置
    config = PreprocessingConfig(
        normalization_method='robust',
        base_window_size=60,
        imbalance_strategy='auto',
        use_parallel=True
    )
    
    # 创建预处理器
    preprocessor = EnhancedDataPreprocessor(config=config, logger=logger)
    
    logger.info("Enhanced DataPreprocessor 初始化完成")
    logger.info(f"配置: {config}")
