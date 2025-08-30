import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import shap
import logging
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import TimeSeriesSplit

def safe_setup_logging():
    """配置日志系统"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 文件输出
    file_handler = logging.FileHandler('model_trainer.log')
    file_handler.setFormatter(formatter)
    
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger

def reshape_for_training(X):
    """将3D序列数据转换为2D特征矩阵"""
    return X.reshape(X.shape[0], -1)

def apply_smote_time_series(X_train_3d, y_train, k_neighbors=None):
    """保持时间序列结构的SMOTE应用"""
    original_shape = X_train_3d.shape
    X_2d = X_train_3d.reshape(original_shape[0], -1)
    
    # 动态设置k_neighbors
    if k_neighbors is None:
        min_count = min(np.bincount(y_train))
        k_neighbors = min(5, max(1, min_count - 1))
    
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_res_2d, y_res = smote.fit_resample(X_2d, y_train)
    
    # 恢复原始维度
    X_res = X_res_2d.reshape(-1, original_shape[1], original_shape[2])
    return X_res, y_res

def stable_feature_selection(model, X_3d, y, n_splits=3, top_k=8):
    """使用时序交叉验证的特征选择 - 处理3D时间序列数据"""
    # 将3D数据转换为2D: [samples, timesteps * features]
    n_samples, n_timesteps, n_features = X_3d.shape
    X_2d = X_3d.reshape(n_samples, -1)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    feature_scores = np.zeros(n_features)  # 原始特征数量
    
    for train_idx, _ in tscv.split(X_2d):
        X_train_2d = X_2d[train_idx]
        y_train = y[train_idx]
        
        # 训练模型
        model.fit(X_train_2d, y_train)
        
        # 获取特征重要性: [timesteps * features]
        importances = model.feature_importances_
        
        # 重塑为 [timesteps, features] 并聚合
        importances_reshaped = importances.reshape(n_timesteps, n_features)
        aggregated = np.mean(importances_reshaped, axis=0)  # 跨时间步平均
        
        feature_scores += aggregated
    
    # 获取最重要的原始特征索引
    top_indices = np.argsort(feature_scores)[-top_k:]
    return top_indices

def neutral_f1_eval(y_pred, data):
    """自定义中性状态F1评估函数"""
    y_true = data.get_label()
    num_classes = len(np.unique(y_true))
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    neutral_mask = (y_true == 1)
    if np.sum(neutral_mask) == 0:
        return 'neutral_f1', 0.0, True
    
    neutral_true = (y_true == 1).astype(int)
    neutral_pred = (y_pred_labels == 1).astype(int)
    
    score = f1_score(neutral_true, neutral_pred, average='binary')
    return 'neutral_f1', score, True

def train_model(X_train, y_train, X_valid, y_valid, feature_names):
    """训练并评估模型"""
    # 计算类别权重
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    # 创建Dataset对象
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
  
    # 配置参数
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'learning_rate': 0.01,
        'max_depth': 4,
        'num_leaves': 15,
        'min_child_samples': 50,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'importance_type': 'gain',
        'verbose': -1
    }
  
    # 回调函数
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=50)
    ]
  
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[valid_data],
        feval=neutral_f1_eval,
        callbacks=callbacks
    )
  
    return model

def evaluate_model(model, X_valid, y_valid, feature_names, window_size=60):
    """模型评估与解释 - 兼容集成模型"""
    y_pred_proba = model.predict(X_valid)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 性能报告
    logger.info(f"混淆矩阵:\n{confusion_matrix(y_valid, y_pred)}")
    logger.info(f"分类报告:\n{classification_report(y_valid, y_pred, target_names=['down', 'neutral', 'up'])}")
    
    # SHAP解释（使用第一个模型作为代表）
    if hasattr(model, 'models'):
        explainer = shap.TreeExplainer(model.models[0])
    else:
        explainer = shap.TreeExplainer(model)
    
    sample_idx = np.random.choice(len(X_valid), min(100, len(X_valid)), replace=False)
    shap_values = explainer.shap_values(X_valid[sample_idx])
    
    # 生成展开的特征名
    expanded_names = [f'{name}_{t}' for name in feature_names for t in range(window_size)]
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_valid[sample_idx], feature_names=expanded_names, show=False)
    plt.savefig("shap_summary.png")
    plt.close()

class ModelEnsemble:
    def __init__(self, models):
        self.models = models
       
    def predict(self, X):
        preds = np.zeros((X.shape[0], 3))
        for model in self.models:
            preds += model.predict(X)
        return preds / len(self.models)

if __name__ == "__main__":
    logger = safe_setup_logging()
    logger.info("===== 模型训练开始 =====")
    
    try:
        # 加载预处理数据
        X_train = np.load("preprocessed_data/X_train.npy")
        y_train = np.load("preprocessed_data/y_train.npy")
        X_valid = np.load("preprocessed_data/X_test.npy")
        y_valid = np.load("preprocessed_data/y_test.npy")
        
        # 加载元数据
        metadata = joblib.load("preprocessed_data/metadata.pkl")
        feature_cols = metadata['feature_cols']
        window_size = metadata['window_size']
        
        logger.info(f"加载数据: 训练集{X_train.shape}, 验证集{X_valid.shape}")
        logger.info(f"特征维度: {len(feature_cols)} 个特征, 窗口大小: {window_size}")

        # 数据标准化
        scaler = StandardScaler()
        X_train_2d = reshape_for_training(X_train)
        X_valid_2d = reshape_for_training(X_valid)
        
        scaler.fit(X_train_2d)
        X_train_scaled = scaler.transform(X_train_2d)
        X_valid_scaled = scaler.transform(X_valid_2d)
        
        # 将标准化数据恢复为3D格式
        X_train_scaled_3d = X_train_scaled.reshape(X_train.shape)
        X_valid_scaled_3d = X_valid_scaled.reshape(X_valid.shape)
        
        # 初始化模型用于特征选择
        base_model = lgb.LGBMClassifier(n_estimators=300, verbose=-1)
        
        # 特征选择
        top_feature_indices = stable_feature_selection(
            base_model, 
            X_train_scaled_3d, 
            y_train,
            n_splits=3,
            top_k=10
        )
        
        selected_features = [feature_cols[i] for i in top_feature_indices]
        logger.info(f"选择的特征: {selected_features}")
        
        # 检查时间特征是否被选中
        time_features = [f for f in selected_features if 'hour' in f or 'day' in f or 'time' in f]
        logger.info(f"选中时间特征数量: {len(time_features)}")
        logger.info(f"选中时间特征: {time_features}")
        
        # 提取选定特征（使用标准化数据）
        X_train_selected = X_train_scaled_3d[:, :, top_feature_indices]
        X_valid_selected = X_valid_scaled_3d[:, :, top_feature_indices]
        
        # 实现交叉验证（避免数据泄露）
        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = []
        fold_models = [] # 保存所有交叉验证模型
        for train_idx, val_idx in tscv.split(X_train_selected):
            # 获取当前折的训练/验证数据
            X_fold_train = X_train_selected[train_idx]
            y_fold_train = y_train[train_idx]
            
            # 注意：验证集需要保持原始索引
            X_fold_val = X_train_selected[val_idx] # 保持3D
            y_fold_val = y_train[val_idx]
            
            # 将训练集和验证集都转换为2D
            X_fold_train_2d = reshape_for_training(X_fold_train)
            X_fold_val_2d = reshape_for_training(X_fold_val)
            
            # 不再应用SMOTE，使用原始数据训练
            fold_model = train_model(
                X_fold_train_2d, y_fold_train,  # 使用原始数据
                X_fold_val_2d, y_fold_val, selected_features
            )
            
            # 预测
            y_pred = fold_model.predict(X_fold_val_2d)
            y_pred = np.argmax(y_pred, axis=1)
            
            fold_score = f1_score(y_fold_val, y_pred, average='weighted')
            fold_scores.append(fold_score)
            fold_models.append(fold_model)  # 保存模型
            logger.info(f"Fold {len(fold_scores)} F1: {fold_score:.4f}")

        logger.info(f"平均交叉验证F1: {np.mean(fold_scores):.4f}")
        # 创建模型集成
        final_model = ModelEnsemble(fold_models)
        # 检查并记录原始数据分布
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"最终模型训练数据类别分布: {dict(zip(unique, counts))}")
        # 准备验证集数据
        X_valid_final = reshape_for_training(X_valid_selected)
        # 模型评估
        evaluate_model(final_model, X_valid_final, y_valid, selected_features, window_size=window_size)
        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"trading_model_{timestamp}.pkl"
        pipeline = {
            'model': final_model,
            'scaler': scaler,
            'feature_indices': top_feature_indices,
            'feature_names': selected_features,
            'window_size': window_size
        }
        joblib.dump(pipeline, model_path)
        logger.info(f"模型保存至: {model_path}")
        
    except Exception as e:
        logger.error(f"训练异常: {str(e)}")
        logger.exception("错误详情:")
    
    logger.info("===== 模型训练完成 ======")