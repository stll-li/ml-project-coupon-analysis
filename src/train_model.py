import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

# ==================== 全局配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = r"E:\Python_Project\meituan.xlsx"
OUTPUT_DIR = r"E:\Python_Project\严谨结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42  # 统一随机种子（核心）


# ==================== 复现性设置 ====================
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_global_seed(SEED)


# ==================== 数据准备 ====================
def load_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()

    # 假设：第1列是Y，其余是特征（保持你原逻辑不变）
    y_raw = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    X_df = df.iloc[:, 1:].copy()

    # 特征强制数值化
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors='coerce')

    # 只丢掉 Y 缺失（因为没标签没法学）
    mask = y_raw.notna()
    y_raw = y_raw[mask]
    X_df = X_df.loc[mask].copy()

    # 处理 Y：兼容 0/1 或 1/2
    uniq = set(y_raw.unique().tolist())
    if uniq.issubset({0, 1}):
        y = y_raw.astype(int).values
    elif uniq.issubset({1, 2}):
        y = y_raw.map({1: 0, 2: 1}).astype(int).values
    else:
        raise ValueError(f"Y 列取值异常：{sorted(uniq)}，请确认是 0/1 或 1/2")

    # 特征缺失：用中位数填（避免 dropna 误杀某一类）
    for c in X_df.columns:
        med = X_df[c].median()
        if pd.isna(med):
            med = 0
        X_df[c] = X_df[c].fillna(med)

    X = X_df.values.astype(float)

    # sanity check
    vc = pd.Series(y).value_counts()
    print("\n===== Y 分布 =====")
    print(vc)
    if len(vc) < 2:
        raise ValueError("当前数据只有一个类别（只剩0或只剩1），无法训练分类模型。")

    return X, y, X_df.columns.tolist()


# ==================== 模型配置 ====================
def get_models_config(seed: int = 42):
    # 模型顺序严格保持不变（你要求的位置不变）
    models = {
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=seed, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(
            eval_metric='logloss',
            random_state=seed,
            n_estimators=200,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            random_state=seed,
            n_estimators=200,
            n_jobs=-1
        ),
        'CatBoost': cb.CatBoostClassifier(
            verbose=0,
            random_seed=seed,
            # 为了更稳定，可选：thread_count=1（更慢但更稳）
            # thread_count=1
        ),
        'Gradient Boosting': GradientBoostingClassifier(random_state=seed),
        'AdaBoost': AdaBoostClassifier(random_state=seed)
    }

    params = {
        'KNN': {
            'n_neighbors': list(range(3, 31, 2)),
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'LightGBM': {
            'num_leaves': [31, 63, 127],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300]
        },
        'CatBoost': {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.03, 0.1]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [2, 3, 4]
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.5, 1.0, 1.5]
        }
    }
    return models, params


# ==================== 训练评估流程 ====================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    }


def generate_visualizations(model, name, X_test, y_test):
    # 混淆矩阵
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    plt.title(f'{name} 混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_confusion_matrix.png'), dpi=300)
    plt.close()

    # ROC曲线
    if hasattr(model, 'predict_proba'):
        plt.figure(figsize=(8, 6))
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f'{name} ROC曲线')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_roc_curve.png'), dpi=300)
        plt.close()


def train_evaluate_models(models, params, X_train, X_test, y_train, y_test, seed: int = 42):
    results = []

    # 固定 CV 划分（这一步会显著提高复现性）
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    for name, model in models.items():
        print(f"\n>> 正在处理 {name} <<")

        # 标准化处理（KNN 等需要；树模型不需要但不影响）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 保存标准化器
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, f'{name}_scaler.pkl'))

        # 参数搜索（固定 random_state）
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params[name],
            n_iter=10,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=seed
        )
        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_

        # 评估
        metrics = evaluate_model(best_model, X_test_scaled, y_test)
        results.append({'Model': name, **metrics})

        # 可视化
        generate_visualizations(best_model, name, X_test_scaled, y_test)

        # 保存模型
        joblib.dump(best_model, os.path.join(OUTPUT_DIR, f'{name}_model.pkl'))

    return pd.DataFrame(results)


# ==================== 主流程 ====================
if __name__ == "__main__":
    X, y, feature_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y  # stratify 让类别比例稳定
    )

    models, params = get_models_config(SEED)
    results_df = train_evaluate_models(models, params, X_train, X_test, y_train, y_test, seed=SEED)

    # 保存结果
    results_path = os.path.join(OUTPUT_DIR, '模型结果.xlsx')
    results_df.to_excel(results_path, index=False)
    print("\n✅ 所有模型训练完成，结果已保存至:", results_path)
