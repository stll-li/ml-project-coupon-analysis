import os
import pandas as pd
import numpy as np
import joblib

# 配置路径（请确认路径正确）
PATHS = {
    "model": r"E:\Python_Project\严谨结果\LightGBM_model.pkl",
    "scaler": r"E:\Python_Project\严谨结果\LightGBM_scaler.pkl",
    "data": r"E:\Python_Project\1\yuceji.xlsx"
}

# ========== 1) 加载文件 ==========
df = pd.read_excel(PATHS['data'], header=None)

# 约定：第1列（col0）作为“要写预测值”的列；第2列及以后是特征
x_df = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# ========== 2) 过滤有效行（特征全有值） ==========
valid_mask = ~x_df.isna().any(axis=1)
X = x_df.loc[valid_mask].to_numpy(dtype=float)

if X.shape[0] == 0:
    raise ValueError("没有任何有效行（特征列存在缺失或无法转数值），无法预测。")

# ========== 3) 加载 scaler / model ==========
scaler = joblib.load(PATHS['scaler'])
model = joblib.load(PATHS['model'])

# ========== 4) 维度校验（防止列错位） ==========
expected_n_features = getattr(scaler, "mean_", None)
if expected_n_features is not None:
    expected_n_features = len(scaler.mean_)
    if X.shape[1] != expected_n_features:
        raise ValueError(
            f"特征列数不匹配：预测文件有 {X.shape[1]} 列特征，但 scaler 期望 {expected_n_features} 列。\n"
            f"请检查 yuceji.xlsx 的特征列是否与训练时一致（列数/顺序/是否多了ID列等）。"
        )

# ========== 5) 标准化 & 预测 ==========
X_scaled = scaler.transform(X)

# 优先输出“正类概率”，否则输出类别
if hasattr(model, "predict_proba"):
    preds = model.predict_proba(X_scaled)[:, 1]   # 概率
else:
    preds = model.predict(X_scaled)              # 类别

# ========== 6) 回填预测到首列（col0），保持原始行号对应 ==========
df.loc[np.where(valid_mask)[0], 0] = np.round(preds, 6)

# ========== 7) 另存为新文件（避免覆盖原文件） ==========
base, ext = os.path.splitext(PATHS["data"])
out_path = base + "_pred.xlsx"
df.to_excel(out_path, header=False, index=False, engine="openpyxl")

print(f"✅ 成功填充 {len(preds)} 条预测值到首列")
print(f"✅ 输出文件：{out_path}")

