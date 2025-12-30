import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

import shap
from openpyxl.drawing.image import Image as ExcelImage

# ========= 全局配置 =========
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows: SimHei / Mac 可换 Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False

# 你本地路径（按你电脑改）
DATA_PATH = r"E:\Python_Project\meituan.xlsx"
OUTPUT_DIR = r"E:\Python_Project\美团相关性"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 'Y-优惠券是否被核销'
BASE_FEATURES = ['券面额', '券使用门槛', '有效期', '活跃率']
RATIO_FEATURE = '券面额-门槛比值'
FEATURES = BASE_FEATURES + [RATIO_FEATURE]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """把指定列尽量转成数值，转不了的变 NaN（后面再填充）"""
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def normalize_target_y(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    兼容两种常见情况：
    - 原始 Y 是 1/2：映射为 0/1（1->0, 2->1）
    - 原始 Y 已经是 0/1：保持不变
    """
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors='coerce')

    unique_vals = set(df[target].dropna().unique().tolist())

    if unique_vals.issubset({0, 1}):
        # 已经是 0/1，不动
        pass
    elif unique_vals.issubset({1, 2}):
        df[target] = df[target].map({1: 0, 2: 1})
    else:
        raise ValueError(
            f"Y 列出现了非预期取值：{sorted(unique_vals)}。"
            f"请确认 Y 是 0/1 或 1/2。"
        )

    df[target] = df[target].astype('Int64')  # 允许 NA 的整数类型
    return df


def build_ratio_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    券面额-门槛比值：
    - 门槛为 0 或缺失：按你说的规则填 1000
    """
    df = df.copy()
    denom = df['券使用门槛']

    ratio = df['券面额'] / denom
    # denom 为 0 或 NaN → ratio 会 inf/NaN，统一设成 1000
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio.fillna(1000)

    df[RATIO_FEATURE] = ratio
    return df


def fill_missing_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """特征缺失用中位数填，避免 dropna 把某一类全删掉"""
    df = df.copy()
    for c in features:
        # 万一整列都是 NaN，用 0 填底
        med = df[c].median()
        if pd.isna(med):
            med = 0
        df[c] = df[c].fillna(med)
    return df


def check_y_distribution(df: pd.DataFrame, target: str) -> None:
    print("\n===== 清洗后 Y 分布 =====")
    print(df[target].value_counts(dropna=False))
    vc = df[target].value_counts(dropna=True)
    if len(vc) < 2:
        print("⚠️ 警告：当前数据只剩一个类别（只有 0 或只有 1），箱线图的 0/1 对比将没有意义。")


# ========= 1) 读数据 =========
df = pd.read_excel(DATA_PATH)
df = ensure_columns(df)

# ========= 2) 数值化 + 处理 Y =========
df = coerce_numeric(df, [TARGET] + BASE_FEATURES)
df = normalize_target_y(df, TARGET)

# ========= 3) 构造比值特征（DIV/0 填 1000） =========
# 这里假设你 Excel 里已经填过 1000 也没关系，我们会重新算一遍并保证规则一致
df = build_ratio_feature(df)

# ========= 4) 只保留需要列，处理缺失（不 drop 特征缺失，避免误杀某类） =========
use_cols = [TARGET] + FEATURES
missing_cols = [c for c in use_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Excel 缺少这些列：{missing_cols}。你现在的列名是：{df.columns.tolist()}")

data = df[use_cols].copy()

# 只 drop 掉 Y 缺失（因为分类必须有标签）
data = data[data[TARGET].notna()].copy()

# 特征缺失填充
data = fill_missing_features(data, FEATURES)

# 强制把 Y 变成 0/1 int
data[TARGET] = data[TARGET].astype(int)

check_y_distribution(data, TARGET)

X = data[FEATURES]
y = data[TARGET]

# ========= 5) 标准化（仅用于模型） =========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========= 6) Excel 写入器 =========
output_excel = os.path.join(OUTPUT_DIR, '分析结果汇总.xlsx')
writer = pd.ExcelWriter(output_excel, engine='openpyxl')

# ========= A) 单因素分析：箱线图（横轴强制 0/1） =========
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(FEATURES):
    ax = axes[i]
    sns.boxplot(
        data=data,
        x=TARGET, y=col,
        order=[0, 1],   # 强制 0/1 顺序
        ax=ax
    )
    ax.set_title(f"{col} 分布 vs 使用情况")
    ax.set_xlabel(TARGET)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['0', '1'])

    # 中位数标注（按类别分别标注）
    med = data.groupby(TARGET)[col].median()
    for cls in [0, 1]:
        if cls in med.index:
            ax.text(cls, med.loc[cls], f"{med.loc[cls]:.2f}",
                    ha='center', va='bottom', color='red')

# 删除多余子图
for j in range(len(FEATURES), len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
boxplot_path = os.path.join(OUTPUT_DIR, '单因素分析.png')
fig.savefig(boxplot_path, dpi=300)
plt.close(fig)

# ========= B) 随机森林特征重要性 =========
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_scaled, y)
rf_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(rf_imp.index, rf_imp.values)
ax.set_title('随机森林特征重要性')
for yi, v in enumerate(rf_imp.values):
    ax.text(v, yi, f"{v:.3f}", va='center', color='blue')
rf_plot_path = os.path.join(OUTPUT_DIR, '随机森林特征重要性.png')
fig.tight_layout()
fig.savefig(rf_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)

rf_imp.to_excel(writer, sheet_name='随机森林特征重要性')

# ========= C) XGBoost 特征重要性 =========
xgb_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9
)
xgb_model.fit(X_scaled, y)

xgb_imp = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(xgb_imp.index, xgb_imp.values)
ax.set_title('XGBoost 特征重要性')
for yi, v in enumerate(xgb_imp.values):
    ax.text(v, yi, f"{v:.3f}", va='center', color='green')
xgb_plot_path = os.path.join(OUTPUT_DIR, 'XGBoost特征重要性.png')
fig.tight_layout()
fig.savefig(xgb_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)

xgb_imp.to_excel(writer, sheet_name='XGBoost特征重要性')

# ========= D) SHAP（全局重要性） =========
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_scaled)

# 兼容旧版 shap 返回 list 的情况
if isinstance(shap_values, list):
    # 二分类一般 [负类, 正类]
    shap_arr = shap_values[1]
else:
    shap_arr = shap_values

shap_mean = np.abs(shap_arr).mean(axis=0)
shap_std = np.abs(shap_arr).std(axis=0)

shap_df = pd.DataFrame({
    '特征': FEATURES,
    'SHAP均值': shap_mean,
    'SHAP标准差': shap_std
}).sort_values('SHAP均值', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(shap_df['特征'], shap_df['SHAP均值'])
ax.set_title('SHAP 全局特征重要性')
for yi, v in enumerate(shap_df['SHAP均值'].values):
    ax.text(v, yi, f"{v:.3f}", va='center', color='purple')
shap_plot_path = os.path.join(OUTPUT_DIR, 'SHAP特征重要性.png')
fig.tight_layout()
fig.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)

shap_df.to_excel(writer, sheet_name='SHAP分析', index=False)

# ========= E) 保存数据（保存清洗后 data，最有用） =========
data.to_excel(writer, sheet_name='用于分析的数据', index=False)

# ========= F) 插图到 Excel =========
wb = writer.book
img_sheet = wb.create_sheet("可视化结果")

def insert_image(sheet, img_path, cell, width=900, height=600):
    img = ExcelImage(img_path)
    img.width = width
    img.height = height
    sheet.add_image(img, cell)

insert_image(img_sheet, boxplot_path, 'A1',  width=1100, height=700)
insert_image(img_sheet, rf_plot_path,  'A40', width=900,  height=450)
insert_image(img_sheet, xgb_plot_path, 'A65', width=900,  height=450)
insert_image(img_sheet, shap_plot_path,'A90', width=900,  height=450)

writer.close()
print(f"\n✅ 分析结果已保存至：{output_excel}")

