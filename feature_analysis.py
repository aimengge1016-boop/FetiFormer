import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# =====================================================
# 数据加载与预处理
# =====================================================
def load_and_prepare(path):
    df = pd.read_csv(path)

    # parse date (day/month/year)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    numeric_cols = [
        'Usage_kWh',
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor',
        'NSM'
    ]

    categorical_cols = ['WeekStatus', 'Day_of_week', 'Load_Type']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, numeric_cols, encoders


# =====================================================
# Mutual Information
# =====================================================
def export_mutual_info(df, feature_cols, target_col, out_csv):
    X = df[feature_cols]
    y = df[target_col]

    mi = mutual_info_regression(X, y, random_state=42)
    fi = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi
    }).sort_values('mi_score', ascending=False).reset_index(drop=True)

    fi.to_csv(out_csv, index=False)
    print(f'互信息得分已保存: {out_csv}')
    return fi


# =====================================================
# Permutation Importance
# =====================================================
def compute_permutation_importance(df, feature_cols, target_col, out_csv, random_state=42):
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    r = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_mean': r.importances_mean,
        'importance_std': r.importances_std
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

    perm_df.to_csv(out_csv, index=False)
    print(f'Permutation importance 已保存: {out_csv}')

    return perm_df, model, X_test


# =====================================================
# SHAP（论文可用 & 不慢版本）
# =====================================================
def compute_shap(
    model,
    X,
    feature_cols,
    out_csv,
    out_png=None,
    max_samples=300,
    random_state=42
):
    try:
        import shap
    except Exception as e:
        print('SHAP 未安装或导入失败:', e)
        return None

    # ---------- 子采样（关键，防止卡死） ----------
    rng = np.random.RandomState(random_state)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_shap = X[idx]
    else:
        X_shap = X

    # ---------- TreeExplainer（树模型最快） ----------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # ---------- 全局 SHAP 重要性 ----------
    abs_mean = np.mean(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'shap_abs_mean': abs_mean
    }).sort_values('shap_abs_mean', ascending=False).reset_index(drop=True)

    shap_df.to_csv(out_csv, index=False)
    print(f'SHAP 重要性已保存（样本数={len(X_shap)}）: {out_csv}')

    # ---------- SHAP Summary Plot ----------
    if out_png:
        try:
            import matplotlib.pyplot as plt
            shap.summary_plot(
                shap_values,
                X_shap,
                feature_names=feature_cols,
                show=False
            )
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'SHAP 图已保存: {out_png}')
        except Exception as e:
            print('保存 SHAP 图失败:', e)

    return shap_df


# =====================================================
# Main
# =====================================================
def main():
    base = os.path.dirname(__file__)

    data_path = os.path.join(base, 'dataset', 'IRON.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(base, 'dataset', 'IRON', 'IRON.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'找不到数据文件: {data_path}')

    df, numeric_cols, encoders = load_and_prepare(data_path)

    feature_cols = [
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor',
        'NSM',
        'WeekStatus_encoded',
        'Day_of_week_encoded',
        'Load_Type_encoded'
    ]
    target_col = 'Usage_kWh'

    # Mutual Information
    out_mi = os.path.join(base, 'feature_importance_scores.csv')
    export_mutual_info(df, feature_cols, target_col, out_mi)

    # Permutation Importance
    out_perm = os.path.join(base, 'permutation_importance.csv')
    _, model, X_test = compute_permutation_importance(
        df, feature_cols, target_col, out_perm
    )

    # SHAP
    out_shap = os.path.join(base, 'shap_importance.csv')
    out_shap_png = os.path.join(base, 'shap_summary.png')
    shap_df = compute_shap(
        model,
        X_test,
        feature_cols,
        out_shap,
        out_shap_png,
        max_samples=300
    )

    print('\n完成：已生成文件：')
    print(' -', out_mi)
    print(' -', out_perm)
    if shap_df is not None:
        print(' -', out_shap)
        print(' -', out_shap_png)
    else:
        print(' - SHAP 未生成')


if __name__ == '__main__':
    main()
