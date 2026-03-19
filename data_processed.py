import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """
    读取IRON.csv数据并进行初步处理
    """
    print("正在读取数据...")
    df = pd.read_csv(filepath)

    # 检查数据基本信息
    print(f"数据形状: {df.shape}")
    print("数据列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")

    print("\n数据类型:")
    print(df.dtypes)

    print("\n数据前5行:")
    print(df.head())

    return df

def handle_data_types(df):
    """
    处理数据类型转换
    """
    print("\n正在处理数据类型...")

    # 日期时间转换 - 使用日/月/年格式
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')

    # 数值列转换
    numeric_cols = [
        'Usage_kWh',
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor',
        'NSM'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 分类变量编码
    categorical_cols = ['WeekStatus', 'Day_of_week', 'Load_Type']
    encoders = {}

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col + '_encoded'] = encoder.fit_transform(df[col])
        encoders[col] = encoder
        print(f"{col} 编码映射: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    print("数据类型处理完成")
    return df, encoders

def z_score_normalization(df, numeric_cols):
    """
    实现Z-score标准化
    """
    print("\n正在进行Z-score标准化...")

    scaler = StandardScaler()

    # 标准化数值特征
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("标准化完成")
    print("标准化后的统计信息:")
    print(df_normalized[numeric_cols].describe())

    return df_normalized, scaler

def mutual_info_feature_selection(X, y, feature_names):
    """
    使用互信息进行特征选择
    """
    print("\n正在计算互信息(MI)...")

    # 计算互信息
    mi_scores = mutual_info_regression(X, y, random_state=42)

    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    print("互信息得分:")
    for idx, row in feature_importance.iterrows():
        print(".4f")

    return feature_importance

def select_features_based_on_paper(feature_importance):
    """
    根据论文描述选择特征
    论文中保留的7个关键特征：
    1. 滞后电流功率
    2. 超前电流功率
    3. CO₂浓度
    4. 滞后电流功率因数
    5. 超前电流功率因数
    6. 设备当日运行秒数
    7. 负载类型
    """
    print("\n根据论文进行特征选择...")

    # 论文中保留的特征（对应列名）
    selected_features = [
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor',
        'NSM',
        'Load_Type_encoded'  # 负载类型（编码后）
    ]

    # 论文中剔除的特征（MI值很低）
    excluded_features = [
        'Day_of_week_encoded',  # 星期几
        'WeekStatus_encoded'    # 周状态
    ]

    print("保留的特征:")
    for feature in selected_features:
        mi_score = feature_importance[feature_importance['feature'] == feature]['mi_score'].values
        if len(mi_score) > 0:
            print(".4f")

    print("\n剔除的特征:")
    for feature in excluded_features:
        mi_score = feature_importance[feature_importance['feature'] == feature]['mi_score'].values
        if len(mi_score) > 0:
            print(".4f")

    return selected_features

def save_processed_data(df, selected_features, output_path):
    """
    保存处理后的数据
    """
    print(f"\n正在保存处理后的数据到 {output_path}...")

    # 选择特征和目标变量
    columns_to_save = selected_features + ['Usage_kWh']  # 包含预测目标

    df_processed = df[columns_to_save].copy()

    # 保存到CSV
    df_processed.to_csv(output_path, index=False)

    print(f"数据保存完成，形状: {df_processed.shape}")
    print("最终特征:")
    for i, col in enumerate(columns_to_save[:-1], 1):  # 排除目标变量
        print(f"{i}. {col}")
    print(f"目标变量: {columns_to_save[-1]}")

    return df_processed

def plot_feature_importance(feature_importance):
    """
    可视化特征重要性
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='mi_score', y='feature', data=feature_importance)
    plt.title('Feature Importance (Mutual Information)')
    plt.xlabel('MI Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    主函数：执行完整的数据预处理流程
    """
    # 尝试多个可能的路径
    possible_input_paths = [
        'dataset/IRON/IRON.csv',  # AutoDL环境中的路径
        'IRON.csv',
        '../dataset/IRON.csv',
        'iTransformer-main/datasets/IRON.csv',
        'dataset/IRON_processed.csv'  # 避免使用已处理的文件
    ]

    input_file = None
    for path in possible_input_paths:
        try:
            pd.read_csv(path)
            input_file = path
            print(f"找到输入文件: {path}")
            break
        except:
            continue

    if input_file is None:
        print("错误: 找不到IRON.csv文件")
        return None, None

    output_file = 'dataset/IRON_processed.csv'

    # 1. 读取数据
    df = load_and_preprocess_data(input_file)

    # 2. 处理数据类型
    df, encoders = handle_data_types(df)

    # 3. Z-score标准化
    numeric_cols = [
        'Usage_kWh',
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2(tCO2)',
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor',
        'NSM'
    ]

    df_normalized, scaler = z_score_normalization(df, numeric_cols)

    # 4. 互信息特征选择
    # 准备特征和目标变量
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

    X = df_normalized[feature_cols]
    y = df_normalized['Usage_kWh']

    feature_importance = mutual_info_feature_selection(X, y, feature_cols)

    # 5. 根据论文选择特征
    selected_features = select_features_based_on_paper(feature_importance)

    # 6. 保存处理后的数据
    df_processed = save_processed_data(df_normalized, selected_features, output_file)

    # 7. 可视化特征重要性（可选）
    print("\n生成特征重要性图表...")
    plot_feature_importance(feature_importance)

    print("\n数据预处理完成！")
    print(f"原始数据形状: {df.shape}")
    print(f"处理后数据形状: {df_processed.shape}")

    return df_processed, feature_importance

if __name__ == "__main__":
    processed_data, feature_importance = main()
