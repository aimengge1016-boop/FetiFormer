#!/usr/bin/env python3
import pandas as pd
import os

def convert_date_format():
    """将IRON数据集的日期格式转换为ISO标准格式"""

    # 文件路径
    input_file = 'dataset/IRON/IRON_processed_with_date.csv'
    output_file = 'dataset/IRON/IRON_processed_iso.csv'

    print(f"Reading {input_file}...")
    # 读取数据
    df = pd.read_csv(input_file)

    print("Converting date format...")
    # 将日期列转换为datetime，然后格式化为ISO标准格式
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Saving to {output_file}...")
    # 保存为新文件
    df.to_csv(output_file, index=False)

    print("Date format conversion completed!")
    print(f"Sample dates:")
    print(df['date'].head())
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    convert_date_format()
