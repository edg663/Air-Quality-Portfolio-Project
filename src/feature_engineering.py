import pandas as pd
import numpy as np
from pathlib import Path

def create_features():
    """
    读取清洗后的数据，创建完整的时间特征、滞后特征、滚动特征和气象滞后特征，
    并保存到新文件中。
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_path = BASE_DIR / "data" / "cleaned" / "air_quality_clean.csv"

    try:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        print(f"成功读取文件: {input_path}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_path}")
        return

    # =========================
    # 1. 时间特征
    # =========================
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday

    # 周期性编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # 删除原始列可选
    df = df.drop(columns=['hour', 'month', 'day_of_week'])

    # =========================
    # 2. PM2.5 滞后特征
    # =========================
    df['pm25_lag1'] = df['pm2.5'].shift(1)
    df['pm25_lag24'] = df['pm2.5'].shift(24)

    # =========================
    # 3. 滚动平均特征
    # =========================
    df['pm25_rolling_24'] = df['pm2.5'].rolling(window=24).mean()

    # =========================
    # 4. 气象滞后特征
    # =========================
    if 'Iws' in df.columns:
        df['Iws_lag6'] = df['Iws'].shift(6)  # 6小时前风速
    if 'TEMP' in df.columns:
        df['TEMP_lag3'] = df['TEMP'].shift(3)  # 3小时前温度
    if 'PRES' in df.columns:
        df['PRES_lag6'] = df['PRES'].shift(6)  # 6小时前气压

    # =========================
    # 5. 去除因 shift/rolling 产生的 NaN
    # =========================
    print(f"创建特征前数据行数: {len(df)}")
    df = df.dropna()
    print(f"去除NaN后数据行数: {len(df)}")

    # =========================
    # 6. 保存为新文件
    # =========================
    output_path = BASE_DIR / "data" / "cleaned" / "air_quality_features.csv"
    df.to_csv(output_path)
    print(f"特征工程完成！数据已保存至: {output_path}")


if __name__ == "__main__":
    create_features()
