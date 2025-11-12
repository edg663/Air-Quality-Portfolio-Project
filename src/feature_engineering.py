import pandas as pd
from pathlib import Path


def create_features():
    """
    读取清洗后的数据，创建时间特征、滞后特征和滚动特征，
    并保存到新文件中。
    """
    # 1. 动态路径读取
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_path = BASE_DIR / "data" / "cleaned" / "air_quality_clean.csv"

    try:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
        print(f"成功读取文件: {input_path}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_path}")
        print("请先确保已运行数据清洗脚本。")
        return

    # 2. 提取时间特征
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek

    # 3. ***核心***：创建滞后特征 (Lag Features)
    # 逻辑：现在的 PM2.5 很可能跟 1 小时前、24 小时前的 PM2.5 有关
    df['pm25_lag1'] = df['pm2.5'].shift(1)  # 1小时前
    df['pm25_lag24'] = df['pm2.5'].shift(24)  # 昨天此时

    # 4. 创建滚动平均特征
    # 逻辑：过去 24 小时的平均污染水平
    df['pm25_rolling_24'] = df['pm2.5'].rolling(window=24).mean()

    # 5. 清除因为 shift 和 rolling 产生的空值 (前24行会变成 NaN)
    print(f"创建特征前数据行数: {len(df)}")
    df = df.dropna()
    print(f"去除NaN后数据行数: {len(df)}")

    # 6. 保存为新文件，供模型使用
    output_path = BASE_DIR / "data" / "cleaned" / "air_quality_features.csv"
    df.to_csv(output_path)
    print(f"特征工程完成！数据已保存至: {output_path}")


if __name__ == "__main__":
    create_features()