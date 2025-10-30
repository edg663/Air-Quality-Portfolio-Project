import pandas as pd
import numpy as np
from pathlib import Path

path = Path(r"D:\air-quality-portfolio\data\cleaned\air_quality_clean.csv")
df = pd.read_csv(path, index_col=0, parse_dates=True)

# 1️⃣ NumPy 基础统计
pm25 = df['pm2.5'].dropna().values
mean = np.mean(pm25)
std = np.std(pm25)
median = np.median(pm25)

print(f"PM2.5 平均值: {mean:.2f}")
print(f"PM2.5 标准差: {std:.2f}")
print(f"PM2.5 中位数: {median:.2f}")

# 2️⃣ Pandas 统计对比
desc = df['pm2.5'].describe()
print("\nPandas describe() 结果:")
print(desc)
