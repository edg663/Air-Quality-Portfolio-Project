import pandas as pd
import numpy as np
from pathlib import Path

path = Path(r"D:\air-quality-portfolio\data\cleaned\air_quality_step1.csv")
df = pd.read_csv(path, index_col=0, parse_dates=True)

print("✅ 文件读取成功！")

# 1️⃣ 统计缺失比例
missing_ratio = df.isna().mean() * 100
print("\n每列缺失率 (%):")
print(missing_ratio.sort_values(ascending=False))

# 2️⃣ 查看 pm2.5 缺失比例
pm25_missing = df['pm2.5'].isna().mean() * 100
print(f"\nPM2.5 缺失比例：{pm25_missing:.2f}%")

# 3️⃣ 检查是否有异常值（如负数）
neg_pm25 = (df['pm2.5'] < 0).sum()
print(f"PM2.5 负值数量: {neg_pm25}")

import pandas as pd
import numpy as np
from pathlib import Path

path = Path(r"D:\air-quality-portfolio\data\cleaned\air_quality_step1.csv")
df = pd.read_csv(path, index_col=0, parse_dates=True)

# 1️⃣ 插值（处理短缺失）
df['pm2.5'] = df['pm2.5'].interpolate(method='time', limit=6)

# 2️⃣ 滑动均值（平滑较长缺失）
df['pm2.5'] = df['pm2.5'].fillna(df['pm2.5'].rolling(24, min_periods=1, center=True).mean())

# 3️⃣ 最终检查
print("剩余缺失值数量：", df['pm2.5'].isna().sum())

# 4️⃣ 保存最终清洗结果
out_path = Path(r"D:\air-quality-portfolio\data\cleaned\air_quality_clean.csv")
df.to_csv(out_path)
print("✅ 清洗完成并保存:", out_path)
