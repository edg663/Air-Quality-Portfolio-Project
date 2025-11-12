import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 自动获取当前脚本(src)的父目录(项目根目录)
BASE_DIR = Path(__file__).resolve().parent.parent
# 构造动态路径
path = BASE_DIR / "data" / "cleaned" / "air_quality_clean.csv"

df = pd.read_csv(path, index_col=0, parse_dates=True)

# 1️⃣ 绘制逐小时 PM2.5 时间序列
plt.figure(figsize=(12,4))
plt.plot(df.index, df['pm2.5'], lw=0.4, color='steelblue')
plt.title("Hourly PM2.5 Concentration (2010–2014, Beijing)")
plt.xlabel("Time")
plt.ylabel("PM2.5 (μg/m³)")
plt.tight_layout()
plt.show()

# 2️⃣ 计算每月平均值
monthly_mean = df['pm2.5'].resample('ME').mean()

plt.figure(figsize=(10,4))
plt.plot(monthly_mean.index, monthly_mean.values, lw=2, color='orange')
plt.title("Monthly Average PM2.5 (2010–2014)")
plt.xlabel("Month")
plt.ylabel("PM2.5 (μg/m³)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 🔹 计算7天滚动平均（168小时）
df['pm25_rolling'] = df['pm2.5'].rolling(window=168, min_periods=1).mean()

plt.figure(figsize=(12,4))
plt.plot(df.index, df['pm2.5'], lw=0.2, color='lightgray', label='Hourly PM2.5')
plt.plot(df.index, df['pm25_rolling'], lw=2, color='red', label='7-day Rolling Mean')
plt.title("PM2.5 Hourly Trend with 7-Day Rolling Average (2010–2014, Beijing)")
plt.xlabel("Time")
plt.ylabel("PM2.5 (μg/m³)")
plt.legend()
plt.tight_layout()
plt.show()

# 🔹 计算每年平均浓度
yearly_mean = df['pm2.5'].resample('YE').mean()

plt.figure(figsize=(7,4))
plt.bar(yearly_mean.index.year, yearly_mean.values, color='skyblue')
plt.title("Yearly Average PM2.5 (Beijing 2010–2014)")
plt.xlabel("Year")
plt.ylabel("Average PM2.5 (μg/m³)")
plt.tight_layout()
plt.show()
