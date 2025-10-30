import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from pathlib import Path

# 读取数据
path = Path(r"D:\air-quality-portfolio\data\cleaned\air_quality_clean.csv")
df = pd.read_csv(path, index_col=0, parse_dates=True)

# 去掉缺失行
df = df.dropna(subset=['pm2.5', 'TEMP', 'PRES', 'Iws'])

# 自变量与因变量
X_temp = df['TEMP'].values
X_pres = df['PRES'].values
X_wind = df['Iws'].values
Y_pm25 = df['pm2.5'].values

# 1️⃣ 定义模型函数：PM2.5 = a*T + b*P + c*W + d
def model(Xdata, a, b, c, d):
    T, P, W = Xdata
    return a*T + b*P + c*W + d

# 2️⃣ 使用 curve_fit 拟合
popt, pcov = optimize.curve_fit(model, (X_temp, X_pres, X_wind), Y_pm25)
a, b, c, d = popt

print(f"拟合结果:")
print(f"a (温度系数): {a:.4f}")
print(f"b (气压系数): {b:.4f}")
print(f"c (风速系数): {c:.4f}")
print(f"d (常数项): {d:.4f}")

# 3️⃣ 计算预测值与残差
Y_pred = model((X_temp, X_pres, X_wind), *popt)
residuals = Y_pm25 - Y_pred

# 4️⃣ 拟合优度 R²
ss_res = np.sum(residuals**2)
ss_tot = np.sum((Y_pm25 - np.mean(Y_pm25))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"R² = {r2:.4f}")

# 5️⃣ 可视化：预测 vs 实际
plt.figure(figsize=(6,6))
plt.scatter(Y_pm25, Y_pred, s=5, alpha=0.4)
plt.plot([0,1000], [0,1000], 'r--')
plt.xlabel("Actual PM2.5 (μg/m³)")
plt.ylabel("Predicted PM2.5 (μg/m³)")
plt.title(f"PM2.5 Prediction (Multi-variable Model)\nR² = {r2:.3f}")
plt.tight_layout()
plt.show()
