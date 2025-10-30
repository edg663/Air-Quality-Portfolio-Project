import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from pathlib import Path

# 读取数据
path = Path(r"D:\air-quality-portfolio\data\cleaned\air_quality_clean.csv")
df = pd.read_csv(path, index_col=0, parse_dates=True)
df = df.dropna(subset=['pm2.5','TEMP','PRES','Iws'])

# 定义模型：PM2.5 = a*T + b*T^2 + c*P + d*W + e
def model_quad(Xdata, a, b, c, d, e):
    T, P, W = Xdata
    return a*T + b*T**2 + c*P + d*W + e

# 提取数据
T = df['TEMP'].values
P = df['PRES'].values
W = df['Iws'].values
Y = df['pm2.5'].values

# 拟合
popt, pcov = optimize.curve_fit(model_quad, (T, P, W), Y, maxfev=10000)
a,b,c,d,e = popt
print("拟合参数：")
print(f"a (T): {a:.4f}, b (T²): {b:.4f}, c (P): {c:.4f}, d (W): {d:.4f}, e: {e:.4f}")

# 预测 & 计算 R²
Y_pred = model_quad((T,P,W), *popt)
resid = Y - Y_pred
r2 = 1 - np.sum(resid**2)/np.sum((Y - np.mean(Y))**2)
print(f"R² = {r2:.4f}")

# 可视化预测效果
plt.figure(figsize=(6,6))
plt.scatter(Y, Y_pred, s=5, alpha=0.4)
plt.plot([0,1000],[0,1000],'r--')
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title(f"PM2.5 Prediction with TEMP² Term\nR²={r2:.3f}")
plt.tight_layout()
plt.show()
