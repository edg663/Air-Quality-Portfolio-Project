import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from modeling_pm25_multivar import model

# 读取数据与参数（直接写上上一步的系数）
a, b, c, d = -3.0765, -3.0323, -0.4556, 3229.8609

path = Path(r"D:\air-quality-portfolio\data\cleaned\air_quality_clean.csv")
df = pd.read_csv(path, index_col=0, parse_dates=True)
df = df.dropna(subset=['pm2.5','TEMP','PRES','Iws'])

# 计算预测值与残差
Y_true = df['pm2.5'].values
Y_pred = model((df['TEMP'].values, df['PRES'].values, df['Iws'].values), a,b,c,d)
residuals = Y_true - Y_pred

# 1️⃣ 残差直方图 + 正态分布拟合
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=60, density=True, alpha=0.6, color='gray')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
plt.plot(x, p, 'r', lw=2)
plt.title("Residual Distribution of Multi-variable PM2.5 Model")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# 2️⃣ QQ图检查正态性
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Residuals Q-Q Plot")
plt.tight_layout()
plt.show()
