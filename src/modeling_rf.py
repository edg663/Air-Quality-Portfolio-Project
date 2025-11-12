import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def train_random_forest():
    """
    使用特征工程后的数据训练随机森林模型。
    """
    # 1. 动态路径读取“特征数据”
    BASE_DIR = Path(__file__).resolve().parent.parent
    path = BASE_DIR / "data" / "cleaned" / "air_quality_features.csv"

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {path}")
        print("请先运行 src/feature_engineering.py 来生成该文件。")
        return

    # 2. 准备输入(X)和输出(y)
    # 注意：要把刚刚造出来的特征全放进去
    # (如果你的数据有DEWP, cbwd等，也加进来)
    feature_cols = ['TEMP', 'PRES', 'Iws', 'hour', 'month',
                    'day_of_week', 'pm25_lag1', 'pm25_lag24', 'pm25_rolling_24']

    X = df[feature_cols]
    y = df['pm2.5']

    # 3. 切分训练集和测试集 (不打乱顺序，因为是时间序列)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 4. 训练随机森林模型
    print("正在训练随机森林模型 (可能需要几秒钟)...")
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # 5. 预测与评估
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("-" * 30)
    print(f"模型 R² (拟合优度): {r2:.4f}  <-- (期待看到 0.85 以上!)")
    print(f"均方误差 (MSE): {mse:.2f}")
    print("-" * 30)

    # 6. 可视化：特征重要性
    print("正在绘制特征重要性...")
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    importances.plot(kind='barh', color='teal')
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.show()

    # 7. 可视化：预测 vs 真实 (取最后 200 个小时看细节)
    print("正在绘制预测对比图...")
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index[-200:], y_test.values[-200:], label='Actual PM2.5', color='black', alpha=0.7)
    plt.plot(y_test.index[-200:], y_pred[-200:], label='Predicted PM2.5', color='red', linestyle='--')
    plt.title("PM2.5 Prediction: Actual vs Predicted (Last 200 Hours)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_random_forest()