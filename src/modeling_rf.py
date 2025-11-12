import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_and_validate_rf(test_size=0.2):
    """
    使用特征工程后的数据训练随机森林模型，并进行全面模型验证：
    1) 时间序列交叉验证
    2) 留出测试集评估
    3) 特征重要性可视化
    4) 预测 vs 实际图
    5) 残差分析
    """
    # 1. 读取特征数据
    BASE_DIR = Path(__file__).resolve().parent.parent
    path = BASE_DIR / "data" / "cleaned" / "air_quality_features.csv"

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"成功读取文件: {path}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {path}")
        print("请先运行 src/feature_engineering.py 来生成该文件。")
        return

    # 2. 选择特征列
    feature_cols = [
        'TEMP', 'PRES', 'Iws', 'day_of_week',          # 原始气象 + weekday
        'pm25_lag1', 'pm25_lag24', 'pm25_rolling_24', # PM2.5特征
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',  # 周期性时间特征
        'Iws_lag6', 'TEMP_lag3', 'PRES_lag6'          # 气象滞后特征
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df['pm2.5']

    # ----------------------------
    # 3. 时间序列交叉验证
    # ----------------------------
    print("\n=== 时间序列交叉验证 ===")
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=tscv, scoring='r2', n_jobs=-1)
    print(f"交叉验证 R² 均值: {cv_scores.mean():.4f}")
    print(f"交叉验证 R² 标准差: {cv_scores.std():.4f}")

    # ----------------------------
    # 4. 留出测试集评估
    # ----------------------------
    print("\n=== 留出测试集评估 ===")
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)

    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"测试集 R²: {r2_test:.4f}")
    print(f"测试集 MSE: {mse_test:.2f}")
    print(f"测试集 MAE: {mae_test:.2f}")

    # ----------------------------
    # 5. 特征重要性
    # ----------------------------
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(8,6))
    importances.plot(kind='barh', color='teal')
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 6. 预测 vs 实际图
    # ----------------------------
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test.values, label='Actual PM2.5', color='black', alpha=0.7)
    plt.plot(y_test.index, y_pred_test, label='Predicted PM2.5', color='red', linestyle='--')
    plt.title("PM2.5 Prediction: Actual vs Predicted (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 7. 残差分析
    # ----------------------------
    residuals = y_test - y_pred_test
    plt.figure(figsize=(12,5))
    plt.plot(y_test.index, residuals, label='Residuals', color='purple')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Residuals (Actual - Predicted)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_and_validate_rf()
