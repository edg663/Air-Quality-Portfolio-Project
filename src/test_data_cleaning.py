import pandas as pd
from pathlib import Path

# 读取原始数据
path = Path(r"D:\air-quality-portfolio\data\raw\PRSA_data_2010.1.1-2014.12.31.csv")
df = pd.read_csv(path)

# 🕒 合并时间列
df['datetime'] = pd.to_datetime(
    dict(year=df['year'], month=df['month'], day=df['day'], hour=df['hour'])
)

# 按时间排序并设置为索引
df = df.sort_values('datetime').set_index('datetime')

# 查看结果
print("索引类型:", type(df.index))
print("前5行:")
print(df.head())

# 保存清洗后的版本（下一步还会继续改进）
out_path = Path(r"D:\air-quality-portfolio\data\cleaned\air_quality_step1.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path)
print("✅ 已保存:", out_path)
