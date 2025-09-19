import pandas as pd
import random

# 设置随机种子（可选，确保结果可重复）
random.seed(42)

# 读取 CSV 文件（无列名）
df = pd.read_csv('/hpc2hdd/home/lsheng847/1122321/FakeVideoDetection/data/FakeAVCeleb/label.csv', header=None)

# 获取最后三列作为标签
labels = df.iloc[:, -3:]

# 过滤出标签不为 "1,1,1" 的行（即至少一列不为 '1'）
# 假设 "1,1,1" 意味着三列都是 '1'
filtered_df = df[~((labels.iloc[:, 0] == '1') & (labels.iloc[:, 1] == '1') & (labels.iloc[:, 2] == '1'))]

# 检查是否有足够的行（至少 1400 个）
if len(filtered_df) < 1400:
    print(f"错误：只有 {len(filtered_df)} 个符合条件的行，无法删除 1400 个。")
else:
    # 随机选择 1400 个索引进行删除
    indices_to_delete = random.sample(list(filtered_df.index), 1400)
    
    # 从原 DataFrame 中删除这些行
    df_cleaned = df.drop(indices_to_delete)
    
    # 保存新的 CSV 文件（避免覆盖原文件）
    df_cleaned.to_csv('label_cleaned.csv', index=False)
    
    print(f"成功删除 1400 个随机行。原文件有 {len(df)} 行，新文件有 {len(df_cleaned)} 行。")