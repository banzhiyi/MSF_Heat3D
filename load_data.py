import numpy as np
import pandas as pd

# 加载数据
data = np.load('./results/Indian/Indian_test_metrics.npz')

# 创建DataFrame显示主要指标
metrics_df = pd.DataFrame({
    '指标': ['OA (总体准确率)', 'AA (平均准确率)', 'Kappa系数'],
    '数值': [data['OA'], data['AA_mean'], data['Kappa']]
})
print("主要指标:")
print(metrics_df)

# 显示各类别AA值
if 'AA' in data.files:
    aa_df = pd.DataFrame({
        '类别': range(1, len(data['AA']) + 1),
        'AA值': data['AA']
    })
    print("\n各类别AA值:")
    print(aa_df)