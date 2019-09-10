import pandas as pd
from sklearn.metrics import mean_squared_error as mse

# 导入数据
fn = 'data/hc_row3.csv'
df = pd.read_csv(fn)

# alpha和window的所有可能值，共9*9=81种组合
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
windows = [5, 15, 20, 23, 25, 27, 28, 29, 30]

# 初始化结果表
df_mse = pd.DataFrame(columns=['alpha', 'window', 'mse_v1', 'mse_v2'])

# alpha和window添加进结果表
for alpha in alphas:
    for window in windows:
        df_mse.loc[alphas.index(alpha) * len(alphas) + windows.index(window), 'alpha'] = alpha
        df_mse.loc[alphas.index(alpha) * len(alphas) + windows.index(window), 'window'] = window

# 计算mse
for alpha in alphas:
    for window in windows:
        _df = df.copy()
        _df['ewma_3_2'] = _df['3_2_rate'].ewm(alpha=alpha, min_periods=window).mean()
        _df['forecast_3_1'] = (4 / 5) * _df['ewma_3_2']
        df_mse.loc[alphas.index(alpha) * len(alphas) + windows.index(window), 'mse_v1'] = mse(
            _df['3_2_rate'][window - 1:], _df['ewma_3_2'][window - 1:])
        df_mse.loc[alphas.index(alpha) * len(alphas) + windows.index(window), 'mse_v2'] = mse(
            _df['3_1_rate'][window - 1:], _df['forecast_3_1'][window - 1:])

df_mse['beta'] = df_mse['mse_v2'] / df_mse['mse_v1']
df_mse['sum'] = df_mse['mse_v2'] + df_mse['mse_v1']

# 选择两列mse相加的比较
print(df_mse.loc[df_mse['sum'] == df_mse['sum'].min()])

out_fn = 'output/ewma_optimize_row3_hc.csv'
df_mse.to_csv(out_fn, index=False)
