import pandas as pd
import numpy as np

# 导入数据
fn = 'data/trans_matrix.csv'
df = pd.read_csv(fn)

# jm处理
df_jm = df[df['InstrumentID'].astype(str).str.slice(stop=2) == 'jm']
df_jm.reset_index(drop=True, inplace=True)

df_jm['3_2_rate'] = np.nan
df_jm['3_1_rate'] = np.nan

for row in range(len(df_jm)):
    # 5*5矩阵
    _df = pd.DataFrame(df_jm.loc[row][:25].values.reshape(5, 5), index=[0, 1, 2, 3, 4], columns=[0, 1, 2, 3, 4])
    _df_rate = _df.div(_df.T.apply(sum), axis=0)
    df_jm.loc[row, '3_2_rate'] = _df_rate.at[3, 2]
    df_jm.loc[row, '3_1_rate'] = _df_rate.at[3, 1]

df_jm['ewma_3_2'] = df_jm['3_2_rate'].ewm(alpha=0.1, min_periods=5).mean()
df_jm['forecast_3_1'] = (4 / 5) * df_jm['ewma_3_2']
df_jm = df_jm[['TradeDay', '3_2_rate', '3_1_rate', 'ewma_3_2', 'forecast_3_1']]

# hc处理
df_hc = df[df['InstrumentID'].astype(str).str.slice(stop=2) == 'hc']
df_hc.reset_index(drop=True, inplace=True)

df_hc['3_2_rate'] = np.nan
df_hc['3_1_rate'] = np.nan

for row in range(len(df_hc)):
    _df = pd.DataFrame(df_hc.loc[row][:25].values.reshape(5, 5), index=[0, 1, 2, 3, 4], columns=[0, 1, 2, 3, 4])
    _df_rate = _df.div(_df.T.apply(sum), axis=0)
    df_hc.loc[row, '3_2_rate'] = _df_rate.at[3, 2]
    df_hc.loc[row, '3_1_rate'] = _df_rate.at[3, 1]

df_hc['ewma_3_2'] = df_hc['3_2_rate'].ewm(alpha=0.1, min_periods=29).mean()
df_hc['forecast_3_1'] = (4 / 5) * df_hc['ewma_3_2']
df_hc = df_hc[['TradeDay', '3_2_rate', '3_1_rate', 'ewma_3_2', 'forecast_3_1']]

# 输出EWMA预测值
out1_fn = 'output/jm_row3.csv'
df_jm.to_csv(out1_fn, index=False)
out2_fn = 'output/hc_row3.csv'
df_hc.to_csv(out2_fn, index=False)
