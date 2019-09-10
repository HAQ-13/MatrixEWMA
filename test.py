import pandas as pd

fn = 'output/ewma_optimize_row4_jm.csv'
df_mse = pd.read_csv(fn)

print(df_mse.loc[df_mse['sum'] == df_mse['sum'].min()])

# print('\n'.join((' '.join(['{}*{}={}'.format(y, x, y*x) for y in range(1, x+1)])) for x in range(1, 10)))
