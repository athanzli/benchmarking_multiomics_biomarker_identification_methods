# https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt

import pandas as pd
hgnc_df = pd.read_csv('/home/athan.li/eval_bk/data/TCGA/hgnc_complete_set.txt', sep='\t')
hgnc_df = hgnc_df[['prev_symbol', 'alias_symbol', 'status', 'symbol']]
# hgnc_df['status'].value_counts()

df1 = hgnc_df.copy()
df1['prev_symbol'] = df1['prev_symbol'].str.split('|')
df1 = df1.explode('prev_symbol')
df1 = df1.loc[df1['prev_symbol'].notna()]
df1 = df1[['prev_symbol', 'symbol']].drop_duplicates()
df1.rename(columns={'prev_symbol': 'alternative_symbol'}, inplace=True)

df2 = hgnc_df.copy()
df2['alias_symbol'] = df2['alias_symbol'].str.split('|')
df2 = df2.explode('alias_symbol')
df2 = df2.loc[df2['alias_symbol'].notna()]
df2 = df2[['alias_symbol', 'symbol']].drop_duplicates()
df2.rename(columns={'alias_symbol': 'alternative_symbol'}, inplace=True)
df = pd.concat([df1, df2], axis=0)
df = df.drop_duplicates()
df.index = df['alternative_symbol']
df.to_csv('./hgnc_alternative_symbol_mapping.csv')
