import pandas as pd

chunk_size = 1
csv_file_path = 'df_output/v4_new_getdata().csv'

for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
    print(chunk.head())