import numpy as np

npz_file_path = 'df_output/raw_features_only_raw_3.npz'
data = np.load(npz_file_path)

# Print keys and data
for key in data.keys():
    print(f"Key: {key}")
    print(data[key])