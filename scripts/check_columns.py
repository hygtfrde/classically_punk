import pandas as pd
import numpy as np

def check_column_types(csv_file):
    df = pd.read_csv(csv_file)
    column_info = {}

    for col in df.columns:
        try:
            data = df[col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
        except Exception as e:
            print(f"Error converting column '{col}': {e}")
            continue

        first_valid_value = data.dropna().iloc[0]
        data_type = type(first_valid_value)
        if isinstance(first_valid_value, np.ndarray):
            shape = first_valid_value.shape
        else:
            shape = "Scalar"

        column_info[col] = {"Data Type": data_type, "Shape": shape}

    for col, info in column_info.items():
        print(f"Column: {col}")
        print(f"  Data Type: {info['Data Type']}")
        print(f"  Shape: {info['Shape']}")
        print()


csv_file = 'df_output/v5_5.csv'
check_column_types(csv_file)
