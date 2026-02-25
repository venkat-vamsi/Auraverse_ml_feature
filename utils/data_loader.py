import os
import pandas as pd

def load_multiple_csv(folder_path):
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    dfs = []

    for file in all_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)