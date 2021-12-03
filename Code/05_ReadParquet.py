import pandas as pd
import os

file_name = os.path.abspath("D:\\UAB\\4to\\DataSetTFG/CVC/dataframes/cvc_eeg_power_filt_none_window_40_30.parquet")

dic = pd.read_parquet(file_name)

print(dic.columns)