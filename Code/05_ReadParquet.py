import pandas as pd
import os

file_name = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + 'chb01' + '/parquet/')

dic = pd.read_parquet(file_name)

print(dic.columns)
print(dic[0:30])