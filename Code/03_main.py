import pyedflib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os, sys
import lib.Format_edf_to_paquet as fra

file_name = os.getcwd() + "/Data/chb01_01.edf"
path_parquet = os.getcwd() + "/Data/ex.parquet"
edf_f = pyedflib.EdfReader(file_name)
sig, val = fra.edf_to_numpy(edf_f)
fra.edf_to_parquet(edf_f, path_parquet)