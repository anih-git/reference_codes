# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:10:52 2020

@author: anirban.hati
"""




# Handling large dataset using parallel computing using dask
# https://docs.dask.org/en/latest/dataframe.html

import time
import psutil
import numpy as np
import pandas as pd
import multiprocessing as mp


# Check the number of cores and memory usage
num_cores = mp.cpu_count()
print("This kernel has ",num_cores,"cores and you can find the information regarding the memory usage:",psutil.virtual_memory())


#reading the file using pandas
%time temp = pd.read_csv("email_master_deduped_nontest.csv") 

#reading the file using dask
import dask.dataframe as dd

#help in dask
dd.read_csv?


%time df1 = dd.read_csv("PartD_Prescriber_PUF_NPI_Drug_17.txt",sep='\t') 


len(df1['npi'])
df1.shape.compute
df1.size.compute
df1.index.size.compute()

aa = df1.describe()

aa.to_csv('aa.csv')


%time df1 = dd.read_csv("npidata_pfile_20050523-20200308.csv")
%time df2 = dd.read_csv("hcp_master_2.csv")

aa = df1.head(100)

aa.to_csv("sample.csv")

df1.columns
df2.columns


%time ab = dd.merge(df1, df2, on=['npi_number'])

# Save parquet file
df1.to_parquet("test.parquet")
