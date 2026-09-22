import requests
import json
import numpy as np

url = "http://localhost:8003/predict"
params = {"empatica":[[[ ],
       [ ],
       [ ],
       [ ],
       [ ],
       [ ],
       [ ],
#       [ ],
#       [ ],
#       [ ],
#       [ ],
       ]]}    

import time
# for i in range(100000):
#     r=requests.post(url, json=params)
#     #print(r)
#     print(r.json())
#     time.sleep(1)

import pandas as pd 

df = pd.read_csv('20260114_171701/SUBJECT005_20260113_180915_plux_processed.csv')  
#df = pd.read_csv('nback_20251204_empatica_idun_merged.csv')  
df = df[df['eda'].notna()]

#df = df.loc[df['nback_level']==4.0]
#df = df.loc[(df['marker']=="phase_baseline_1") ]
print(len(df))
eda4= df['eda'].tolist()
print(len(eda4))
print(eda4[:4])

from scipy import signal

eda64 = signal.resample_poly(eda4, up=64, down=64)*1.5#*4000#1.5
print(len(eda64))
print(eda64[:4])


for idx in range(9,len(eda64)-512):
    padding = [0.0 for i in range(512)]
    params["empatica"][0][0] =  padding
    params["empatica"][0][1] =  padding
    params["empatica"][0][2] =  padding
    params["empatica"][0][3] =  padding
    params["empatica"][0][4] =  df['hr'].tolist()[idx:idx+512]
    params["empatica"][0][5] =  eda64[idx:idx+512].tolist()
    params["empatica"][0][6] =  padding
#    params["empatica"][0][7] =  padding
#    params["empatica"][0][8] =  padding
#    params["empatica"][0][9] =  padding
#    params["empatica"][0][10] =  padding
    r=requests.post(url, json=params)
    print("status:", r.status_code)
    print("headers:", dict(r.headers))
    print("body:", r.text)
    #print(params)
    #time.sleep(1)
