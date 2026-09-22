import requests
import json
import numpy as np

url = "http://localhost:8005/predict"
params = {"empatica":[[[ ],
       [ ],
       [ ],
       [ ],
       [ ],
       [ ],
       [ ]]]}    

import time
# for i in range(100000):
#     r=requests.post(url, json=params)
#     #print(r)
#     print(r.json())
#     time.sleep(1)

import pandas as pd 

df = pd.read_csv('nback_20251204_empatica_idun_merged.csv')  
df = df[df['eda'].notna()]

#df = df.loc[df['nback_level']==4.0]
#df = df.loc[(df['marker']=="phase_baseline_1") ]
print(len(df))
eda4= df['eda'].tolist()
print(len(eda4))
print(eda4[:4])

from scipy import signal

eda64 = signal.resample_poly(eda4, up=256, down=64)*1000
print(len(eda64))
print(eda64[:4])


from linear_prob import LitSensorPT
import torch
import numpy as np

import os
import json
from typing import Dict, List, Union, Any
from sklearn import preprocessing


# init model
ckpt_path = './data/epoch=199-step=3400.ckpt'
#ckpt_path = './data/epoch=199-step=1400.ckpt'
#ckpt_path = './data/epoch=199-step=6600.ckpt'
model = LitSensorPT.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
model.eval()
stress_prob = []
for idx in range(len(eda4)-512):
    padding = [0.0 for i in range(512)]
    params["empatica"][0][0] =  padding
    params["empatica"][0][1] =  padding
    params["empatica"][0][2] =  padding
    params["empatica"][0][3] =  padding
    params["empatica"][0][4] =  padding
    params["empatica"][0][5] =  eda64[idx*4:idx*4+512].tolist()
    params["empatica"][0][6] =  padding
    try:
        req = np.array(params["empatica"], dtype="float32")
        test_dataset = torch.from_numpy(req)
        _, logit = model(test_dataset)
        #print('Y hat',torch.argmax(logit,  dim=-1))
        probs = logit.detach().numpy()[0]
        norm_val = -.8
        probs = np.append(probs,[norm_val])
        print(probs)
        probs_norm = (probs - probs.min()) / (probs - probs.min()).sum()
        #print(probs_norm)
        #res = dict(zip(["no stress","stress"], probs_norm))
        res = dict(zip(["stress"], probs_norm[1:2]))
        #print(probs_norm[1])
        #print(df['subject'][idx])
        print(res)
        stress_prob.append(probs_norm[1])
    except Exception as e:
        print('%s', repr(e))

    if (idx*4 > len(eda64)-512):
        break
print(len(stress_prob))
df = df.drop(df.index[len(stress_prob):])
df['stress_prob'] = stress_prob
df.to_csv('test.csv', encoding='utf-8', index=False, header=True)
#print((params["empatica"][0][5]))
# for i in range(100000):

#     idx = i#+1350#int(20484*3/4)
#     print(len(eda64[idx:]))
#     padding = [0.0 for i in range(512)]
#     params["empatica"][0][0] =  padding
#     params["empatica"][0][1] =  padding
#     params["empatica"][0][2] =  padding
#     params["empatica"][0][3] =  padding
#     params["empatica"][0][4] =  padding
#     params["empatica"][0][5] =  eda64[idx:idx+512].tolist()
#     params["empatica"][0][6] =  padding
#     r=requests.post(url, json=params)
#     print(r.json())
#     time.sleep(1)
