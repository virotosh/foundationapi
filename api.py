from linear_prob import LitSensorPT
import torch
import numpy as np

import os
import json
from typing import Dict, List, Union, Any
from sklearn import preprocessing

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import logging
logging.basicConfig(filename='api.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running API")
logger = logging.getLogger(__name__)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# init model
stress_ckpt = './data/epoch=199-step=3400.ckpt'
mw_ckpt = './data/epoch=199-step=1400.ckpt'
stressmodel = LitSensorPT.load_from_checkpoint(stress_ckpt, map_location=torch.device("cpu"))
stressmodel.eval()
mwmodel = LitSensorPT.load_from_checkpoint(mw_ckpt, map_location=torch.device("cpu"))
mwmodel.eval()

@app.get("/")
def read_root():
    return {"Hello": "World"}
    
@app.post("/predict")
async def get_probs(request: Request):
    res = {}
    try:
        _req = await request.json()
        #print(_req["empatica"])
        req = np.array(_req["empatica"], dtype="float32")
        test_dataset = torch.from_numpy(req)
        _, stresslogit = stressmodel(test_dataset)
        #print('Y hat',torch.argmax(logit,  dim=-1))
        #logger.info('Y hat: '+str(torch.argmax(logit,  dim=-1)))
        stressprobs = stresslogit.detach().numpy()[0]
        _, mwlogit = mwmodel(test_dataset)
        mwprobs = mwlogit.detach().numpy()[0]
        #print(stressprobs)
        #res = dict(zip(["no stress","stress"], stressprobs))
        res = dict(zip(["stress","mental_workload"], [round(stressprobs[1], 5),round(mwprobs[1],5) ] ))
    except Exception as e:
        print('%s', repr(e))
    
    return json.dumps(str(res))

if __name__=="__main__":
    
    
    print('api intializing')
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8008,
        log_level="info",
        reload=False,
    )
