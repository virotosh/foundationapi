from linear_prob import LitSensorPT
import torch
import numpy as np

import os
import json
from typing import Dict, List, Union, Any
from sklearn import preprocessing

from flask import Flask, request, jsonify
from flask_cors import CORS

import logging
logging.basicConfig(filename='api.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running API")
logger = logging.getLogger(__name__)


app = Flask(__name__)

# Add CORS support (equivalent to the FastAPI CORSMiddleware config)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["*"], allow_headers=["*"])

# init model
stress_ckpt = './data/epoch=199-step=3400.ckpt'
mw_ckpt = './data/epoch=199-step=1400.ckpt'
stressmodel = LitSensorPT.load_from_checkpoint(stress_ckpt, map_location=torch.device("cpu"))
stressmodel.eval()
mwmodel = LitSensorPT.load_from_checkpoint(mw_ckpt, map_location=torch.device("cpu"))
mwmodel.eval()


@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"Hello": "World"})


@app.route("/health", methods=["GET"])
def health_check():
    """
    Basic liveness/readiness check.
    Confirms the process is up and that both models loaded successfully.
    """
    try:
        models_ready = stressmodel is not None and mwmodel is not None
        status = "ok" if models_ready else "degraded"
        return jsonify({
            "status": status,
            "models_loaded": {
                "stress": stressmodel is not None,
                "mental_workload": mwmodel is not None,
            }
        }), 200 if models_ready else 503
    except Exception as e:
        logger.error("Health check failed: %s", repr(e))
        return jsonify({"status": "error", "detail": str(e)}), 503


@app.route("/predict", methods=["POST"])
def get_probs():
    res = {}
    try:
        _req = request.get_json(force=True)
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
        res = dict(zip(["stress", "mental_workload"], [round(stressprobs[1], 5), round(mwprobs[1], 5)]))
    except Exception as e:
        print('%s', repr(e))
        logger.error("Predict failed: %s", repr(e))

    return json.dumps(str(res))


if __name__ == "__main__":

    print('api intializing')

    app.run(
        host="0.0.0.0",
        port=8003,
        debug=False,
    )