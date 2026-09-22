from linear_prob import LitSensorPT
import torch
import numpy as np

import os
import json
from typing import Dict, List, Union, Any
from sklearn import preprocessing

from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger

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

# Example /predict payload, kept in its own file so this module stays readable
# (the array is 1 batch x 7 channels x 512 samples). Ships alongside this file
# at examples/sample_predict_request.json; also usable directly with curl:
#   curl -X POST http://localhost:8003/predict \
#        -H "Content-Type: application/json" \
#        -d @examples/sample_predict_request.json
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')
SAMPLE_PREDICT_REQUEST_PATH = os.path.join(EXAMPLES_DIR, 'sample_predict_request.json')
try:
    with open(SAMPLE_PREDICT_REQUEST_PATH) as f:
        SAMPLE_PREDICT_REQUEST = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.warning("Could not load sample predict request: %s", repr(e))
    SAMPLE_PREDICT_REQUEST = {"empatica": []}

# Swagger / OpenAPI docs (equivalent to FastAPI's auto-generated /docs).
# UI is served at /apidocs, raw OpenAPI spec at /apispec_1.json
app.config['SWAGGER'] = {
    'title': 'Sensor Prediction API',
    'uiversion': 3,
    'specs_route': '/apidocs/',
}
swagger = Swagger(app, template={
    "info": {
        "title": "Sensor Prediction API",
        "description": "Stress and mental workload prediction from Empatica sensor data.",
        "version": "1.0.0",
    },
    "definitions": {
        "PredictRequest": {
            "type": "object",
            "required": ["empatica"],
            "properties": {
                "empatica": {
                    "type": "array",
                    "description": "Batch of Empatica sensor windows: [batch][channel][sample]. "
                                    "The sample payload is 1 batch x 7 channels x 512 samples.",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"}
                        }
                    }
                }
            },
            "example": SAMPLE_PREDICT_REQUEST
        }
    }
})

# init model
stress_ckpt = './data/epoch=199-step=3400.ckpt'
mw_ckpt = './data/epoch=199-step=1400.ckpt'
stressmodel = LitSensorPT.load_from_checkpoint(stress_ckpt, map_location=torch.device("cpu"))
stressmodel.eval()
mwmodel = LitSensorPT.load_from_checkpoint(mw_ckpt, map_location=torch.device("cpu"))
mwmodel.eval()


@app.route("/", methods=["GET"])
def read_root():
    """
    Root endpoint
    ---
    tags:
      - General
    responses:
      200:
        description: Basic liveness greeting
        examples:
          application/json: {"Hello": "World"}
    """
    return jsonify({"Hello": "World"})


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check
    ---
    tags:
      - General
    responses:
      200:
        description: Service is healthy and both models are loaded
        examples:
          application/json: {"status": "ok", "models_loaded": {"stress": true, "mental_workload": true}}
      503:
        description: Service is degraded or unhealthy
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
    """
    Predict stress and mental workload
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          $ref: '#/definitions/PredictRequest'
    responses:
      200:
        description: >
          Predicted probabilities, JSON-encoded as a string. The EDA channel
          (index 5) is scaled by 1.5 server-side before inference.
        examples:
          application/json: "{'stress': 0.12345, 'mental_workload': 0.6789}"
    """
    res = {}
    try:
        _req = request.get_json(force=True)
        #print(_req["empatica"])
        req = np.array(_req["empatica"], dtype="float32")

        # Scale the EDA channel by 1.5. Channel layout follows the gateway's
        # ordering: [ACC0, ACC1, ACC2, ACC3, HR, EDA, spare] -> EDA is index 5.
        EDA_CHANNEL_INDEX = 5
        EDA_SCALE = 1.0
        if req.ndim == 3 and req.shape[1] > EDA_CHANNEL_INDEX:
            req[:, EDA_CHANNEL_INDEX, :] *= EDA_SCALE

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
