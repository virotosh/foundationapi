# foundationapi

A Flask API that serves stress and mental workload predictions from Empatica
sensor data, using two fine-tuned `LitSensorPT` (Sensor Transformer)
checkpoints.

## Requirements

- **Python 3.12.11**
- `pip`
- The two model checkpoint files (see [Model checkpoints](#model-checkpoints) below) — these are **not** included in the repo

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/virotosh/foundationapi.git
   cd foundationapi
   ```

2. **Make sure you're on Python 3.12.11**

   ```bash
   python3 --version
   ```

   If you don't have 3.12.11 installed, get it via [pyenv](https://github.com/pyenv/pyenv)
   (`pyenv install 3.12.11 && pyenv local 3.12.11`) or your system's package manager.

3. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > `requirements.txt` is a full environment export, so it pulls in more than
   > the API strictly needs — installation can take a few minutes.
   > `torch` and `torchvision` are unpinned; the default `pip install` gives
   > you the CPU build. If you need GPU support, install the matching CUDA
   > build from the [PyTorch install page](https://pytorch.org/get-started/locally/)
   > *before* running `pip install -r requirements.txt`, or reinstall it
   > afterward.

## Model checkpoints

`api.py` loads two checkpoint files from a local `data/` folder, which is
excluded from the repo via `.gitignore` (checkpoints are large binary files
not meant for git):

```python
stress_ckpt = './data/epoch=199-step=3400.ckpt'
mw_ckpt = './data/epoch=199-step=1400.ckpt'
```

`data/` also holds the pretrained foundation model checkpoint that
`LitSensorPT` defaults to in `linear_prob.py` (used for fine-tuning /
pretraining scripts rather than by `api.py` directly):

```python
# linear_prob.py
def __init__(self, ckpt=f'./data/epoch=199-step=10800.ckpt'):
```

Before running the API, create the folder and place all three files inside
it:

```bash
mkdir -p data
# copy/download the checkpoints into data/, so you end up with:
#   data/epoch=199-step=3400.ckpt    (stress model)
#   data/epoch=199-step=1400.ckpt    (mental workload model)
#   data/epoch=199-step=10800.ckpt   (pretrained foundation model)
```

The API will fail to start if either of the first two files is missing from
`data/`; the foundation model checkpoint is only needed if you run
`finetune.py` or `pretrain.py`.

## Running the API

```bash
python api.py
```

This starts the Flask server on `0.0.0.0:8003`. Logs are appended to
`api.txt` in the project root.

## API docs

Interactive Swagger UI is available once the server is running:

- **UI:** http://localhost:8003/apidocs/
- **Raw OpenAPI spec:** http://localhost:8003/apispec_1.json

## Endpoints

| Method | Path        | Description                                             |
|--------|-------------|----------------------------------------------------------|
| GET    | `/`         | Basic liveness greeting                                  |
| GET    | `/health`   | Health check — confirms both models loaded                |
| POST   | `/predict`  | Predicts `stress` and `mental_workload` from sensor data  |

### `POST /predict`

Expects a JSON body with an `empatica` key: a nested array shaped
`[batch][channel][sample]`, where the 7 channels follow this order
(from `linear_prob.py`):

```
["ACC0", "ACC1", "ACC2", "BVP", "HR", "EDA", "TEMP"]
```

A ready-to-use sample request is included at
`examples/sample_predict_request.json` (shape `1 × 7 × 512`):

```bash
curl -X POST http://localhost:8003/predict \
     -H "Content-Type: application/json" \
     -d @examples/sample_predict_request.json
```

You can also try this directly from the Swagger UI at `/apidocs/` — the
`empatica` field is pre-filled with this same sample payload.

## Project structure

```
foundationapi/
├── api.py                          # Flask API (this README's main entry point)
├── linear_prob.py                  # LitSensorPT model wrapper used by api.py
├── model/                          # Sensor Transformer architecture
├── util/                           # Data loading / preprocessing utilities
├── examples/
│   └── sample_predict_request.json # Example /predict request body
├── data/                           # (not in repo) place model checkpoints here
│   ├── epoch=199-step=3400.ckpt    #   stress model
│   ├── epoch=199-step=1400.ckpt    #   mental workload model
│   └── epoch=199-step=10800.ckpt   #   pretrained foundation model
├── finetune.py                     # Fine-tuning script
├── pretrain.py                     # Pretraining script
├── requirements.txt
└── api.txt                         # Runtime log output (created on first run)
```

## Troubleshooting

- **`FileNotFoundError` on startup** — the checkpoint files aren't in `data/`, or the filenames don't match exactly (they include an `=` character, so quote/escape paths carefully on some shells).
- **Predictions look off** — double-check the channel order of your `empatica` payload matches `["ACC0", "ACC1", "ACC2", "BVP", "HR", "EDA", "TEMP"]`.
- **`pip install` fails on `torch`/`torchvision`** — install a matching build for your platform from [pytorch.org](https://pytorch.org/get-started/locally/) first.
