# NEURIM

NEURIM is an online human-in-the-loop optimizer. It searches a diffusion
model's anchor space using one scalar reward decoded from EEG.

It does not reconstruct images from brain activity. The EEG pipeline produces
a noisy approval signal; the optimizer uses that signal to choose which image
to render next.

## Architecture

```text
EEG -> FAA reward -> Optimizer -> z -> Private diffusion server -> PNG
                            |                              |
                            +-> local live_frame.png <-----+
```

The runtime is split across two machines:

| Component | Location | Package |
|---|---|---|
| Frontend API, EMOTIV lifecycle, and session lifecycle | Local machine | `src/server/api/` |
| Prompt curation | Local machine | `src/session/curation.py` |
| EEG/mock optimizer session | Local machine | `src/session/` |
| Manifest-driven diffusion renderer | Private GPU machine | `src/server/diffusion/` |

Files in `scripts/` are compatibility CLI entrypoints. Runtime logic belongs
under `src/`.

## Manifest-Driven Flow

1. Curate the user's prompt into a seven-anchor manifest:

```bash
python scripts/run_prompt_curation.py \
  --user-prompt "Woolly mammoths" \
  --out data/processed/prompt_sessions/mammoths.json \
  --verbose
```

2. Copy the manifest to the private GPU machine and start its render server:

```bash
python scripts/run_general_stable_diffusion.py \
  --session-manifest data/processed/prompt_sessions/mammoths.json \
  --host 0.0.0.0 \
  --port 8766
```

Select a specific GPU when the private machine exposes multiple CUDA devices:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_general_stable_diffusion.py \
  --session-manifest data/processed/prompt_sessions/mammoths.json \
  --host 0.0.0.0 \
  --port 8766
```

The private server exposes:

- `GET /manifest`: active manifest metadata and render endpoint.
- `POST /render`: accepts `{"z": [...], "frame_size": 512}` and returns PNG bytes.

3. Start the local API bridge. It immediately attempts to connect to EMOTIV
   Cortex, retries every 60 seconds on failure, and calibrates for 30 seconds
   once connected:

```bash
python scripts/api_server.py --host 127.0.0.1 --port 8000
```

4. Validate the optimizer without EEG hardware:

```bash
python scripts/run_mock_optimizer.py --server-url http://GPU_HOST:8766 --seed 3
```

5. Run the EEG optimizer directly, without the frontend API:

```bash
python scripts/run_real_eeg_optimizer.py --server-url http://GPU_HOST:8766
```

Use `--mock` for synthetic EEG while retaining the FAA and optimizer wiring:

```bash
python scripts/run_real_eeg_optimizer.py \
  --mock \
  --server-url http://GPU_HOST:8766
```

Both optimizer clients atomically update `data/processed/live_frame.png`.
The basic viewer at `frontend/live_view.html` polls that file. Real EEG sessions
also save `session_start.png` and `session_end.png`.

## Frontend API

Start the local API bridge:

```bash
python scripts/api_server.py --host 127.0.0.1 --port 8000
```

Run the frontend separately:

```bash
cd frontend-app
NEURIM_API_URL=http://127.0.0.1:8000 npm run dev
```

The API exposes:

- `GET /health`
- `GET /eeg/status`
- `POST /eeg/retry`
- `POST /session/start`
- `POST /session/stop`
- `GET /session/status`
- `GET /session/logs`

The API owns the EMOTIV connection. On startup it connects to Cortex, retries
every 60 seconds if no headset is available, and runs a 30-second FAA baseline
calibration immediately after connection. `POST /eeg/retry` forces an immediate
retry.

When `POST /session/start` receives a prompt, the API curates a local
manifest, compares it with the private diffusion server's `GET /manifest`
response, and starts the optimizer only if they match. The private diffusion
server must already be running with that manifest.

## Core Services

- `src/signal_service/`: EEG sources, baseline calibration, and frontal alpha
  asymmetry reward computation.
- `src/optimizer/`: optimizer algorithms and the
  `CALIBRATE -> EXPLORE -> REFINE -> SETTLE` state machine.
- `src/generator/`: manifest validation, interpolation, and generator support.
- `src/session/`: diffusion HTTP client, atomic frame storage, shared optimizer
  render loop, and real/mock session runners.
- `src/server/api/`: FastAPI app, request models, settings, logs, and subprocess
  lifecycle management.
- `src/server/diffusion/`: anchor renderer, HTTP transport, pipeline loading,
  and GPU server composition.

`config/config.yaml` controls FAA windows, optimizer behavior, seven-dimensional
anchor search, frame size, and model defaults.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-diffusion.txt  # GPU diffusion machine only
pytest
```

Prompt curation requires `OPENAI_API_KEY`. Real EEG acquisition requires the
EMOTIV credentials described in `.env.example`.

## Documentation

- [Preference reward pipeline](docs/preference-reward.md)
