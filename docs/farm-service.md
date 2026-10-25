# Farm RVC service (`infer-api.py`)

Sidecar used by the Farm API's Morph voice-conversion tasks. The API worker
(`sys/api/lib/clients/rvc.py`) starts it from this checkout's `.venv`, waits for
`/health` to report `model_loaded`, posts conversions, and stops it after 300 s
without work. It is not a systemd service and does not need `run.sh`.

## Run by hand

```sh
cd /srv/farm/code/rvc
MODELS_PATH=/home/gradywoodruff/models/rvc/voice \
BASE_MODEL_PATH=/home/gradywoodruff/models/base \
.venv/bin/python infer-api.py --port 7878
```

`MODELS_PATH` holds one directory per voice (`.pth` and `.index`);
`BASE_MODEL_PATH` holds `hubert_base.pt` and `rmvpe.pt`. Both are inherited from
the API worker's environment in production.

## Endpoints

- `GET /health` — `status` (`loading`, `ready`, `error`), `model_loaded`
  (hubert resident), `voice` (cached model), `busy`, `conversions`, `device`,
  `is_half`, `torch`, `error`.
- `POST /convert` — JSON: `model_name`, `input_path`, `output_path` (absolute),
  and the Morph settings `sid`, `f0up_key`, `filter_radius`, `index_rate`,
  `hop_length`, `f0method`, `split_audio`, `f0autotune`, `rms_mix_rate`,
  `protect`, `resample_sr`, `f0_file`, optional `seed`. Writes the WAV and
  returns `output_path`, `sample_rate`, `duration_seconds`, `elapsed_seconds`.
  One conversion runs at a time; errors return HTTP 500 with the exception text.
- `POST /unload` — drop the cached voice model and empty the CUDA cache.

## Engine

`./rvc` is the inference package that `sys/api` previously ran in-process
(configs, `infer/pipeline.py`, `lib/infer_pack`, the rmvpe and FCPE f0
predictors, `models/hubert.py`, `models/voice.py`). Training and download tooling
stayed behind and were not copied. The package imports unchanged under this
checkout's managed Python 3.11 with torch 2.5.1+cu121, fairseq 0.12.3 and numpy
2.3. `infer-api.py` adds only the HTTP surface, hubert loaded once at startup,
and a one-model cache.

## Validation, 2026-10-25

Against the former in-process client on the same 14.2 s input and model: seeded
outputs agree to 43.5 dB SNR, inside the pipeline's own run-to-run spread
(57 to 60 dB seeded; about 10 dB unseeded, because the synthesizer samples noise
at inference time). Cold conversion 7 s including process boot; warm 0.5 s.
