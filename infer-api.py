"""Farm RVC voice-conversion service.

Sidecar for the Farm API's Morph tasks. The API worker starts this process from
code/rvc/.venv (sys/api lib/clients/rvc.py), waits for /health to report
model_loaded, posts conversions to /convert and stops the process when idle.
The inference code under ./rvc is the engine the API previously ran in-process;
this file adds only the HTTP surface and model caching.

Run: .venv/bin/python infer-api.py --port 7878
Env: MODELS_PATH (voice model directories), BASE_MODEL_PATH (hubert_base.pt and
rmvpe.pt). Both are inherited from the API worker's environment.
"""
import argparse
import gc
import logging
import os
import sys
import threading
import time
import traceback
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("fairseq").setLevel(logging.WARNING)
logger = logging.getLogger("rvc-service")

app = FastAPI(title="Farm RVC service")

STATE = {
    "status": "loading",
    "error": None,
    "voice": None,
    "busy": False,
    "conversions": 0,
    "started": time.time(),
}
ENGINE = None
ENGINE_LOCK = threading.Lock()


class ConvertRequest(BaseModel):
    model_name: str
    input_path: str
    output_path: str
    sid: int = 0
    f0up_key: int = 0
    filter_radius: int = 3
    index_rate: float = 0.3
    hop_length: int = 128
    f0method: str = "rmvpe"
    split_audio: bool = False
    f0autotune: bool = False
    rms_mix_rate: float = 1.0
    protect: float = 0.33
    resample_sr: int = 0
    f0_file: Optional[str] = None
    seed: Optional[int] = None


class Engine:
    """Hubert plus one cached voice model, mirroring the former in-process client."""

    def __init__(self):
        import torch
        from rvc.configs.config import Config
        from rvc.models.hubert import Hubert

        self.torch = torch
        self.config = Config()
        self.config.initialize_device()
        self.hubert = Hubert(self.config)
        self.hubert.load()
        self.model_name = None
        self.voice = None
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.version = None
        self.tgt_sr = None
        self.n_spk = None

    def unload_voice(self):
        self.net_g = None
        self.vc = None
        self.cpt = None
        self.voice = None
        self.model_name = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def load_voice(self, model_name):
        if self.net_g is not None and model_name == self.model_name:
            return
        self.unload_voice()
        from rvc.models.voice import Voice

        voice = Voice(model_name)
        if not voice.pth_file:
            raise FileNotFoundError(f"No .pth file for model {model_name}")
        (
            self.n_spk,
            self.tgt_sr,
            self.net_g,
            self.vc,
            self.cpt,
            self.version,
        ) = self._load_checkpoint(voice.pth_file)
        self.voice = voice
        self.model_name = model_name
        logger.info(
            "Loaded voice %s (version %s, sr %s, f0 %s)",
            model_name, self.version, self.tgt_sr, self.cpt.get("f0", 1),
        )

    def _load_checkpoint(self, weight_root):
        from rvc.infer.pipeline import VC
        from rvc.lib.infer_pack.models import (
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs256NSFsid_nono,
            SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs768NSFsid_nono,
        )

        torch = self.torch
        cpt = torch.load(weight_root, map_location="cuda:0", weights_only=True)
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")
        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.config.is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        elif version == "v2":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=self.config.is_half)
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        else:
            raise ValueError(f"Unsupported model version: {version}")
        del net_g.enc_q
        logger.info(net_g.load_state_dict(cpt["weight"], strict=False))
        net_g.eval().to(self.config.device)
        if self.config.is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()
        vc = VC(tgt_sr, self.config)
        n_spk = cpt["config"][-3]
        return n_spk, tgt_sr, net_g, vc, cpt, version

    def _index_file(self):
        index_file = self.voice.index_file
        if not index_file:
            logger.warning("Model %s has no index file; converting without retrieval", self.model_name)
            return ""
        return (
            index_file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            .replace("trained", "added")
        )

    def _convert_file(self, request, input_path, tgt_sr, if_f0, index_file):
        import numpy as np
        from rvc.lib.utils import load_audio

        audio = load_audio(input_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        if not self.hubert.model:
            self.hubert.load()
        return self.vc.pipeline(
            self.hubert.model,
            self.net_g,
            request.sid,
            audio,
            input_path,
            request.f0up_key,
            request.f0method,
            index_file,
            request.index_rate,
            if_f0,
            request.filter_radius,
            tgt_sr,
            request.resample_sr,
            request.rms_mix_rate,
            self.version,
            request.protect,
            request.hop_length,
            request.f0autotune,
            f0_file=request.f0_file,
        )

    def convert(self, request):
        import soundfile as sf
        from rvc.lib.tools.split_audio import merge_audio, process_audio

        self.load_voice(request.model_name)
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        if request.seed is not None:
            self.torch.manual_seed(request.seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(request.seed)
        tgt_sr = self.tgt_sr
        if tgt_sr != request.resample_sr >= 16000:
            tgt_sr = request.resample_sr
        if_f0 = self.cpt.get("f0", 1)
        index_file = self._index_file()

        if request.split_audio:
            result, new_dir_path = process_audio(request.input_path)
            if result == "Error":
                raise RuntimeError("Error with Split Audio")
            dir_path = new_dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            paths = []
            if dir_path != "":
                paths = [
                    os.path.join(root, name)
                    for root, _, files in os.walk(dir_path, topdown=False)
                    for name in files
                    if name.endswith(".wav") and root == dir_path
                ]
            for path in paths:
                segment = self._convert_file(request, path, tgt_sr, if_f0, index_file)
                sf.write(path, segment, tgt_sr, format="WAV")
            merge_timestamps_file = os.path.join(
                os.path.dirname(str(new_dir_path)),
                f"{os.path.basename(str(request.input_path)).split('.')[0]}_timestamps.txt",
            )
            tgt_sr, audio_opt = merge_audio(merge_timestamps_file)
            os.remove(merge_timestamps_file)
        else:
            audio_opt = self._convert_file(request, request.input_path, tgt_sr, if_f0, index_file)

        sf.write(request.output_path, audio_opt, tgt_sr, format="WAV")
        return tgt_sr, len(audio_opt) / float(tgt_sr)


def _load_engine():
    global ENGINE
    try:
        for name in ("MODELS_PATH", "BASE_MODEL_PATH"):
            value = os.environ.get(name)
            if not value or not os.path.isdir(value):
                raise RuntimeError(f"{name} is unset or not a directory: {value!r}")
        started = time.time()
        ENGINE = Engine()
        STATE["status"] = "ready"
        logger.info("Hubert loaded on %s in %.1fs", ENGINE.config.device, time.time() - started)
    except Exception as error:
        STATE["status"] = "error"
        STATE["error"] = f"{type(error).__name__}: {error}"
        logger.error("Engine failed to load:\n%s", traceback.format_exc())


@app.on_event("startup")
def startup():
    threading.Thread(target=_load_engine, name="rvc-engine-load", daemon=True).start()


@app.get("/health")
def health():
    import torch

    return {
        "status": STATE["status"],
        "model_loaded": STATE["status"] == "ready",
        "voice": STATE["voice"],
        "busy": STATE["busy"],
        "conversions": STATE["conversions"],
        "uptime_seconds": round(time.time() - STATE["started"], 1),
        "device": ENGINE.config.device if ENGINE else None,
        "is_half": ENGINE.config.is_half if ENGINE else None,
        "torch": torch.__version__,
        "error": STATE["error"],
    }


@app.post("/convert")
def convert(request: ConvertRequest):
    if STATE["status"] != "ready":
        raise HTTPException(503, f"service not ready: {STATE['status']} {STATE['error'] or ''}".strip())
    if not os.path.isfile(request.input_path):
        raise HTTPException(400, f"input file not found: {request.input_path}")
    if not os.path.isabs(request.output_path):
        raise HTTPException(400, "output_path must be absolute")
    with ENGINE_LOCK:
        STATE["busy"] = True
        started = time.time()
        try:
            tgt_sr, duration = ENGINE.convert(request)
            STATE["conversions"] += 1
            STATE["voice"] = ENGINE.model_name
        except HTTPException:
            raise
        except Exception as error:
            logger.error("Conversion failed:\n%s", traceback.format_exc())
            raise HTTPException(500, f"{type(error).__name__}: {error}")
        finally:
            STATE["busy"] = False
    elapsed = time.time() - started
    logger.info("Converted %s with %s in %.1fs", request.input_path, request.model_name, elapsed)
    return {
        "output_path": request.output_path,
        "sample_rate": tgt_sr,
        "duration_seconds": round(duration, 3),
        "elapsed_seconds": round(elapsed, 3),
        "model_name": request.model_name,
    }


@app.post("/unload")
def unload():
    with ENGINE_LOCK:
        if ENGINE is not None:
            ENGINE.unload_voice()
        STATE["voice"] = None
    return {"ok": True}


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Farm RVC voice-conversion service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7878)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
