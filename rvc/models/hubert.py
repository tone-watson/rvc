import os
import threading

from dotenv import load_dotenv

try:
    from fairseq import checkpoint_utils

    FAIRSEQ_AVAILABLE = True
except ImportError:
    FAIRSEQ_AVAILABLE = False
    checkpoint_utils = None

from rvc.configs.config import Config

load_dotenv()


class Hubert:
    lock = threading.Lock()
    base_path = os.getenv("BASE_MODEL_PATH", "/")

    def __init__(self, config: Config):
        """Init"""
        self.config = config
        self.model = None

    def load(self):
        """Load the Hubert model
        for audio processing.
        """
        if not FAIRSEQ_AVAILABLE:
            raise ImportError("fairseq is not installed. Cannot load Hubert model.")

        with Hubert.lock:
            if not self.model:
                models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                    [f"{self.base_path}/hubert_base.pt"]
                )

                self.model = models[0].to(self.config.device)
                if self.config.is_half:
                    self.model = self.model.half()
                else:
                    self.model = self.model.float()

                self.model.eval()
