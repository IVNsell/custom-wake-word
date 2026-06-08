"""Wake word SDK: 3-10 user recordings + shared platform noise corpus."""

from .config import load_config
from .inference import WakeWordEngine

__all__ = ["load_config", "WakeWordEngine"]
