"""Wake word SDK для ассистента: 3–10 записей пользователя + общий корпус шума."""

from .config import load_config
from .inference import WakeWordEngine

__all__ = ["load_config", "WakeWordEngine"]
