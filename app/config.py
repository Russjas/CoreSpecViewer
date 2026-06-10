"""
Global configuration singleton for CoreSpecViewer.

Import the instance, not the class:
    from ..config import config, feature_keys
"""
import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path.home() / ".corespecviewer" / "config.json"

VALID_CONVENTIONS = {'rl_tb', 'lr_tb', 'rl_bt', 'lr_bt'}
CONVENTION_DISPLAY = {
        'rl_tb': 'Right→Left, Top→Bottom',
        'lr_tb': 'Left→Right, Top→Bottom',
        'rl_bt': 'Right→Left, Bottom→Top',
        'lr_bt': 'Left→Right, Bottom→Top',
    }
@dataclass
class AppConfig:

    # Band slice bounds (inclusive-exclusive) per sensor
    swir_slice_start: int = 13
    swir_slice_stop: int = 262
    mwir_slice_start: int = 5
    mwir_slice_stop: int = 142
    rgb_slice_start: int = 0
    rgb_slice_stop: int = -1
    default_slice_start: int = 5
    default_slice_stop: int = -5
    fenix_slice_start: int = 20
    fenix_slice_stop: int = -20

    # Savitzky-Golay
    savgol_window: int = 11
    savgol_polyorder: int = 2

    # Feature detection
    feature_detection_threshold: float = 0.1

    # Box reading convention
    # rl_tb: right-left, top-bottom (default)
    # lr_tb: left-right, top-bottom
    # rl_bt: right-left, bottom-top
    # lr_bt: left-right, bottom-top
    box_convention: str = 'rl_tb'

    min_seg_width: int = 10
    min_seg_area: int = 300

    def as_dict(self) -> dict:
        """Return all settings as a dict. For GUI table population."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def set(self, key: str, value) -> None:
        """Type-safe setter for GUI edits. Raises KeyError for unknown keys."""
        if not hasattr(self, key):
            raise KeyError(f"Unknown config key: '{key}'")
        setattr(self, key, type(getattr(self, key))(value))

    def reset(self) -> None:
        """Reset all fields to declared defaults."""
        for f in fields(self):
            setattr(self, f.name, f.default)

    def save(self) -> None:
        """Persist current settings to disk."""
        try:
            _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CONFIG_PATH.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
            logger.info(f"Config saved to {_CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save config to {_CONFIG_PATH}: {e}")

    def load(self) -> None:
        """Load persisted settings from disk, falling back to defaults on any error."""
        if not _CONFIG_PATH.exists():
            return
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            for key, value in data.items():
                if hasattr(self, key):
                    self.set(key, value)
                else:
                    logger.warning(f"Ignoring unrecognised config key '{key}' from {_CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load config from {_CONFIG_PATH}, using defaults: {e}")


# Shared singleton — always import this instance, never instantiate AppConfig directly
config = AppConfig()
config.load()

feature_keys = [
    '1400W', '1480W', '1550W', '1760W', '1850W',
    '1900W', '2080W', '2160W', '2200W', '2250W',
    '2290W', '2320W', '2350W', '2390W', '2950W',
    '2950AW', '2830W', '3000W', '3500W', '4000W',
    '4000WIDEW', '4470TRUEW', '4500SW', '4500CW',
    '4670W', '4920W', '4000V_NARROWW', '4000shortW', '2950BW'
]

