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

FEATURE_BOUNDS = {
    '1400W':  (1387, 1445, 1350, 1450),
    '1480W':  (1471, 1491, 1440, 1520),
    '1550W':  (1520, 1563, 1510, 1610),
    '1760W':  (1751, 1764, 1730, 1790),
    '1850W':  (1749, 1949, 1720, 1980),
    '1900W':  (1840, 1990, 1820, 2010), 
    '2080W':  (1980, 2180, 1950, 2200), 
    '2160W':  (2159, 2166, 2138, 2179),
    '2200W':  (2185, 2215, 2120, 2245),
    '2250W':  (2248, 2268, 2230, 2280),
    '2290W':  (2279, 2310, 2270, 2350), 
    '2320W':  (2300, 2340, 2295, 2355),
    '2350W':  (2320, 2366, 2310, 2370),
    '2390W':  (2377, 2406, 2375, 2435),
    '2830W':  (2790, 2890, 2790, 2920),  
    '2950W':  (2920, 2980, 2900, 3000),
    '2950AW': (2900, 2960, 2900, 3000),
    '2950BW': (2920, 2990, 2790, 3200),
    '3000W':  (2900, 3100, 2795, 3900),  
    '3500W':  (3400, 3600, 3300, 3700), #NOT from Laukamp
    '4000W':  (3930, 4150, 3800, 4200),
    '4000WIDEW': (3910, 4150, 3800, 4200),
    '4000V-NARROWW': (3930, 4150, 3800, 4200),
    '4000shortW': (3850, 4000, 3800, 4200),
    '4470TRUEW': (4460, 4490, 4350, 4550),
    '4500SW': (4570, 4850, 4090, 5040),
    '4500CW': (4625, 4770, 4090, 5040),
    '4670W':  (4300, 4800, 4300, 4800),
    '4920W':  (4850, 5100, 4850, 5157),
    
}



feature_keys = list(FEATURE_BOUNDS.keys())

