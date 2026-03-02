"""
Global configuration singleton for CoreSpecViewer.

Import the instance, not the class:
    from ..config import config, feature_keys
"""

from dataclasses import dataclass, fields


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
    savgol_window: int = 10
    savgol_polyorder: int = 2

    # Feature detection
    feature_detection_threshold: float = 0.1

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


# Shared singleton — always import this instance, never instantiate AppConfig directly
config = AppConfig()


feature_keys = [
    '1400W', '1480W', '1550W', '1760W', '1850W',
    '1900W', '2080W', '2160W', '2200W', '2250W',
    '2290W', '2320W', '2350W', '2390W', '2950W',
    '2950AW', '2830W', '3000W', '3500W', '4000W',
    '4000WIDEW', '4470TRUEW', '4500SW', '4500CW',
    '4670W', '4920W', '4000V_NARROWW', '4000shortW', '2950BW'
]


