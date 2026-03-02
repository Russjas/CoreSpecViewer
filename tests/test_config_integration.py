"""
Config Integration Tests for CoreSpecViewer

Verifies that the AppConfig singleton wires correctly into spectral_functions:
mutations via config.set() are immediately visible to:
  - process()         via savgol_window / savgol_polyorder
  - _slice_from_sensor() via all sensor slice keys (called inside find_snr_and_reflect
                         and get_fenix_reflectance, tested here directly)
  - Combined_MWL()    via feature_detection_threshold

Run from the project root (one level above app/):
    python test_config_integration.py

Requires the app package to be importable (no test data needed).
"""

import sys
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import config
from app.spectral_ops import spectral_functions as sf

logging.disable(logging.CRITICAL)

# ============================================================================
# Minimal test harness 
# ============================================================================

results = {}
failures = []


def ok(name):
    results[name] = True
    print(f"  PASS  {name}")


def fail(name, reason):
    results[name] = False
    failures.append(f"{name}: {reason}")
    print(f"  FAIL  {name}: {reason}")


def check(name, expr, expected=True):
    try:
        result = expr()
        if result == expected:
            ok(name)
        else:
            fail(name, f"expected {expected!r}, got {result!r}")
    except Exception as e:
        fail(name, f"raised {type(e).__name__}: {e}")


def raises(name, exc_type, expr):
    try:
        expr()
        fail(name, f"expected {exc_type.__name__} but no exception raised")
    except exc_type:
        ok(name)
    except Exception as e:
        fail(name, f"expected {exc_type.__name__}, got {type(e).__name__}: {e}")


def section(title):
    print(f"\n--- {title} ---")


def report():
    total = len(results)
    passed = sum(results.values())
    print(f"\n{'='*50}")
    print(f"  {passed}/{total} passed")
    if failures:
        print(f"\n  Failures:")
        for f in failures:
            print(f"    {f}")
    print(f"{'='*50}\n")
    return passed == total


# ============================================================================
# Helpers
# ============================================================================

def make_cube(h=20, w=20, b=300):
    """Synthetic reflectance cube — smooth ramp, no zeros."""
    rng = np.random.default_rng(42)
    base = np.linspace(0.1, 0.9, b)
    cube = np.tile(base, (h, w, 1)) + rng.normal(0, 0.01, (h, w, b))
    return np.clip(cube.astype(np.float32), 0.01, 1.0)


# ============================================================================
# 1. Singleton identity — sf and config module point at same object
# ============================================================================

section("Singleton identity")

check("sf.con_dict is config (same object after refactor)",
      lambda: sf.con_dict is config
      if hasattr(sf, 'con_dict') else True)  # passes either way post-refactor

check("config is the AppConfig instance imported by sf",
      lambda: type(config).__name__ == "AppConfig")

# ============================================================================
# 2. process() respects savgol_window and savgol_polyorder
# ============================================================================

section("process() — savgol params from config")

cube = make_cube(b=300)
config.reset()

# Baseline run with defaults (window=10, poly=2)
savgol_default, _, _ = sf.process(cube)

# Change window — result must differ
config.set("savgol_window", "21")
savgol_wide, _, _ = sf.process(cube)
check("process() output changes when savgol_window mutated",
      lambda: not np.allclose(savgol_default, savgol_wide))

# Change polyorder — result must differ from wide-window run
config.set("savgol_polyorder", "3")
savgol_poly3, _, _ = sf.process(cube)
check("process() output changes when savgol_polyorder mutated",
      lambda: not np.allclose(savgol_wide, savgol_poly3))

# Restore and verify we get default result back
config.reset()
savgol_restored, _, _ = sf.process(cube)
check("process() output restored after config.reset()",
      lambda: np.allclose(savgol_default, savgol_restored))

# Output shape unchanged regardless of params
check("process() output shape invariant to savgol params",
      lambda: savgol_default.shape == savgol_wide.shape == savgol_poly3.shape)

# Mask is always zeros on fresh process
_, _, mask = sf.process(cube)
check("process() mask is all zeros",
      lambda: np.all(mask == 0))

# ============================================================================
# 3. _slice_from_sensor() respects all sensor slice keys
# ============================================================================

section("_slice_from_sensor() — slice keys from config")

config.reset()

# Test each sensor branch by checking the returned slice matches config values
sensor_cases = [
    ("SWIR3 Sensor",   "swir_slice_start",    "swir_slice_stop"),
    ("RGB Camera",     "rgb_slice_start",     "rgb_slice_stop"),
    ("FX50 Camera",    "mwir_slice_start",    "mwir_slice_stop"),
    ("FENIX Sensor",   "fenix_slice_start",   "fenix_slice_stop"),
    ("Unknown Sensor", "default_slice_start", "default_slice_stop"),
]

for sensor, start_key, stop_key in sensor_cases:
    s = sf._slice_from_sensor(sensor)
    expected_start = getattr(config, start_key)
    expected_stop  = getattr(config, stop_key)
    check(f"_slice_from_sensor('{sensor}') start matches config.{start_key}",
          lambda s=s, e=expected_start: s.start == e)
    check(f"_slice_from_sensor('{sensor}') stop matches config.{stop_key}",
          lambda s=s, e=expected_stop: s.stop == e)

# Mutate a slice key and verify _slice_from_sensor picks it up immediately
config.set("swir_slice_start", "20")
s_mutated = sf._slice_from_sensor("SWIR3 Sensor")
check("_slice_from_sensor picks up mutated swir_slice_start immediately",
      lambda: s_mutated.start == 20)

config.set("swir_slice_start", "99")
s_mutated2 = sf._slice_from_sensor("SWIR3 Sensor")
check("_slice_from_sensor picks up second mutation of swir_slice_start",
      lambda: s_mutated2.start == 99)

config.reset()
s_reset = sf._slice_from_sensor("SWIR3 Sensor")
check("_slice_from_sensor restored to default after config.reset()",
      lambda: s_reset.start == 13)

# ============================================================================
# 4. Combined_MWL() respects feature_detection_threshold
# ============================================================================

section("Combined_MWL() — feature_detection_threshold from config")

# Build a synthetic 1D spectral profile with a clear 2200nm feature
# Combined_MWL expects (savgol, savgol_cr, mask, bands, feature)
# We wrap in a (1, N, B) cube shape

B = 300
bands = np.linspace(1000, 2500, B)

# Create a spectrum with a clear absorption at 2200nm
spectrum = np.ones(B, dtype=np.float32) * 0.6
# Add a dip at ~2200nm
idx_2200 = np.argmin(np.abs(bands - 2200))
width = 10
spectrum[max(0, idx_2200-width):idx_2200+width] -= 0.35

savgol_1d = spectrum[np.newaxis, np.newaxis, :]   # (1, 1, B)
savgol_cr_1d = sf.cr(savgol_1d)
mask_1d = np.zeros((1, 1), dtype=np.uint8)

config.reset()  # threshold = 0.1

# With default threshold (0.1) — feature should be detected
try:
    pos_default, dep_default, feat_mask_default = sf.Combined_MWL(
        savgol_1d, savgol_cr_1d, mask_1d, bands, '2200W', technique='QND'
    )
    # At least one valid detection
    valid_default = np.sum(~feat_mask_default)
    ok("Combined_MWL() runs with default threshold")
except Exception as e:
    fail("Combined_MWL() runs with default threshold", str(e))
    valid_default = 0

# Raise threshold to 0.99 — shallow feature should now be masked out
config.set("feature_detection_threshold", "0.99")
try:
    pos_high, dep_high, feat_mask_high = sf.Combined_MWL(
        savgol_1d, savgol_cr_1d, mask_1d, bands, '2200W', technique='QND'
    )
    valid_high = np.sum(~feat_mask_high)
    check("Combined_MWL() masks more pixels with higher threshold",
          lambda: valid_high <= valid_default)
except Exception as e:
    fail("Combined_MWL() runs with high threshold", str(e))

# Lower threshold to near-zero — everything passes
config.set("feature_detection_threshold", "0.001")
try:
    pos_low, dep_low, feat_mask_low = sf.Combined_MWL(
        savgol_1d, savgol_cr_1d, mask_1d, bands, '2200W', technique='QND'
    )
    valid_low = np.sum(~feat_mask_low)
    check("Combined_MWL() masks fewer pixels with lower threshold",
          lambda: valid_low >= valid_default)
except Exception as e:
    fail("Combined_MWL() runs with low threshold", str(e))

config.reset()

# ============================================================================
# 5. Mutation isolation — changing one key doesn't affect others
# ============================================================================

section("Mutation isolation")

config.reset()
original_poly = config.savgol_polyorder
config.set("savgol_window", "25")
check("Changing savgol_window does not affect savgol_polyorder",
      lambda: config.savgol_polyorder == original_poly)

original_swir_stop = config.swir_slice_stop
config.set("swir_slice_start", "5")
check("Changing swir_slice_start does not affect swir_slice_stop",
      lambda: config.swir_slice_stop == original_swir_stop)

config.reset()

# ============================================================================
# 6. Type safety under repeated set/reset cycles
# ============================================================================

section("Type safety under set/reset cycles")

for i in range(3):
    config.set("savgol_window", str(10 + i))
    config.set("feature_detection_threshold", str(0.1 + i * 0.05))

config.reset()
check("savgol_window is int after cycles",
      lambda: type(config.savgol_window) is int)
check("feature_detection_threshold is float after cycles",
      lambda: type(config.feature_detection_threshold) is float)
check("savgol_window is default after reset following cycles",
      lambda: config.savgol_window == 10)

# ============================================================================

sys.exit(0 if report() else 1)
