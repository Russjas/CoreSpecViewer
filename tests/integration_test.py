"""
CoreSpecViewer Integration Test Script

This script performs a full end-to-end workflow test:
1. Load raw Lumo directory
2. Crop to specified rectangle
3. Process to reflectance
4. Run k-means clustering
5. Extract multiple spectral features
6. Apply band math expressions
7. Save processed object
8. Load spectral library
9. Perform correlation analyses (Pearson, SAM, MSAM)
10. Save results

Usage:
    python integration_test.py --raw-dir <path> --output-dir <path> --library-db <path>
    
    Or edit the paths directly in the script and run:
    python integration_test.py
"""

import sys
import logging
from pathlib import Path
import re
import json
from datetime import datetime, timedelta
import argparse

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


from app.models import RawObject, ProcessedObject, CurrentContext, HoleObject
from app.models.lib_manager import LibraryManager, ID_COLUMN_INDEX
from app.interface import tools as t
from app.interface import profile_tools as pt
from app.spectral_ops import spectral_functions as sf
from app.spectral_ops import band_maths as bm

logging.getLogger("spectral").setLevel(logging.CRITICAL)
logging.getLogger("app").setLevel(logging.CRITICAL)
logging.getLogger("numpy").setLevel(logging.CRITICAL)
logging.getLogger("numba").setLevel(logging.CRITICAL)

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"

RAW_LUMO_DIR = TEST_DATA_DIR / "Synthetic_test_box"
BASELINE_DIR = TEST_DATA_DIR / "baselines"
OUTPUT_DIR = TEST_DATA_DIR / "test_outputs"
LIBRARY_DB_PATH = TEST_DATA_DIR / "TestDB.db"

log_dir = TEST_DIR
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'integration_test.log'),
        logging.StreamHandler()  # Still prints to console
    ]
)
logger = logging.getLogger(__name__)


# Crop rectangle (y_min, y_max, x_min, x_max)
CROP_RECT = (2, 98, 8, 96)  # Adjust to your data

# Masking test pixels (y, x) - choose pixels representing background/unwanted material
MASK_PIXEL_NEW = (5, 54)  # Pixel for creating new mask
MASK_PIXEL_ENHANCE = (93, 57)  # Pixel for enhancing mask

# K-means parameters
KMEANS_CLUSTERS = 5
KMEANS_ITERATIONS = 100

# Features to extract (subset of available features)
FEATURES_TO_EXTRACT = [
    '1400W', '1900W', '2200W', '2320W', '2350W'
]

# Band math expressions to test
BAND_MATH_EXPRESSIONS = [
    {
        'name': 'test_ratio_2200_2320',
        'expression': 'R2200 / R2320',
        'continuum_removed': False
    },
    {
        'name': 'test_difference_cr',
        'expression': 'R2200 - R2320',
        'continuum_removed': True
    }
]

# Correlation parameters
CORRELATION_COLLECTION = "test_collection"  # Name of collection in library to use

#integer products
EXACT_KEYS = {"mask", "segments",}
#float products
CLOSE_KEYS = {"cropped", "savgol", "savgol_cr",    "bands"}
JSON_KEYS = {"metadata"}
# Regexes for derived products
FEATURE_RE = re.compile(r"^\d{3,5}W(POS|DEP)$")        # e.g., 2200WPOS, 2200WDEP
MINMAP_RE  = re.compile(r"^MinMap-.*(INDEX|CONF)$")    # legends are json
KMEANS_RE  = re.compile(r"^kmeans-\d+-\d+(INDEX|CLUSTERS)$")
# Tolerances
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _non_numeric_key(key):
    return (key == "metadata"
        or key.endswith("LEGEND")
        )


def _assert_allclose(name, a, b, rtol, atol):
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    
    # Handle masked arrays - compare mask and data separately
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        # Ensure both are masked arrays for consistent comparison
        a_masked = np.ma.asarray(a)
        b_masked = np.ma.asarray(b)
        
        # Compare masks first (must be identical)
        mask_a = np.ma.getmaskarray(a_masked)
        mask_b = np.ma.getmaskarray(b_masked)
        if not np.array_equal(mask_a, mask_b):
            n_diff = np.sum(mask_a != mask_b)
            raise AssertionError(f"{name}: masks differ ({n_diff} pixels masked differently)")
        
        # Compare data (only where unmasked)
        data_a = a_masked.data
        data_b = b_masked.data
        valid = ~mask_a  # Where data is valid in both
        
        if np.any(valid):
            if not np.allclose(data_a[valid], data_b[valid], rtol=rtol, atol=atol, equal_nan=True):
                diff = np.nanmax(np.abs(data_a[valid] - data_b[valid]))
                raise AssertionError(f"{name}: data values differ (max abs diff {diff})")
        
    else:
        # Regular arrays
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            diff = np.nanmax(np.abs(a - b))
            raise AssertionError(f"{name}: allclose failed (max abs diff {diff})")
    
    logger.debug(f"{name} passed baseline comparison")





def _assert_equal(name, a, b):
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    if not np.array_equal(a, b):
        n = np.sum(a != b)
        raise AssertionError(f"{name}: array_equal failed ({n} differing elements)")
    logger.debug(f"{name} passed baseline comparison")
    
def _compare_one_key(key, base_entry, act_entry, rtol, atol):
    
    if _non_numeric_key(key):
        if base_entry != act_entry:
            logger.error(f"{key}: json/list differs")
            raise AssertionError(f"{key}: json/list differs")
        return
    elif key in EXACT_KEYS:
        _assert_equal(key, base_entry, act_entry)

    elif key in CLOSE_KEYS or FEATURE_RE.match(key) or MINMAP_RE.match(key):
        _assert_allclose(key, base_entry, act_entry, rtol=rtol, atol=atol)

    elif KMEANS_RE.match(key):
        # KMEANS is often nondeterministic + labels can permute.
        # Default strategy here:
        #   - CLUSTERS: compare centroid spectra after sorting by their mean (handles permutation)
        #   - INDEX: compare histogram of cluster sizes (handles label permutation)
        if key.endswith("CLUSTERS"):
            a2 = np.asarray(base_entry)
            b2 = np.asarray(act_entry)
            if a2.shape != b2.shape:
                logger.error(f"{key}: shape mismatch {a2.shape} vs {b2.shape}")
                raise AssertionError(f"{key}: shape mismatch {a2.shape} vs {b2.shape}")
            a_sort = a2[np.argsort(np.nanmean(a2, axis=1))]
            b_sort = b2[np.argsort(np.nanmean(b2, axis=1))]
            _assert_allclose(key, a_sort, b_sort, rtol=max(rtol, 1e-3), atol=max(atol, 1e-5))
        else:
            a_lab = base_entry[base_entry >= 0].ravel().astype(int)
            b_lab = act_entry[act_entry >= 0].ravel().astype(int)
            ha = np.bincount(a_lab) if a_lab.size else np.array([], dtype=int)
            hb = np.bincount(b_lab) if b_lab.size else np.array([], dtype=int)
            if ha.shape != hb.shape:
                logger.error(f"{key}: cluster histogram shape mismatch {ha.shape} vs {hb.shape}")
                raise AssertionError(f"{key}: cluster histogram shape mismatch {ha.shape} vs {hb.shape}")
            if not np.array_equal(ha, hb):
                logger.error(f"{key}: cluster-size histogram differs")
                raise AssertionError(f"{key}: cluster-size histogram differs")
            logger.debug(f"{key} passed baseline comparison")
    
    elif key in JSON_KEYS:
        if base_entry != act_entry:
            logger.error(f"{key}: json differs")
            raise AssertionError(f"{key}: json differs")
        logger.debug(f"{key} passed baseline comparison")
    else:
        # Conservative default for unknown arrays
        _assert_allclose(key, base_entry, act_entry, rtol=rtol, atol=atol)
        logger.debug(f"{key} passed baseline comparison")



def validate_paths():
    """Validate that required paths exist."""
    if not RAW_LUMO_DIR.exists():
        logger.error(f"Raw directory does not exist: {RAW_LUMO_DIR}")
        return False
    
    if not OUTPUT_DIR.exists():
        logger.debug(f"Creating output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    if not BASELINE_DIR.exists():
        logger.debug(f"Creating baseline directory: {BASELINE_DIR}")
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        
    if not LIBRARY_DB_PATH.exists():
        logger.warning(f"Library database not found: {LIBRARY_DB_PATH}")
        logger.warning("Correlation tests will be skipped")
    
    return True


def print_section(title):
    """Print a formatted section header."""
    logger.debug("")
    logger.debug("=" * 80)
    logger.debug(f"  {title}")
    logger.debug("=" * 80)


def print_object_info(obj, obj_type="Object"):
    """Print information about a data object."""
    logger.debug(f"{obj_type} Information:")
    logger.debug(f"  Basename: {obj.basename}")
    logger.debug(f"  Root dir: {obj.root_dir}")
    
    if hasattr(obj, 'metadata') and obj.metadata:
        logger.debug(f"  Metadata keys: {list(obj.metadata.keys())[:10]}...")  # First 10 keys
    
    if hasattr(obj, 'datasets'):
        logger.debug(f"  Datasets: {list(obj.datasets.keys())}")
    
    if hasattr(obj, 'temp_datasets') and obj.temp_datasets:
        logger.debug(f"  Temp datasets: {list(obj.temp_datasets.keys())}")


def print_mask_stats(mask, label="Mask"):
    """Print statistics about a mask array."""
    if mask is None:
        logger.warning(f"{label}: None")
        return
    
    total_pixels = mask.size
    masked_pixels = np.sum(mask == 1)
    valid_pixels = np.sum(mask == 0)
    
    logger.debug(f"{label} Statistics:")
    logger.debug(f"  Shape: {mask.shape}")
    logger.debug(f"  Total pixels: {total_pixels}")
    logger.debug(f"  Masked pixels: {masked_pixels} ({masked_pixels/total_pixels*100:.1f}%)")
    logger.debug(f"  Valid pixels: {valid_pixels} ({valid_pixels/total_pixels*100:.1f}%)")


# ============================================================================
# PART 1: RAW LOADING AND PROCESSING
# ============================================================================

def test_raw_loading():
    """Test loading raw Lumo directory."""
    print_section("PART 1A: Loading Raw Lumo Directory")
    
    try:
        ro = RawObject.from_Lumo_directory(RAW_LUMO_DIR)
        logger.debug("PASS Successfully loaded raw object")
        print_object_info(ro, "RawObject")
        
        # Check reflectance shape
        if hasattr(ro, 'reflectance') and ro.reflectance is not None:
            logger.debug(f"  Reflectance shape: {ro.reflectance.shape}")
        else:
            logger.error("FAIL Reflectance not loaded")
            return None
        
        return ro
    
    except Exception as e:
        logger.error(f"FAIL Failed to load raw directory: {e}", exc_info=True)
        return None


def test_crop(ro):
    """Test cropping raw object."""
    print_section("PART 1B: Cropping Raw Data")
    
    try:
        y_min, y_max, x_min, x_max = CROP_RECT
        logger.debug(f"Applying crop: y[{y_min}:{y_max}], x[{x_min}:{x_max}]")
        
        ro_cropped = t.crop(ro, y_min, y_max, x_min, x_max)
        
        if hasattr(ro_cropped, 'temp_reflectance') and ro_cropped.temp_reflectance is not None:
            logger.debug(f"PASS Cropped shape: {ro_cropped.temp_reflectance.shape}")
            return ro_cropped
        else:
            logger.error("FAIL Crop did not create temp_reflectance")
            return None
    
    except Exception as e:
        logger.error(f"FAIL Cropping failed: {e}", exc_info=True)
        return None


def test_process_to_reflectance(ro, create_baseline = False):
    """Test processing raw to ProcessedObject."""
    print_section("PART 1C: Processing to Reflectance")
    
    try:
        logger.debug(f"Processing raw data...")
        
        # Convert to ProcessedObject (creates in raw directory initially)
        po = ro.process()
        
        logger.debug("PASS Successfully created ProcessedObject")
        print_object_info(po, "ProcessedObject")
        
        # Update root directory to output location before saving
        # update_root_dir reconstructs paths as: output_dir / f"{basename}_{key}{ext}"
        logger.debug(f"Updating output directory to: {OUTPUT_DIR}")
        if create_baseline:
            po.update_root_dir(BASELINE_DIR)
        else:
            po.update_root_dir(OUTPUT_DIR)
        
        # Verify base datasets exist
        required_datasets = ['cropped', 'savgol', 'savgol_cr', 'bands', 'metadata', 'mask']
        missing = [ds for ds in required_datasets if not po.has(ds)]
        
        if missing:
            logger.warning(f" Missing expected datasets: {missing}")
        else:
            logger.debug(f"PASS All base datasets present: {required_datasets}")
        
        return po
    
    except Exception as e:
        logger.error(f"FAIL Processing failed: {e}", exc_info=True)
        return None
# ============================================================================
# PART 2: MASKING OPERATIONS
# ============================================================================

def test_create_new_mask(po):
    """Test creating a new mask using point-based correlation."""
    print_section("PART 2A: Create New Mask")
    
    try:
        y, x = MASK_PIXEL_NEW
        logger.debug(f"Creating new mask using pixel ({y}, {x})")
        logger.debug(f"Correlation threshold: 0.9 (pixels with corr > 0.9 will be masked)")
        
        # Store initial mask for comparison
        initial_mask = np.array(po.mask)
        print_mask_stats(initial_mask, "Initial mask")
        
        # Create new mask
        po = t.mask_point(po, mode='new', y=y, x=x)
        
        if po.has_temp('mask'):
            new_mask = po.temp_datasets['mask'].data
            logger.debug("PASS New mask created")
            print_mask_stats(new_mask, "New mask")
            
            changed_pixels = np.sum(new_mask != initial_mask)
            logger.debug(f"Changed pixels from initial: {changed_pixels}")
            return True
        else:
            logger.error("FAIL Mask not found in temp_datasets")
            return False
    
    except Exception as e:
        logger.error(f"FAIL Failed to create new mask: {e}", exc_info=True)
        return False


def test_enhance_mask(po):
    """Test enhancing existing mask with additional pixel."""
    print_section("PART 2B: Enhance Mask")
    
    try:
        y, x = MASK_PIXEL_ENHANCE
        logger.debug(f"Enhancing mask using pixel ({y}, {x})")
        
        # Get current mask
        current_mask = po.temp_datasets['mask'].data if po.has_temp('mask') else po.mask
        current_masked = np.sum(current_mask == 1)
        
        # Enhance mask
        po = t.mask_point(po, mode='enhance', y=y, x=x)
        
        if po.has_temp('mask'):
            enhanced_mask = po.temp_datasets['mask'].data
            new_masked = np.sum(enhanced_mask == 1)
            additional_masked = new_masked - current_masked
            
            logger.debug("PASS Mask enhanced")
            logger.debug(f"Additional pixels masked: {additional_masked}")
            print_mask_stats(enhanced_mask, "Enhanced mask")
            return True
        else:
            logger.error("FAIL Enhanced mask not found")
            return False
    
    except Exception as e:
        logger.error(f"FAIL Failed to enhance mask: {e}", exc_info=True)
        return False


def test_despeckle_mask(po):
    """Test despeckle operation on mask."""
    print_section("PART 2C: Despeckle Mask")
    
    try:
        current_mask = po.temp_datasets['mask'].data if po.has_temp('mask') else po.mask
        logger.debug("Running despeckle operation to remove isolated pixels")
        
        po = t.despeckle_mask(po)
        
        if po.has_temp('mask'):
            despeckled_mask = po.temp_datasets['mask'].data
            changes = np.sum(despeckled_mask != current_mask)
            
            logger.debug("PASS Mask despeckled")
            logger.debug(f"Pixels changed: {changes}")
            return True
        else:
            logger.error("FAIL Despeckled mask not found")
            return False
    
    except Exception as e:
        logger.error(f"FAIL Failed to despeckle mask: {e}", exc_info=True)
        return False


def test_improve_mask(po):
    """Test heuristic mask improvement."""
    print_section("PART 2D: Improve Mask")
    
    try:
        current_mask = po.temp_datasets['mask'].data if po.has_temp('mask') else po.mask
        logger.debug("Running heuristic mask improvement")
        
        po = t.improve_mask(po)
        
        if po.has_temp('mask'):
            improved_mask = po.temp_datasets['mask'].data
            changes = np.sum(improved_mask != current_mask)
            
            logger.debug("PASS Mask improved")
            logger.debug(f"Pixels changed: {changes}")
            print_mask_stats(improved_mask, "Improved mask")
            return True
        else:
            logger.error("FAIL Improved mask not found")
            return False
    
    except Exception as e:
        logger.error(f"FAIL Failed to improve mask: {e}", exc_info=True)
        return False


def test_calc_unwrap_stats(po):
    """Test calculating connected components for unwrapping."""
    print_section("PART 2E: Calculate Unwrap Stats")
    
    try:
        logger.debug("Calculating connected components for downhole unwrapping")
        
        po = t.calc_unwrap_stats(po)
        
        has_stats = po.has_temp('stats')
        has_segments = po.has_temp('segments')
        
        if has_stats and has_segments:
            stats = po.temp_datasets['stats'].data
            segments = po.temp_datasets['segments'].data
            
            logger.debug("PASS Unwrap stats calculated")
            logger.debug(f"  Segments shape: {segments.shape}")
            logger.debug(f"  Segments range: [{np.min(segments):.3f}, {np.max(segments):.3f}]")
            
            if hasattr(stats, 'shape'):
                logger.debug(f"  Number of segments: {len(stats)}")
            
            return True
        else:
            logger.error(f"FAIL Missing datasets - stats: {has_stats}, segments: {has_segments}")
            return False
    
    except Exception as e:
        logger.error(f"FAIL Failed to calculate unwrap stats: {e}", exc_info=True)
        return False


def test_unwrap_preview(po):
    """Test generating unwrapped preview."""
    print_section("PART 2F: Generate Unwrap Preview")
    
    # Check metadata requirements
    required_meta = ['core depth start', 'core depth stop']
    missing_meta = [k for k in required_meta if k not in po.metadata]
    
    if missing_meta:
        logger.warning(f" Cannot test unwrap - missing metadata: {missing_meta}")
        return False
    
    if not po.has_temp('stats'):
        logger.warning(" Cannot test unwrap - no stats calculated")
        return False
    
    try:
        depth_start = po.metadata['core depth start']
        depth_stop = po.metadata['core depth stop']
        logger.debug(f"Generating unwrapped preview (depth {depth_start} to {depth_stop})")
        
        po = t.unwrapped_output(po)
        
        has_average = po.has_temp('DholeAverage')
        has_mask = po.has_temp('DholeMask')
        has_depths = po.has_temp('DholeDepths')
        
        if has_average and has_mask and has_depths:
            dh_avg = po.temp_datasets['DholeAverage'].data
            dh_mask = po.temp_datasets['DholeMask'].data
            dh_depths = po.temp_datasets['DholeDepths'].data
            
            logger.debug("PASS Unwrapped preview generated")
            logger.debug(f"  DholeAverage shape: {dh_avg.shape}")
            logger.debug(f"  Depth range: [{dh_depths[0]:.2f}, {dh_depths[-1]:.2f}]")
            
            return True
        else:
            logger.error(f"FAIL Missing unwrap datasets")
            return False
    
    except Exception as e:
        logger.error(f"FAIL Failed to generate unwrap: {e}", exc_info=True)
        return False


# ============================================================================
# PART 3: ANALYSIS OPERATIONS
# ============================================================================

def test_kmeans(po):
    """Test k-means clustering."""
    print_section("PART 3A: K-means Clustering")
    
    try:
        logger.debug(f"Running k-means: {KMEANS_CLUSTERS} clusters, {KMEANS_ITERATIONS} iterations")
        
        # Run k-means using tools interface
        po = t.kmeans_caller(po, KMEANS_CLUSTERS, KMEANS_ITERATIONS)
        
        # Check if datasets were created
        key_prefix = f'kmeans-{KMEANS_CLUSTERS}-{KMEANS_ITERATIONS}'
        has_index = po.has_temp(f'{key_prefix}INDEX')
        has_clusters = po.has_temp(f'{key_prefix}CLUSTERS')
        
        if has_index and has_clusters:
            cluster_map = po.temp_datasets[f'{key_prefix}INDEX'].data
            centroids = po.temp_datasets[f'{key_prefix}CLUSTERS'].data
            
            logger.debug(f"PASS Clustering complete")
            logger.debug(f"  Cluster map shape: {cluster_map.shape}")
            logger.debug(f"  Centroids shape: {centroids.shape}")
            logger.debug(f"  Unique clusters: {np.unique(cluster_map[cluster_map >= 0])}")
            
            return True
        else:
            logger.error("FAIL K-means datasets not created")
            return False
    
    except Exception as e:
        logger.error(f"FAIL K-means clustering failed: {e}", exc_info=True)
        return False


def test_feature_extraction(po):
    """Test feature extraction."""
    print_section("PART 3B: Feature Extraction")
    
    success_count = 0
    
    for feature_key in FEATURES_TO_EXTRACT:
        try:
            logger.debug(f"Extracting feature: {feature_key}")
            
            # Run feature extraction using tools interface
            po = t.run_feature_extraction(po, feature_key)
            
            # Check if datasets were created
            has_pos = po.has_temp(f'{feature_key}POS')
            has_dep = po.has_temp(f'{feature_key}DEP')
            
            if has_pos and has_dep:
                position = po.temp_datasets[f'{feature_key}POS'].data
                depth = po.temp_datasets[f'{feature_key}DEP'].data
                
                logger.debug(f"  PASS {feature_key}: position shape {position.shape}")
                success_count += 1
            else:
                logger.warning(f"   {feature_key}: datasets not created")
        
        except Exception as e:
            logger.error(f"  FAIL {feature_key} failed: {e}", exc_info=True)
    
    logger.debug(f"\nFeature extraction summary: {success_count}/{len(FEATURES_TO_EXTRACT)} successful")
    return success_count > 0


def test_band_maths(po):
    """Test band math expressions."""
    print_section("PART 3C: Band Math Expressions")
    
    success_count = 0
    
    for expr_config in BAND_MATH_EXPRESSIONS:
        try:
            name = expr_config['name']
            expression = expr_config['expression']
            use_cr = expr_config['continuum_removed']
            
            logger.debug(f"Evaluating: {name}")
            logger.debug(f"  Expression: {expression}")
            logger.debug(f"  Continuum removed: {use_cr}")
            
            # Use the tools interface which handles the details
            po = t.band_math_interface(po, name, expression, cr=use_cr)
            
            # band_math_interface sanitizes the name (replaces _/:*?"<>| with -)
            clean_key = re.sub(r'[\\/:*?"<>|_]', '-', name)
            
            # Check if dataset was created
            if po.has_temp(clean_key):
                result = po.temp_datasets[clean_key].data
                logger.debug(f"  PASS Result shape: {result.shape}")
                if hasattr(result, 'data'):  # Masked array
                    logger.debug(f"  PASS Value range: [{np.nanmin(result.data):.3f}, {np.nanmax(result.data):.3f}]")
                else:
                    logger.debug(f"  PASS Value range: [{np.nanmin(result):.3f}, {np.nanmax(result):.3f}]")
                success_count += 1
            else:
                logger.warning(f"   Dataset '{clean_key}' not created")
        
        except Exception as e:
            logger.error(f"  FAIL Band math failed: {e}", exc_info=True)
    
    logger.debug(f"\nBand math summary: {success_count}/{len(BAND_MATH_EXPRESSIONS)} successful")
    return success_count > 0


# ============================================================================
# PART 4: SAVING
# ============================================================================

def test_save(po):
    """Test saving ProcessedObject."""
    print_section("PART 4: Saving ProcessedObject")
    
    try:
        logger.debug(f"Committing temporary datasets...")
        po.commit_temps()
        logger.debug(f"PASS Committed {len(po.datasets)} datasets")
        
        logger.debug(f"Saving to disk...")
        po.save_all()
        logger.debug(f"PASS Saved to: {po.root_dir}")
        
        # Verify files exist
        saved_files = list(po.root_dir.glob(f"{po.basename}_*"))
        logger.debug(f"PASS Files on disk: {len(saved_files)}")
        for f in saved_files[:10]:  # Show first 10
            logger.debug(f"    {f.name}")
        if len(saved_files) > 10:
            logger.debug(f"    ... and {len(saved_files) - 10} more")
        
        return True
    
    except Exception as e:
        logger.error(f"FAIL Saving failed: {e}", exc_info=True)
        return False


# ============================================================================
# PART 5: LIBRARY AND CORRELATION
# ============================================================================

def test_library_loading(): 
    """Test loading spectral library and creating a test collection."""
    print_section("PART 5A: Loading Spectral Library & Creating Collection")
    
    if not LIBRARY_DB_PATH.exists():
        logger.warning(" Library database not found, skipping correlation tests")
        return None
    
    try:
        lib_manager = LibraryManager()
        lib_manager.open_database(str(LIBRARY_DB_PATH))
        
        if not lib_manager.is_open():
            logger.error("FAIL Library failed to open")
            return None
        
        logger.debug("PASS Library loaded successfully")
        
        # Get all available sample IDs from the database
        if lib_manager.model is None:
            logger.error("FAIL No model available")
            return None
        
        # Get sample IDs from the model
        sample_ids = []
        row_count = lib_manager.model.rowCount()
        logger.debug(f"  Total samples in database: {row_count}")
        
        if row_count == 0:
            logger.warning(" Database has no samples")
            return None
        
        # Collect sample IDs (first 10 or all if less)
        max_samples = min(10, row_count)
        for row in range(max_samples):
            record = lib_manager.model.record(row)
            sample_id = record.value(0)  # ID_COLUMN_INDEX = 0
            if sample_id is not None:
                sample_ids.append(int(sample_id))
        
        logger.debug(f"  Collected {len(sample_ids)} sample IDs for test collection")
        
        # Create a test collection (collections are in-memory, not persisted)
        collection_name = CORRELATION_COLLECTION
        added, total = lib_manager.add_to_collection(collection_name, sample_ids)
        
        logger.debug(f"PASS Created collection '{collection_name}'")
        logger.debug(f"  Added: {added} samples")
        logger.debug(f"  Total: {total} samples")
        
        # Verify collection exists
        collections = lib_manager.list_collections()
        logger.debug(f"  Active collections: {collections}")
        
        # Get collection exemplars (this fetches the actual spectra)
        collection_data = lib_manager.get_collection_exemplars(collection_name)
        logger.debug(f"PASS Collection exemplars built: {len(collection_data)} spectra")
        
        # Show some info about exemplars
        if collection_data:
            first_id = list(collection_data.keys())[0]
            label, x_nm, y = collection_data[first_id]
            logger.debug(f"  Example spectrum: '{label}'")
            logger.debug(f"    Wavelength range: {x_nm[0]:.1f} - {x_nm[-1]:.1f} nm")
            logger.debug(f"    Reflectance range: {y.min():.3f} - {y.max():.3f}")
        
        return lib_manager
    
    except Exception as e:
        logger.error(f"FAIL Library loading failed: {e}", exc_info=True)
        return None
    
    
    
def test_reload_processed(po_path):
    """Test reloading the saved ProcessedObject."""
    print_section("PART 5B: Reloading Saved ProcessedObject")
    
    try:
        # Find a file from the saved PO to reload from
        saved_files = list(po_path.glob("*.npy"))
        if not saved_files:
            saved_files = list(po_path.glob("*.json"))
        
        if not saved_files:
            logger.error(f"FAIL No files found in {po_path}")
            return None
        
        # Use first file to build PO
        po = ProcessedObject.from_path(saved_files[0])
        logger.debug("PASS Successfully reloaded ProcessedObject")
        print_object_info(po, "Reloaded ProcessedObject")
        
        # Reload all datasets from disk
        po.reload_all()
        logger.debug("PASS Reloaded all datasets from disk")
        
        return po
    
    except Exception as e:
        logger.error(f"FAIL Failed to reload ProcessedObject: {e}", exc_info=True)
        return None


def test_correlation(po, lib_manager):
    """Test correlation analyses."""
    print_section("PART 5C: Correlation Analysis")
    
    if lib_manager is None:
        logger.warning(" Skipping correlation tests (no library)")
        return False
    
    # Get collection exemplars
    try:
        collection_name = CORRELATION_COLLECTION
        exemplars = lib_manager.get_collection_exemplars(collection_name)
        logger.debug(f"Using collection '{collection_name}' with {len(exemplars)} spectra")
        
        if not exemplars:
            logger.error(f"FAIL Collection '{collection_name}' is empty")
            return False
    except Exception as e:
        logger.error(f"FAIL Failed to get collection exemplars: {e}", exc_info=True)
        return False
    
    correlation_methods = [
        ('pearson', t.wta_min_map),
        ('SAM', t.wta_min_map_SAM),
        ('MSAM', t.wta_min_map_MSAM)
    ]
    
    success_count = 0
    
    for method_name, method_func in correlation_methods:
        try:
            logger.debug(f"\nRunning {method_name} correlation...")
            
            # Run correlation - these functions modify po and return it
            po = method_func(po, exemplars, collection_name)
            
            # Check if datasets were created
            key_prefix = f"MinMap-{method_name}-{collection_name.replace('_', '')}"
            has_index = po.has_temp(f"{key_prefix}INDEX")
            has_legend = po.has_temp(f"{key_prefix}LEGEND")
            has_conf = po.has_temp(f"{key_prefix}CONF")
            
            if has_index and has_legend and has_conf:
                index = po.temp_datasets[f"{key_prefix}INDEX"].data
                confidence = po.temp_datasets[f"{key_prefix}CONF"].data
                
                logger.debug(f"  PASS {method_name}: created datasets")
                logger.debug(f"    Index shape: {index.shape}")
                logger.debug(f"    Unique classes: {len(np.unique(index[index >= 0]))}")
                logger.debug(f"    Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
                success_count += 1
            else:
                logger.warning(f"   {method_name}: missing datasets")
        
        except Exception as e:
            logger.error(f"  FAIL {method_name} failed: {e}", exc_info=True)
    
    logger.debug(f"\nCorrelation summary: {success_count}/{len(correlation_methods)} successful")
    
    # Save correlation results
    if success_count > 0:
        try:
            logger.debug("\nSaving correlation results...")
            po.commit_temps()
            po.save_all()
            logger.debug("PASS Correlation results saved")
        except Exception as e:
            logger.error(f"FAIL Failed to save correlation results: {e}", exc_info=True)
    
    return success_count > 0


# =============================================================================
# HOLE LEVEL TESTS
# =============================================================================

def create_three_box_hole_dataset(
    po,
    create_baseline = False
):
    
    hole_id = "TEST_HOLE_001"
    depth_per_box = 1.0
    starting_depth = 0.0
    logger.debug("CREATE THREE-BOX HOLE DATASET")
    logger.debug(f"Hole ID: {hole_id}")
    logger.debug(f"Depth per box: {depth_per_box}m")
    logger.debug(f"Starting depth: {starting_depth}m")
    logger.debug(f"Create baseline: {create_baseline}")
    
    # ========================================================================
    # STEP 1: Verify template has required datasets
    # ========================================================================
    logger.debug("STEP 1: Verifying Template Box")
    
    if not po:
        raise RuntimeError("No ProcessedObject provided")
    
    logger.debug(f"Template has {len(po.datasets)} datasets")
    
    # Verify critical datasets exist
    required = ['cropped', 'savgol', 'mask', 'bands', 'stats', 'segments', 'metadata']
    missing = [ds for ds in required if ds not in po.datasets and ds not in po.temp_datasets]
    if missing:
        logger.error(f"Template missing required datasets: {missing}")
        raise RuntimeError(f"Template missing required datasets: {missing}")
    
    logger.debug("PASS Template box has all required datasets")
    
    # ========================================================================
    # STEP 2: Create three-box hole directory structure
    # ========================================================================
    print_section("STEP 2: Creating Three-Box Hole Structure")
    
    base_dir = BASELINE_DIR if create_baseline else OUTPUT_DIR
    hole_dir = base_dir / hole_id
    hole_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Hole directory: {hole_dir}")
    
    box_configs = [
        {
            'box_num': 1,
            'depth_start': starting_depth,
            'depth_stop': starting_depth + depth_per_box
        },
        {
            'box_num': 2,
            'depth_start': starting_depth + depth_per_box,
            'depth_stop': starting_depth + 2 * depth_per_box
        },
        {
            'box_num': 3,
            'depth_start': starting_depth + 2 * depth_per_box,
            'depth_stop': starting_depth + 3 * depth_per_box
        }
    ]
    
    # Get base timestamp
    base_time = datetime.now()
    metadata = po.temp_datasets.get('metadata', po.datasets.get('metadata'))
    if metadata and metadata.data:
        if 'date' in metadata.data and 'time' in metadata.data:
            try:
                base_time = datetime.strptime(
                    f"{metadata.data['date']} {metadata.data['time']}", 
                    "%Y-%m-%d %H:%M:%S"
                )
            except:
                pass
    
    # ========================================================================
    # STEP 3: Create and save each box
    # ========================================================================
    logger.debug("STEP 3: Creating Individual Boxes")
    
    created_boxes = []
    
    for idx, config in enumerate(box_configs):
        box_num = config['box_num']
        logger.debug(f"\nCreating Box {box_num}...")
        
        try:
            new_basename = f"{hole_id}_{box_num}"
            box_po = ProcessedObject.new(hole_dir, new_basename)
            
            # Get source metadata
            source_meta = po.temp_datasets.get('metadata', po.datasets.get('metadata'))
            new_metadata = source_meta.data.copy() if source_meta else {}
            
            new_metadata['borehole id'] = hole_id
            new_metadata['box number'] = box_num
            new_metadata['core depth start'] = config['depth_start']
            new_metadata['core depth stop'] = config['depth_stop']
            
            scan_time = base_time + timedelta(hours=idx)
            new_metadata['date'] = scan_time.strftime("%Y-%m-%d")
            new_metadata['time'] = scan_time.strftime("%H:%M:%S")
            
            logger.debug(f"  Copying {len(po.datasets) + len(po.temp_datasets)} datasets...")
            
            # Copy from both datasets and temp_datasets
            all_datasets = {**po.datasets, **po.temp_datasets}
            
            for key, dataset in all_datasets.items():
                if key == 'metadata':
                    box_po.add_dataset(key, new_metadata, ext='.json')
                else:
                    data_copy = dataset.data.copy()
                    box_po.add_dataset(key, data_copy, ext=dataset.ext)
            
            logger.debug(f"  Saving to disk...")
            box_po.save_all()
            
            saved_files = list(hole_dir.glob(f"{new_basename}_*"))
            logger.debug(f"  PASS Box {box_num} created: {len(saved_files)} files")
            logger.debug(f"    Depth range: {config['depth_start']:.2f} - {config['depth_stop']:.2f}m")
            
            created_boxes.append(new_basename)
            
        except Exception as e:
            logger.error(f"  FAIL Box {box_num} creation failed: {e}", exc_info=True)
            raise RuntimeError(f"Box {box_num} creation failed: {e}")
    
    # ========================================================================
    # STEP 4: Load as HoleObject and verify
    # ========================================================================
    logger.debug("STEP 4: Loading and Verifying HoleObject")
    
    try:
        logger.debug(f"Loading hole from directory: {hole_dir}")
        hole = HoleObject.build_from_parent_dir(hole_dir, hole_id)
        
        logger.debug(f"PASS HoleObject loaded successfully")
        logger.debug(f"  Hole ID: {hole.hole_id}")
        logger.debug(f"  Number of boxes: {hole.num_box}")
        logger.debug(f"  Box range: {hole.first_box} to {hole.last_box}")
        logger.debug(f"  Root directory: {hole.root_dir}")
        
        for box_num, box_po in hole.iter_items():
            logger.debug(f"\n  Box {box_num}:")
            logger.debug(f"    Basename: {box_po.basename}")
            logger.debug(f"    Datasets: {len(box_po.datasets)}")
            logger.debug(f"    Depth: {box_po.metadata['core depth start']:.2f} - "
                       f"{box_po.metadata['core depth stop']:.2f}m")
            
            required_check = ['cropped', 'savgol', 'mask', 'bands', 'stats', 'segments']
            missing_check = [ds for ds in required_check if ds not in box_po.datasets]
            if missing_check:
                logger.warning(f"    Missing datasets: {missing_check}")
            else:
                logger.debug(f"    All required datasets present")
        
        if not hole.check_for_all_keys('stats'):
            logger.error("FAIL Not all boxes have 'stats' dataset")
            raise RuntimeError("Missing stats in one or more boxes")
        
        logger.debug("\nPASS HoleObject verification complete")
        logger.debug(f"Hole dataset ready for downhole analysis at: {hole_dir}")
        
        return hole, True
        
    except Exception as e:
        logger.error(f"FAIL HoleObject loading/verification failed: {e}", exc_info=True)
        raise RuntimeError(f"HoleObject loading failed: {e}")


def test_hole_iteration(hole):
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    try:
        logger.debug("Testing hole iteration...")
        count = 0
        for po in hole:
            count += 1
            logger.debug(f"  Box {po.metadata['box number']}: {po.basename}")
        if count == 3:
            logger.debug(f"PASS Iterated over {count} boxes")
        else:
            logger.error(f"FAIL Expected 3 boxes, got {count}")
    except Exception as e:
        logger.error(f"FAIL Iteration failed: {e}", exc_info=True)
    return True
    
def test_hole_indexing(hole):
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    try:
        logger.debug("Testing hole indexing...")
        box_1 = hole[1]
        box_2_3 = hole[2:4]
        logger.debug(f"  Single box access: {box_1.basename}")
        logger.debug(f"  Slice access: {len(box_2_3)} boxes")
        logger.debug("PASS Indexing works")
    except Exception as e:
        logger.error(f"FAIL Indexing failed: {e}", exc_info=True)
        return False
    return True

def test_box_by_box(hole, lib_manager):
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    success_count = 0
    fail_count = 0
    for po in hole:
        try:
            test_band_maths(po)
            test_correlation(po, lib_manager)
            test_feature_extraction(po)
            success_count += 1
            po.commit_temps()
            logger.debug(f"PO {success_count} band maths, correlation and feature extraction success")
        except:
            logger.error(f"PO {success_count} band maths, correlation and feature extraction failed")
            fail_count += 1
            return
    return True if fail_count == 0 else False
        
    
def create_base_datasets_save_and_reload(hole):
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    try:
        logger.debug("Testing create_base_datasets...")
        hole.create_base_datasets()
        if 'depths' in hole.base_datasets and 'AvSpectra' in hole.base_datasets:
            depths_shape = hole.base_datasets['depths'].data.shape
            spectra_shape = hole.base_datasets['AvSpectra'].data.shape
            logger.debug(f"  depths shape: {depths_shape}")
            logger.debug(f"  AvSpectra shape: {spectra_shape}")
            logger.debug("PASS Base datasets created")
        else:
            logger.error("FAIL Base datasets not created")
            return False
    except Exception as e:
        logger.error(f"FAIL create_base_datasets failed: {e}", exc_info=True)
        return False
    # Test base_datasets_save_and_reload
    try:
        logger.debug("Testing base datasets save and reload...")
        # Save
        for ds in hole.base_datasets.values():
            ds.save_dataset()
        hole.load_hole_datasets()
        if 'depths' in hole.base_datasets and 'AvSpectra' in hole.base_datasets:
            logger.debug("PASS Base datasets saved and reloaded")
            return True
        else:
            logger.error("FAIL Base datasets not reloaded")
            return False
    except Exception as e:
        logger.error(f"FAIL Base datasets save/reload failed: {e}", exc_info=True)
        return False

def test_dhole_minmap(hole):
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    # Find available mineral maps for testing
    minmap_keys = []
    if hole.boxes:
        first_box = hole[hole.first_box]
        for key in first_box.datasets.keys():
            if key.endswith('INDEX'):
                if "kmeans" not in key:
                    minmap_keys.append(key)
    if minmap_keys:
        try:
            logger.debug(f"Testing create_dhole_minmap with {minmap_keys[0]}...")
            hole.create_dhole_minmap(minmap_keys[0])
            
            fracs_key = minmap_keys[0].replace("INDEX", "FRACTIONS")
            dom_key = minmap_keys[0].replace("INDEX", "DOM-MIN")
            
            if fracs_key in hole.product_datasets and dom_key in hole.product_datasets:
                logger.debug(f"  Created {fracs_key} and {dom_key}")
                logger.debug("PASS create_dhole_minmap works")
                return True
            else:
                logger.error("FAIL Downhole minmap datasets not created")
                return False
        except Exception as e:
            logger.error(f"FAIL create_dhole_minmap failed: {e}", exc_info=True)
            return False
    else:
        logger.warning("SKIP No mineral maps available for testing")
    
def test_dhole_features_box_aggregation(hole):
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    feature_keys = []
    if hole.boxes:
        first_box = hole[hole.first_box]
        for key in first_box.datasets.keys():
            if key.endswith('POS') or key.endswith('DEP'):
                feature_keys.append(key)
    
    # Test create_dhole_features
    if feature_keys:
        try:
            logger.debug(f"Testing create_dhole_features with {feature_keys[0]}...")
            hole.create_dhole_features(feature_keys[0])
            
            if feature_keys[0] in hole.product_datasets:
                logger.debug(f"  Created {feature_keys[0]}")
                logger.debug("PASS create_dhole_features works")
                return True
            else:
                logger.error("FAIL Downhole feature dataset not created")
                return False
        except Exception as e:
            logger.error(f"FAIL create_dhole_features failed: {e}", exc_info=True)
            return False
    else:
        logger.warning("SKIP No features available for testing")
        return False

def test_stepping_datasets(hole):
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    # Look for FRACTIONS/DOM-MIN keys in hole.product_datasets (not box datasets!)
    minmap_keys = []
    for key in hole.product_datasets.keys():
        if key.endswith('FRACTIONS'):
            minmap_keys.append(key)
    
    if not minmap_keys:
        logger.error("FRACTIONS keys not available in hole.product_datasets for testing")
        return False
    
    # Look for feature keys in hole.product_datasets
    feature_keys = []
    for key in hole.product_datasets.keys():
        if key.endswith('POS') or key.endswith('DEP'):
            feature_keys.append(key)
    
    if not feature_keys:
        logger.error("Feature keys not available in hole.product_datasets for testing")
        return False
    # Test step_product_fractions (if minmap was created)
    success_count = 0
    fracs_key = minmap_keys[0].replace("INDEX", "FRACTIONS") if minmap_keys else None
    if fracs_key and fracs_key in hole.product_datasets:
        try:
            logger.debug(f"Testing step_product_dataset with FRACTIONS...")
            depths_s, fractions_s, dominant_s = hole.step_product_dataset(fracs_key)
            logger.debug(f"  Stepped depths shape: {depths_s.shape}")
            logger.debug(f"  Stepped fractions shape: {fractions_s.shape}")
            logger.debug(f"  Stepped dominant shape: {dominant_s.shape}")
            logger.debug("PASS step_product (FRACTIONS) works")
            success_count += 1
        except Exception as e:
            logger.error(f"FAIL step_product FRACTIONS failed: {e}", exc_info=True)
            return False
    else:
        logger.warning("SKIP No FRACTIONS dataset for stepping test")
        return False
    
    # Test step_product_indices (if minmap was created)
    dom_key = minmap_keys[0].replace("INDEX", "DOM-MIN") if minmap_keys else None
    if dom_key and dom_key in hole.product_datasets:
        try:
            logger.debug(f"Testing step_product_dataset with DOM-MIN (indices)...")
            depths_s, indices_s, _ = hole.step_product_dataset(dom_key)
            logger.debug(f"  Stepped depths shape: {depths_s.shape}")
            logger.debug(f"  Stepped indices shape: {indices_s.shape}")
            logger.debug("PASS step_product (indices) works")
            success_count += 1
        except Exception as e:
            logger.error(f"FAIL step_product indices failed: {e}", exc_info=True)
            return False
    else:
        logger.warning("SKIP No DOM-MIN dataset for stepping test")
        return False
    
    # Test step_product_continuous (if feature was created)
    if feature_keys and feature_keys[0] in hole.product_datasets:
        try:
            logger.debug(f"Testing step_product_dataset with continuous feature...")
            depths_s, values_s, _ = hole.step_product_dataset(feature_keys[0])
            logger.debug(f"  Stepped depths shape: {depths_s.shape}")
            logger.debug(f"  Stepped values shape: {values_s.shape}")
            logger.debug("PASS step_product (continuous) works")
            success_count += 1
        except Exception as e:
            logger.error(f"FAIL step_product continuous failed: {e}", exc_info=True)
            return False
    else:
        logger.warning("SKIP No continuous feature dataset for stepping test")
        return False
    if success_count == 3:
        return True
    else:
        return False

    
def test_profile_bandmaths(hole):
    """Test profile band math operations on hole-level average spectra."""
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    
    # Check that base datasets exist
    if 'AvSpectra' not in hole.base_datasets:
        logger.error("FAIL Base datasets not created - run create_base_datasets first")
        return False
    
    success_count = 0
    
    for expr_config in BAND_MATH_EXPRESSIONS:
        try:
            name = expr_config['name']
            expression = expr_config['expression']
            use_cr = expr_config['continuum_removed']
            
            logger.debug(f"Evaluating profile band math: {name}")
            logger.debug(f"  Expression: {expression}")
            logger.debug(f"  Continuum removed: {use_cr}")
            
            # Run band math on profile
            hole = pt.band_math_interface(hole, name, expression, cr=use_cr)
            
            # Check if dataset was created
            clean_key = re.sub(r'[\\/:*?"<>|_]', '-', name)
            if clean_key in hole.product_datasets:
                result = hole.product_datasets[clean_key].data
                logger.debug(f"  PASS {name}: shape {result.shape}")
                logger.debug(f"    Value range: [{result.min():.3f}, {result.max():.3f}]")
                success_count += 1
            else:
                logger.error(f"  FAIL {name}: dataset not created")
        
        except Exception as e:
            logger.error(f"  FAIL {name} failed: {e}", exc_info=True)
    
    logger.debug(f"Profile band math summary: {success_count}/{len(BAND_MATH_EXPRESSIONS)} successful")
    return success_count > 0


def test_profile_kmeans(hole):
    """Test profile k-means clustering on hole-level average spectra."""
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    
    # Check that base datasets exist
    if 'AvSpectra' not in hole.base_datasets:
        logger.error("FAIL Base datasets not created - run create_base_datasets first")
        return False
    
    try:
        clusters = KMEANS_CLUSTERS
        iterations = KMEANS_ITERATIONS
        
        logger.debug(f"Running profile k-means clustering...")
        logger.debug(f"  Clusters: {clusters}")
        logger.debug(f"  Iterations: {iterations}")
        
        # Run k-means on profile
        hole = pt.profile_kmeans(hole, clusters=clusters, iters=iterations)
        
        # Check if datasets were created
        key_prefix = f'PROF-kmeans-{clusters}-{iterations}'
        has_index = f'{key_prefix}INDEX' in hole.product_datasets
        has_clusters = f'{key_prefix}CLUSTERS' in hole.product_datasets
        has_legend = f'{key_prefix}LEGEND' in hole.product_datasets
        
        if has_index and has_clusters and has_legend:
            cluster_map = hole.product_datasets[f'{key_prefix}INDEX'].data
            centroids = hole.product_datasets[f'{key_prefix}CLUSTERS'].data
            
            logger.debug("PASS Profile k-means clustering successful")
            logger.debug(f"  Cluster map shape: {cluster_map.shape}")
            logger.debug(f"  Centroids shape: {centroids.shape}")
            logger.debug(f"  Unique clusters: {np.unique(cluster_map)}")
            return True
        else:
            logger.error("FAIL Profile k-means datasets not created")
            return False
    
    except Exception as e:
        logger.error(f"FAIL Profile k-means clustering failed: {e}", exc_info=True)
        return False


def test_profile_features(hole):
    """Test profile feature extraction on hole-level average spectra."""
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    
    # Check that base datasets exist
    if 'AvSpectra' not in hole.base_datasets:
        logger.error("FAIL Base datasets not created - run create_base_datasets first")
        return False
    
    success_count = 0
    
    for feature_key in FEATURES_TO_EXTRACT:
        try:
            logger.debug(f"Extracting profile feature: {feature_key}")
            
            # Run feature extraction on profile
            hole = pt.run_feature_extraction(hole, feature_key)
            
            # Check if datasets were created
            has_pos = f'PROF-{feature_key}POS' in hole.product_datasets
            has_dep = f'PROF-{feature_key}DEP' in hole.product_datasets
            
            if has_pos and has_dep:
                position = hole.product_datasets[f'PROF-{feature_key}POS'].data
                depth = hole.product_datasets[f'PROF-{feature_key}DEP'].data
                
                logger.debug(f"  PASS {feature_key}: position shape {position.shape}")
                logger.debug(f"    Depth range: [{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}]")
                success_count += 1
            else:
                logger.error(f"  FAIL {feature_key}: datasets not created")
        
        except Exception as e:
            logger.error(f"  FAIL {feature_key} failed: {e}", exc_info=True)
    
    logger.debug(f"Profile feature extraction summary: {success_count}/{len(FEATURES_TO_EXTRACT)} successful")
    return success_count > 0


def test_profile_correlation(hole, lib_manager):
    """Test profile correlation analysis on hole-level average spectra."""
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    
    if not lib_manager:
        logger.error("FAIL No library manager provided")
        return False
    
    # Check that base datasets exist
    if 'AvSpectra' not in hole.base_datasets:
        logger.error("FAIL Base datasets not created - run create_base_datasets first")
        return False
    
    
    # Get collection
    collection_name = CORRELATION_COLLECTION
    
    try:
        exemplars = lib_manager.get_collection_exemplars(collection_name)
        
        if not exemplars:
            logger.error(f"FAIL Collection '{collection_name}' is empty")
            return False
    except Exception as e:
        logger.error(f"FAIL Failed to get collection exemplars: {e}", exc_info=True)
        return False
    
    correlation_methods = [
        ('pearson', pt.wta_min_map),
        ('SAM', pt.wta_min_map_SAM),
        ('MSAM', pt.wta_min_map_MSAM)
    ]
    
    success_count = 0
    
    for method_name, method_func in correlation_methods:
        try:
            logger.debug(f"Running profile {method_name} correlation...")
            
            # Run correlation on profile
            hole = method_func(hole, exemplars, collection_name)
            
            # Check if datasets were created
            key_prefix = f"PROF-MinMap-{method_name}-{collection_name.replace('_', '')}"
            has_index = f"{key_prefix}INDEX" in hole.product_datasets
            has_legend = f"{key_prefix}LEGEND" in hole.product_datasets
            has_conf = f"{key_prefix}CONF" in hole.product_datasets
            
            if has_index and has_legend and has_conf:
                index = hole.product_datasets[f"{key_prefix}INDEX"].data
                confidence = hole.product_datasets[f"{key_prefix}CONF"].data
                
                logger.debug(f"  PASS Profile {method_name}: created datasets")
                logger.debug(f"    Index shape: {index.shape}")
                logger.debug(f"    Unique classes: {len(np.unique(index[index >= 0]))}")
                logger.debug(f"    Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
                success_count += 1
            else:
                logger.error(f"  FAIL Profile {method_name}: missing datasets")
        
        except Exception as e:
            logger.error(f"  FAIL Profile {method_name} failed: {e}", exc_info=True)
    
    logger.debug(f"Profile correlation summary: {success_count}/{len(correlation_methods)} successful")
    return success_count > 0


def test_save_product_datasets(hole):
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    try:
        logger.debug("Testing save_product_datasets...")
        hole.save_product_datasets()
        
        # Verify files exist
        saved_count = 0
        for key, ds in hole.product_datasets.items():
            if ds.path.exists():
                saved_count += 1
        
        logger.debug(f"  Saved {saved_count}/{len(hole.product_datasets)} product datasets")
        if saved_count == len(hole.product_datasets):
            logger.debug("PASS Product datasets saved")
        else:
            logger.error("FAIL Not all product datasets saved")
            return False
    except Exception as e:
        logger.error(f"FAIL save_product_datasets failed: {e}", exc_info=True)
        return False
    
    return True

def test_export_profile_csv(hole):
    """Test CSV export functionality for hole-level product datasets."""
    if not hole:
        logger.error("FAIL No hole provided")
        return False
    
    # Check that base datasets exist (required for export)
    if 'depths' not in hole.base_datasets:
        logger.error("FAIL Base datasets not created - run create_base_datasets first")
        return False
    
    # Check that product datasets exist to export
    if not hole.product_datasets:
        logger.error("FAIL No product datasets available for export")
        return False
    
    from app.interface import profile_tools as pt
    
    success_count = 0
    total_attempted = 0
    
    # Test exporting different types of datasets
    for key in hole.product_datasets.keys():
        # Skip non-exportable types (LEGEND, CLUSTERS)
        if key.endswith('LEGEND') or key.endswith('CLUSTERS'):
            logger.debug(f"  Skipping {key} (not exportable)")
            continue
        
        total_attempted += 1
        
        try:
            logger.debug(f"Exporting {key} to CSV...")
            
            # Export in 'both' mode (full and stepped) using default path
            created_files = pt.export_profile_to_csv(
                hole,
                key,
                mode='both',
                step=hole.step
            )
            
            # Verify files were created
            if created_files:
                all_exist = all(f.exists() for f in created_files)
                if all_exist:
                    logger.debug(f"  PASS {key}: exported {len(created_files)} CSV files")
                    for csv_file in created_files:
                        logger.debug(f"    Created: {csv_file.name}")
                    success_count += 1
                else:
                    logger.error(f"  FAIL {key}: some CSV files not created")
            else:
                logger.error(f"  FAIL {key}: no CSV files returned")
                
        except Exception as e:
            logger.error(f"  FAIL {key} export failed: {e}", exc_info=True)
    
    logger.debug(f"CSV export summary: {success_count}/{total_attempted} datasets exported")
    
    # Return True if at least one dataset was successfully exported
    return success_count > 0

#==============================================================================
# BASELIN TESTING
#==============================================================================


def test_against_baseline(basename):

    """
    Compare artifacts for one PO basename between baseline_dir and actual_dir.
    By default compares everything that exists in baseline_dir for that basename.
    """
    base_meta = BASELINE_DIR / f"{basename}_metadata.json"
    out_meta  = OUTPUT_DIR   / f"{basename}_metadata.json"
    base_po = ProcessedObject.from_path(base_meta)
    out_po  = ProcessedObject.from_path(out_meta)
    
    required_keys = sorted(base_po.datasets.keys())

    missing_in_actual = [k for k in required_keys if k not in out_po.datasets.keys()]
    missing_in_base   = [k for k in required_keys if k not in base_po.datasets.keys()]
    if missing_in_base:
        raise AssertionError(f"Missing in baseline: {missing_in_base}")
    if missing_in_actual:
        raise AssertionError(f"Missing in actual: {missing_in_actual}")

    results = dict()
    failures = []
    for k in required_keys:
        try:
            _compare_one_key(k, base_po.datasets[k].data, out_po.datasets[k].data, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
            results[f"{k} baseline compare"] = True
        except Exception as e:
            results[f"{k} baseline compare"] = False
            failures.append(str(e))
        

    if failures:
        logger.error("Baseline verification FAILED with the following differences:")
        for f in failures:
            logger.error(f"  - {f}")
    else:
        logger.debug("PASS Baseline verification PASSED (actual matches baseline).")
    return results



def test_baseline_hole(hole_id):
    """
    Compare hole-level artifacts between baseline_dir and output_dir.
    Compares HoleObject base_datasets and product_datasets.
    
    Parameters
    ----------
    hole_id : str
        The hole identifier (e.g., "TEST_HOLE_001")
    
    Returns
    -------
    dict
        Dictionary of test results for each compared dataset
    """
    baseline_hole_dir = BASELINE_DIR / hole_id
    output_hole_dir = OUTPUT_DIR / hole_id
    
    # Load both holes
    try:
        baseline_hole = HoleObject.build_from_parent_dir(baseline_hole_dir, hole_id)
        output_hole = HoleObject.build_from_parent_dir(output_hole_dir, hole_id)
    except Exception as e:
        logger.error(f"Failed to load holes for comparison: {e}", exc_info=True)
        return {"hole_load_error": False}
    
    results = {}
    failures = []
    
    # ========================================================================
    # Compare hole-level metadata
    # ========================================================================
    try:
        if baseline_hole.hole_id != output_hole.hole_id:
            failures.append(f"Hole IDs differ: {baseline_hole.hole_id} vs {output_hole.hole_id}")
            results["hole_id_match"] = False
        else:
            results["hole_id_match"] = True
            
        if baseline_hole.num_box != output_hole.num_box:
            failures.append(f"Number of boxes differ: {baseline_hole.num_box} vs {output_hole.num_box}")
            results["num_boxes_match"] = False
        else:
            results["num_boxes_match"] = True
    except Exception as e:
        logger.error(f"Metadata comparison failed: {e}", exc_info=True)
        results["metadata_comparison"] = False
        failures.append(f"Metadata comparison error: {e}")
    
    # ========================================================================
    # Compare base_datasets (depths, AvSpectra)
    # ========================================================================
    baseline_base_keys = set(baseline_hole.base_datasets.keys())
    output_base_keys = set(output_hole.base_datasets.keys())
    
    missing_in_output = baseline_base_keys - output_base_keys
    extra_in_output = output_base_keys - baseline_base_keys
    
    if missing_in_output:
        failures.append(f"Missing base datasets in output: {missing_in_output}")
        for key in missing_in_output:
            results[f"hole_base_{key}"] = False
    
    if extra_in_output:
        logger.warning(f"Extra base datasets in output (not in baseline): {extra_in_output}")
    
    # Compare common base datasets
    for key in baseline_base_keys & output_base_keys:
        try:
            baseline_data = baseline_hole.base_datasets[key].data
            output_data = output_hole.base_datasets[key].data
            
            _compare_one_key(
                key,  # Use just the key, not prefixed
                baseline_data,
                output_data,
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL
            )
            results[f"hole_base_{key}"] = True
        except Exception as e:
            results[f"hole_base_{key}"] = False
            failures.append(f"hole_base_{key}: {e}")
    
    # ========================================================================
    # Compare product_datasets (downhole aggregations, profile products)
    # ========================================================================
    baseline_product_keys = set(baseline_hole.product_datasets.keys())
    output_product_keys = set(output_hole.product_datasets.keys())
    
    missing_in_output = baseline_product_keys - output_product_keys
    extra_in_output = output_product_keys - baseline_product_keys
    
    if missing_in_output:
        failures.append(f"Missing product datasets in output: {missing_in_output}")
        for key in missing_in_output:
            results[f"hole_product_{key}"] = False
    
    if extra_in_output:
        logger.warning(f"Extra product datasets in output (not in baseline): {extra_in_output}")
    
    # Compare common product datasets
    for key in baseline_product_keys & output_product_keys:
        try:
            baseline_data = baseline_hole.product_datasets[key].data
            output_data = output_hole.product_datasets[key].data
            
            # Use appropriate comparison based on key type
            _compare_one_key(
                key,  # Use just the key, not prefixed
                baseline_data,
                output_data,
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL
            )
            results[f"hole_product_{key}"] = True
        except Exception as e:
            results[f"hole_product_{key}"] = False
            failures.append(f"hole_product_{key}: {e}")
    
    # ========================================================================
    # DON'T compare individual boxes - they're already compared in test_against_baseline
    # Skip box comparison to avoid duplicate testing
    # ========================================================================
    
    # ========================================================================
    # Report results
    # ========================================================================
    if failures:
        logger.error("Hole baseline verification FAILED with the following differences:")
        for f in failures:
            logger.error(f"  - {f}")
    else:
        logger.info("PASS Hole baseline verification PASSED (output matches baseline)")
    
    return results



# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_integration_test(create_baseline = False):
    """Run the complete integration test."""
    print_section("CoreSpecViewer Integration Test")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Test configuration:")
    logger.info(f"  Raw directory: {RAW_LUMO_DIR}")
    logger.info(f"  Baseline directory: {BASELINE_DIR}")
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    logger.info(f"  Library database: {LIBRARY_DB_PATH}")
    if create_baseline:
        logger.info(f"  Creating baseline data")
    
    # Validate paths
    if not validate_paths():
        logger.error("Path validation failed. Please check configuration.")
        return False
    
    # Track test results
    results = {
        'raw_loading': False,
        'crop': False,
        'process': False,
        'mask_new': False,
        'mask_enhance': False,
        'mask_despeckle': False,
        'mask_improve': False,
        'mask_stats': False,
        'mask_unwrap': False,
        'kmeans': False,
        'features': False,
        'band_math': False,
        'save': False,
        'library': False,
        'reload': False,
        'correlation': False
    }
    
    # Part 1: Raw loading and processing
    logger.info("Running raw loading test")
    ro = test_raw_loading()
    results['raw_loading'] = ro is not None
    
    if not ro:
        return report_results(results)
    logger.info("Running crop test")
    ro = test_crop(ro)
    results['crop'] = ro is not None
    if not ro:
        return report_results(results)
    logger.info("Running process test")
    po = test_process_to_reflectance(ro, create_baseline=create_baseline)
    results['process'] = po is not None
    if not po:
        return report_results(results)
    
    # Part 2: Masking operations
    logger.info("Running masking tests")
    results['mask_new'] = test_create_new_mask(po)
    results['mask_enhance'] = test_enhance_mask(po)
    results['mask_despeckle'] = test_despeckle_mask(po)
    results['mask_improve'] = test_improve_mask(po)
    results['mask_stats'] = test_calc_unwrap_stats(po)
    
    # Only test unwrap if stats succeeded
    if results['mask_stats']:
        results['mask_unwrap'] = test_unwrap_preview(po)
    
    # Part 3: Analysis
    logger.info("Running analysis tests")
    results['kmeans'] = test_kmeans(po)
    results['features'] = test_feature_extraction(po)
    results['band_math'] = test_band_maths(po)
    
    # Part 4: Save
    logger.info("Running save box test")
    results['save'] = test_save(po)
    if not results['save']:
        return report_results(results)
    
    # Store path for reload
    po_path = po.root_dir
    po_basename = po.basename
    
    # Part 5: Library and correlation
    logger.info("Running library tests")
    lib_manager = test_library_loading()
    results['library'] = lib_manager is not None
    
    logger.info("Running reload tests")
    po_reloaded = test_reload_processed(po_path)
    results['reload'] = po_reloaded is not None
    
    if po_reloaded and lib_manager:
        logger.info("Running correlation tests")
        results['correlation'] = test_correlation(po_reloaded, lib_manager)
        
    #Part 6: full hole datasets
    logger.info("Running Full hole tests")
    hole, flag =  create_three_box_hole_dataset(po, create_baseline=create_baseline) 
    results['hole created'] = flag
    results["hole iteration"] = test_hole_iteration(hole)
    results["hole indexing"] = test_hole_indexing(hole)
    results["box by box data creation"] = test_box_by_box(hole, lib_manager)
    results["base datasets, create and reload"] = create_base_datasets_save_and_reload(hole)
    results["downhole minmap"] = test_dhole_minmap(hole)
    results["downhole features"] = test_dhole_features_box_aggregation(hole)
    results["stepping downhole"] = test_stepping_datasets(hole)
    # Add profile tests
    results["profile bandmaths"] = test_profile_bandmaths(hole)
    results["profile kmeans"] = test_profile_kmeans(hole)
    results["profile features"] = test_profile_features(hole)
    results["profile correlation"] = test_profile_correlation(hole, lib_manager)
    results["Saving product datasets"] = test_save_product_datasets(hole)
    results["Export csvs"] = test_export_profile_csv(hole)
    
    if not create_baseline:
        logger.info("Running baseline tests")
        baseline_results = test_against_baseline(po_basename)
        hole_baseline_results = test_baseline_hole(hole.hole_id)  
        results = results | baseline_results | hole_baseline_results
        
        
        
    
    return report_results(results)


def report_results(results):
    """Print final test results."""
    print_section("TEST RESULTS SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    logger.info("")
    for test_name, passed_flag in results.items():
        status = "PASS" if passed_flag else "FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info("")
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("ALL TESTS PASSED!")
        return True
    else:
        logger.warning(f" {total - passed} test(s) failed")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CoreSpecViewer Integration Test')
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Create baseline data instead of comparing against it'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_integration_test(create_baseline=args.baseline)

