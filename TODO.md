# CoreSpecViewer Development TODO

## Technical Clean Up

- [x] **Set up logging**
  - [x] Replace print statements with proper logging
  - [x] Log levels: DEBUG, INFO, WARNING, ERROR
  - [x] Write logs to file in user directory
  
- [ ] **Sort test harness**
  - [ ] Set up test framework
  - [x] Create test data fixtures (small example datasets)
  - [ ] Basic smoke tests for loading/saving
  
- [ ] **Dependency updates**
  - [ ] Test with newer versions where appropriate
  - [ ] Document any version constraints
  
- [ ] **Refactor spectral_functions.py**
  - [ ] Split into logical modules (io, processing, visualization, correlation)
  - [ ] Decouple `con_dict` from pure algorithms
  - [ ] Document sensor-specific logic clearly

---

## Features

- [ ] **ClusterWindow profiles**
  - [ ] Enable viewing cluster centres derived from profile datasets
  
- [ ] **Profile minmaps**
  - [ ] Minimum wavelength mapping for 1D profiles
  
- [ ] **Export profiles csv, ascii, las**
  - [ ] CSV with header/metadata
  - [ ] ASCII columnar format
  - [ ] LAS format for geological software compatibility
  
- [ ] **Import external data**
  - [ ] Generic ENVI file support
  - [ ] Tabular data (CSV/Excel/las) for overlays
  - [ ] HyLogger format if needed
  
- [ ] **HDBscan?**
  - [ ] Evaluate vs existing k-means
  - [ ] Test on real datasets
  - [ ] Decide if worth the dependency
  
- [ ] **Use external data for ML???**
  - [ ] Supervised learning workflow?
  - [ ] External data as labels
  - [ ] Training data integration?

---

## Code Quality

- [ ] **Naming consistency**
  - [ ] Standardize controller methods (act_* pattern or similar)
  - [ ] Review variable naming conventions
  - [ ] Document any abbreviations used
  
- [ ] **Docstring audit**
  - [ ] Ensure all public methods have docstrings
  - [ ] Stick to NumPy style consistently?
  - [ ] Add examples to complex functions
  
- [ ] **Remove legacy cruft**
  - [ ] Remove commented-out code
  - [ ] Delete unused imports
  - [ ] Audit for dead code paths

- [ ] **Error handling improvements**
  - [ ] Replace generic exceptions with specific ones
  - [ ] User-friendly error dialogs (not stack traces)
  - [ ] Better handling of corrupted/incomplete datasets

---

## Documentation

- [ ] **User guide**
  - [ ] Step-by-step workflow walkthrough
  - [ ] Screenshots of each page/mode
  - [ ] Common troubleshooting issues
  
- [ ] **Keyboard shortcuts reference**
  - [ ] List all defined shortcuts
  - [ ] Make discoverable in UI (Help menu?)
  
---

## Polish

- [ ] **Progress indicators**
  - [ ] Long operations show progress bars
  - [ ] Estimated time remaining
  
- [ ] **Better temp_datasets workflow**
  - [ ] More granular undo patter
  
- [ ] **Configuration save/load** 
  - [ ] Save user configs to disk
  - [ ] Minimum reproducable files for archive and transfer

---

## Distribution (If requested)

- [ ] **PyInstaller bundling**
  - [ ] Create standalone executable for Windows
  
---

_Last Updated: 2026-01-22_
