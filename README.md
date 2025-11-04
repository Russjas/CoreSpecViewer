# Hyperspectral Core Scanning GUI

A modular **PyQt5 application** for managing, processing, and visualizing hyperspectral drill-core datasets.  
It provides an integrated interface for **Specim Lumo** raw exports, **processed reflectance cubes**, and **derived geological products**, combining interactive tools, visual previews, and spectral analysis workflows.

---

## Overview

This repository implements a complete desktop framework for hyperspectral core scanning workflows, combining:

- **Raw data handling** for Specim Lumo exports (`RawObject`)
- **Processed datasets** management and persistence (`ProcessedObject`)
- **Spectral processing** and feature extraction (`spectral_functions.py`)
- **Interactive GUI** with ribbon-style navigation (`ribbon_window.py`, `pages.py`, `ribbon.py`)
- **Custom tools** for masking, cropping, and feature mapping (`tools.py`)
- **Safe event management and inter-window communication** (`tool_dispatcher.py`, `util_windows.py`)

The system is designed for geoscientists and hyperspectral imaging specialists, with an emphasis on **clarity, reproducibility, and modularity**.

---

## Repository Structure

| File | Purpose |
|------|----------|
| **`ribbon_window.py`** | Main application entry point. Defines the `MainRibbonController`, which builds the main window, ribbon, and tabbed interface. Handles file loading/saving and delegates actions to active pages. |
| **`ribbon.py`** | Lightweight implementation of a Microsoft-style ribbon UI (`QTabWidget`-based) with per-tab `QToolBar` population. |
| **`pages.py`** | Defines GUI pages (`RawPage`, `VisualisePage`, `LibraryPage`), each embedding `SpectralImageCanvas`, `ImageCanvas2D`, and `InfoTable` layouts. Responsible for canvas synchronization and data display logic. |
| **`util_windows.py`** | Utility widgets and helpers: busy cursor context manager, `InfoTable` for metadata display, and advanced matplotlib-based image/spectrum canvases. |
| **`objects.py`** | Core data model: defines `Dataset`, `ProcessedObject`, `RawObject`, and `HoleObject`. These encapsulate hyperspectral cubes, metadata, masks, and derived data. Provides automatic discovery, loading, and saving. |
| **`spectral_functions.py`** | Scientific processing backend: ENVI file I/O, reflectance correction, Savitzky-Golay smoothing, mask enhancement, segmentation, SNR estimation, and spectral feature mapping. |
| **`tools.py`** | Workflow-level tools: crop, auto-crop, reset, mask creation, correlation, and reflectance extraction, operating seamlessly on `RawObject` or `ProcessedObject` instances. |
| **`tool_dispatcher.py`** | Event handler wrapper allowing multiple tools to share the same matplotlib canvas callbacks safely. |
| **`util_windows.py`** | Defines shared GUI elements and event bus for inter-component signaling. |

---



## Running the Application

Run the main window directly:

```bash
python ribbon_window.py
```

You’ll be presented with a **Ribbon-based UI** with tabs:

- **Raw:** open and process Specim Lumo directories  
- **Masking:** apply masks, refine and unwrap core regions  
- **Visualise:** correlate features, view extracted products, and inspect spectral data  
- **Libraries:** explore spectral databases (SQLite/ECOSTRESS compatible)

---

## Key Features

### Data Handling
- Automatic recognition of `.hdr`, `.npy`, `.npz`, `.json`, and `.jpg` datasets
- Seamless integration between raw Specim data and processed reflectance cubes
- Automatic memory mapping for large datasets

### Spectral Processing
- Reflectance correction via `reflect_correct`
- Band selection based on SNR (`bands_from_snr`)
- Savitzky–Golay smoothing and continuum removal (`process`)
- Feature extraction and correlation mapping

### GUI & Interaction
- Matplotlib-powered RGB and spectral viewers
- Rectangle and point-based ROI selection
- Live synchronization between canvases
- Mask editing and improvement tools
- Ribbon-based workflow navigation

---

## Architecture Overview

```
┌──────────────────────────┐
│   MainRibbonController   │  ← ribbon_window.py
│ ┌──────────────────────┐ │
│ │      Ribbon (UI)     │ │
│ └────────────┬─────────┘ │
│ ┌────────────┴─────────┐ │
│ │   Tabbed Pages       │ │  ← pages.py
│ │  (Raw / Mask / Vis)  │ │
│ └────────────┬─────────┘ │
│ ┌────────────┴─────────┐ │
│ │  Data Objects (PO)   │ │  ← objects.py
│ │  + Tools & Functions │ │  ← tools.py, spectral_functions.py
│ └──────────────────────┘ │
└──────────────────────────┘
```

---

## Add your own functions

- Add new tools by defining functions in `tools.py` and linking them to a Ribbon action.
- Add new derived datasets by extending `ProcessedObject.add_dataset`.
- Integrate additional visualization pages by subclassing `BasePage`.

---

## License

-This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.  
-Because it depends on **PyQt5**, redistribution and derivative works must also comply with GPL v3 terms.**
---

## Author

Developed by **Russell Rogers**  

