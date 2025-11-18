
"""
CoreSpecViewer application package.

This package contains the full desktop application stack for CoreSpecViewer:
a Qt-based GUI for inspecting, masking, processing, and visualising
hyperspectral drill-core data.

Subpackages
-----------
- ui
    All Qt widgets and pages (Raw/Visualise/Library/Hole views, ribbon,
    catalogue browser, utility dialogs and canvas widgets).

- interface
    The thin interaction layer that connects UI gestures to data operations.
    Includes the ToolDispatcher and tool functions that operate on model
    objects.

- models
    Core data structures such as RawObject, ProcessedObject, HoleObject,
    Dataset and CurrentContext. These wrap on-disk data, memory-mapped arrays,
    metadata and hole-level organisation.

- spectral
    Spectral utilities and feature extraction helpers. Provides convenience
    functions for reflectance correction, false-colour composites, spectral
    features, and correlation/mineral-mapping back-ends.

Other modules
-------------
- config
    Single in-memory configuration dictionary (con_dict) and helpers to load,
    mutate and persist user settings.

- main
    Entry point defining MainRibbonController and the `main()` function to
    launch the GUI.

Typical usage
-------------
Most users will start the application via:

    python -m app.main

Advanced users and plugins can import specific layers, e.g.:

    from app.models import ProcessedObject
    from app.interface import tools
    from app.ui import VisualisePage
"""

