
"""
load_dialogue.py

Custom UI modal dialog for opening datasets, with **per-pag"e Load buttons**.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
import logging
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox,
    QStackedWidget, QFrame, QCheckBox, QSizePolicy
)

from ..interface import tools as t
from ..models.raw_object import RawObject
from ..models.hole_object import HoleObject
from ..models.processed_object import ProcessedObject
from ..models.context import CurrentContext
from .util_windows import busy_cursor, MetadataDialog

logger = logging.getLogger(__name__)

ViewFlag = Literal["raw", "vis", "hol"]
REQUIRED_META_KEYS = (
    "borehole id",
    "box number",
    "core depth start",
    "core depth stop",
)

class LoadDialogue(QDialog):
    def __init__(self, cxt, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open dataset")
        self.setModal(True)
        self.resize(820, 480)

        self.cxt = cxt
        self.view_flag: Optional[ViewFlag] = None

        self._build_ui()

    # --------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("Open files")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)

        subtitle = QLabel(
            "Choose the type of data you wish to open and the menu will guide you."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #555;")
        root.addWidget(subtitle)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        # Pages
        self.page_menu = self._build_menu_page()
        self.page_processed = self._build_processed_page()
        self.page_lumo = self._build_lumo_page()
        self.page_hole = self._build_hole_page()
        self.page_raw = self._build_raw_page()
        self.ref_page = self._build_reflectance_page()

        self.stack.addWidget(self.page_menu)       # index 0
        self.stack.addWidget(self.page_processed)  # index 1
        self.stack.addWidget(self.page_lumo)        # index 2
        self.stack.addWidget(self.page_hole)       # index 3
        self.stack.addWidget(self.page_raw)    # index 4
        self.stack.addWidget(self.ref_page)    # index 5

        # Bottom row: Back + Cancel
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self.btn_back = QPushButton("← Back")
        self.btn_back.clicked.connect(self._go_back)
        self.btn_back.setEnabled(False)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        bottom.addWidget(self.btn_back)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_cancel)
        root.addLayout(bottom)

        self._show_menu()

    def _build_menu_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(12)

        #CSV native files
        lay.addWidget(self._section_label("CoreSpecViewer created datasets"))

        csv_btns = QGridLayout()
        csv_btns.setHorizontalSpacing(12)
        csv_btns.setVerticalSpacing(12)

        def big_button(text: str, help_text: str) -> QPushButton:
            b = QPushButton(text)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            b.setMinimumHeight(78)
            b.setStyleSheet(
                "text-align: left; padding: 12px; font-size: 13px; font-weight: 600;"
            )
            b.setToolTip(help_text)
            return b

        csv_processed_btn = big_button(
            "Open processed dataset (JSON metadata file)",
            "Select the processed dataset JSON. Sibling datasets in the folder are discovered automatically."
        )
        full_hole_btn = big_button(
            "Open hole directory (batch of boxes)",
            "Select a hole directory containing multiple processed boxes (JSON files). Builds a HoleObject."
        )
        
        

        csv_processed_btn.clicked.connect(lambda: self._show_page(1))
        full_hole_btn.clicked.connect(lambda: self._show_page(3))
        csv_btns.addWidget(csv_processed_btn, 0, 0)
        csv_btns.addWidget(full_hole_btn, 0, 1)
        lay.addLayout(csv_btns, 1)
        #Sep + label
        lay.addWidget(self._hline())
        lay.addWidget(self._section_label("Open raw ENVI Files"))
        
                
        #Raw dataset controls
        raw_btns = QGridLayout()
        raw_btns.setHorizontalSpacing(12)
        raw_btns.setVerticalSpacing(12)
        lumo_raw_btn = big_button(
            "Open raw dataset from Lumo output directory",
            "Select the raw dataset directory. Lumo-style naming is inferred/validated inside RawObject."
        )
        
        raw_btn = big_button(
            "Open a raw dataset from indivual ENVI files",
            "Recovery mode: select headers (and optional metadata). Try infer .raw, or provide explicit raw files."
        )
                
        lumo_raw_btn.clicked.connect(lambda: self._show_page(2))
        raw_btn.clicked.connect(lambda: self._show_page(4))
        raw_btns.addWidget(lumo_raw_btn, 0,0)
        raw_btns.addWidget(raw_btn, 0, 1)

        lay.addLayout(raw_btns, 1)
        
        #Sep + label
        lay.addWidget(self._hline())
        lay.addWidget(self._section_label("Open reflectance data"))
        
        
        #post-processed reflectance envi files
        ref_btns = QGridLayout()
        ref_btns.setHorizontalSpacing(12)
        ref_btns.setVerticalSpacing(12)
        
        env_ref_btn = big_button(
            "Open reflectance data stored in an ENVI file",
            "Metadata validation in development - These files may not work as hoped!"
        )
        env_ref_btn.clicked.connect(lambda: self._show_page(5)) 
        ref_btns.addWidget(env_ref_btn, 0,0)
        lay.addLayout(ref_btns, 1)

        return w

    def _build_processed_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        lay.addWidget(self._section_label("Processed dataset"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.le_processed_json = self._path_line("Select a processed dataset JSON...")
        btn_browse = QPushButton("Browse JSON…")
        btn_browse.clicked.connect(
            lambda: self._browse_file_into(
                self.le_processed_json,
                "Select processed dataset metadata (JSON)",
                "JSON (*.json)"
            )
        )

        grid.addWidget(QLabel("Processed metadata JSON:"), 0, 0)
        grid.addWidget(self.le_processed_json, 0, 1)
        grid.addWidget(btn_browse, 0, 2)

        lay.addLayout(grid)

        btn_load = QPushButton("Load processed dataset")
        btn_load.clicked.connect(self._load_processed_clicked)
        lay.addWidget(btn_load)

        hint = QLabel(
            "Loads via tools.load(json_path) → ProcessedObject.from_path(...)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        lay.addWidget(hint)

        lay.addStretch(1)
        return w

    def _build_lumo_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        lay.addWidget(self._section_label("Raw dataset"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.le_raw_dir = self._path_line("Select a raw dataset directory...")
        btn_browse = QPushButton("Browse folder…")
        btn_browse.clicked.connect(
            lambda: self._browse_dir_into(self.le_raw_dir, "Select raw dataset directory")
        )

        grid.addWidget(QLabel("Raw dataset directory:"), 0, 0)
        grid.addWidget(self.le_raw_dir, 0, 1)
        grid.addWidget(btn_browse, 0, 2)

        lay.addLayout(grid)

        btn_load = QPushButton("Load raw dataset")
        btn_load.clicked.connect(self._load_lumo_clicked)
        lay.addWidget(btn_load)

        hint = QLabel(
            "Loads via tools.load(dir_path) → RawObject.from_Lumo_directory(...)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        lay.addWidget(hint)

        lay.addStretch(1)
        return w

    def _build_hole_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        lay.addWidget(self._section_label("Hole directory"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.le_hole_dir = self._path_line("Select a hole directory...")
        btn_browse = QPushButton("Browse folder…")
        btn_browse.clicked.connect(
            lambda: self._browse_dir_into(self.le_hole_dir, "Select hole directory")
        )

        grid.addWidget(QLabel("Hole directory:"), 0, 0)
        grid.addWidget(self.le_hole_dir, 0, 1)
        grid.addWidget(btn_browse, 0, 2)

        lay.addLayout(grid)

        btn_load = QPushButton("Load hole directory")
        btn_load.clicked.connect(self._load_hole_clicked)
        lay.addWidget(btn_load)

        hint = QLabel(
            "Builds HoleObject via HoleObject.build_from_parent_dir(dir_path).\n"
            "Important: this does NOT change the active/current dataset (po/ro)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        lay.addWidget(hint)

        lay.addStretch(1)
        return w

    def _build_raw_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        lay.addWidget(self._section_label("Raw dataset (manual selection)"))

        lay.addWidget(self._subsection_label("Step 1 — Select headers (required) and metadata (optional)"))

        g1 = QGridLayout()
        g1.setHorizontalSpacing(8)
        g1.setVerticalSpacing(8)

        self.le_data_hdr = self._path_line("DATA .hdr ...")
        self.le_white_hdr = self._path_line("WHITE .hdr ...")
        self.le_dark_hdr = self._path_line("DARK .hdr ...")
        self.le_meta_xml = self._path_line("Metadata .xml (optional) ...")

        b_data = QPushButton("Browse…")
        b_white = QPushButton("Browse…")
        b_dark = QPushButton("Browse…")
        b_meta = QPushButton("Browse…")

        b_data.clicked.connect(lambda: self._browse_file_into(self.le_data_hdr, "Select DATA header (.hdr)", "HDR (*.hdr)"))
        b_white.clicked.connect(lambda: self._browse_file_into(self.le_white_hdr, "Select WHITE header (.hdr)", "HDR (*.hdr)"))
        b_dark.clicked.connect(lambda: self._browse_file_into(self.le_dark_hdr, "Select DARK header (.hdr)", "HDR (*.hdr)"))
        b_meta.clicked.connect(lambda: self._browse_file_into(self.le_meta_xml, "Select metadata (.xml) (optional)", "XML (*.xml)"))

        g1.addWidget(QLabel("DATA header (.hdr):"), 0, 0)
        g1.addWidget(self.le_data_hdr, 0, 1)
        g1.addWidget(b_data, 0, 2)

        g1.addWidget(QLabel("WHITE header (.hdr):"), 1, 0)
        g1.addWidget(self.le_white_hdr, 1, 1)
        g1.addWidget(b_white, 1, 2)

        g1.addWidget(QLabel("DARK header (.hdr):"), 2, 0)
        g1.addWidget(self.le_dark_hdr, 2, 1)
        g1.addWidget(b_dark, 2, 2)

        g1.addWidget(QLabel("Metadata (.xml) (optional):"), 3, 0)
        g1.addWidget(self.le_meta_xml, 3, 1)
        g1.addWidget(b_meta, 3, 2)

        lay.addLayout(g1)

        lay.addWidget(self._hline())
        lay.addWidget(self._subsection_label("Step 2 — Resolve the .raw companions"))

        self.cb_infer_raw = QCheckBox("Infer .raw paths by swapping each selected .hdr suffix to .raw (try this first)")
        self.cb_infer_raw.setChecked(True)
        self.cb_infer_raw.stateChanged.connect(self._update_mangled_raw_enabled)
        lay.addWidget(self.cb_infer_raw)

        g2 = QGridLayout()
        g2.setHorizontalSpacing(8)
        g2.setVerticalSpacing(8)

        self.le_data_raw = self._path_line("DATA raw ...")
        self.le_white_raw = self._path_line("WHITE raw ...")
        self.le_dark_raw = self._path_line("DARK raw ...")

        b_data_raw = QPushButton("Browse…")
        b_white_raw = QPushButton("Browse…")
        b_dark_raw = QPushButton("Browse…")

        b_data_raw.clicked.connect(lambda: self._browse_file_into(self.le_data_raw, "Select DATA raw", "* (*.*)"))
        b_white_raw.clicked.connect(lambda: self._browse_file_into(self.le_white_raw, "Select WHITE raw", "* (*.*)"))
        b_dark_raw.clicked.connect(lambda: self._browse_file_into(self.le_dark_raw, "Select DARK raw", "* (*.*)"))

        g2.addWidget(QLabel("DATA raw file:"), 0, 0)
        g2.addWidget(self.le_data_raw, 0, 1)
        g2.addWidget(b_data_raw, 0, 2)

        g2.addWidget(QLabel("WHITE raw file:"), 1, 0)
        g2.addWidget(self.le_white_raw, 1, 1)
        g2.addWidget(b_white_raw, 1, 2)

        g2.addWidget(QLabel("DARK raw file:"), 2, 0)
        g2.addWidget(self.le_dark_raw, 2, 1)
        g2.addWidget(b_dark_raw, 2, 2)

        lay.addLayout(g2)
        self._update_mangled_raw_enabled()

        btn_load = QPushButton("Load raw dataset")
        btn_load.clicked.connect(self._load_raw_clicked)
        lay.addWidget(btn_load)

        hint = QLabel(
            "If inference fails, uncheck inference and select the three raw files explicitly."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        lay.addWidget(hint)

        lay.addStretch(1)
        return w
    
    def _build_reflectance_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        lay.addWidget(self._section_label("Reflectance ENVI files"))

        head_grid = QGridLayout()
        head_grid.setHorizontalSpacing(8)
        head_grid.setVerticalSpacing(8)

        self.le_ref_envi = self._path_line("Select an ENVI header...")
        btn_browse = QPushButton("Browse folder…")
        btn_browse.clicked.connect(
            lambda: self._browse_file_into(self.le_ref_envi, "Select ENVI header", "JSON (*.hdr)")
        )
        
        head_grid.addWidget(QLabel("Header File:"), 0, 0)
        head_grid.addWidget(self.le_ref_envi, 0, 1)
        head_grid.addWidget(btn_browse, 0, 2)

        lay.addLayout(head_grid)
        
        dat_grid = QGridLayout()
        dat_grid.setHorizontalSpacing(8)
        dat_grid.setVerticalSpacing(8)

        self.le_dat_envi = self._path_line("Select an ENVI binary (.dat, .raw, .bil ...")
        btn_browse = QPushButton("Browse folder…")
        btn_browse.clicked.connect(
            lambda: self._browse_file_into(self.le_dat_envi, "Select ENVI binary", "binary (*)")
            )

        dat_grid.addWidget(QLabel("Data File:"), 0, 0)
        dat_grid.addWidget(self.le_dat_envi, 0, 1)
        dat_grid.addWidget(btn_browse, 0, 2)
        
        lay.addLayout(dat_grid)
        #Sep + label
        lay.addWidget(self._hline())
        lay.addWidget(self._subsection_label("Optional"))
        
        meta_grid = QGridLayout()
        meta_grid.setHorizontalSpacing(8)
        meta_grid.setVerticalSpacing(8)

        self.le_met_file = self._path_line("Select a metadata file ...")
        btn_browse = QPushButton("Browse folder…")
        btn_browse.clicked.connect(
            lambda: self._browse_file_into(self.le_met_file, "Select metadata", "XML (*xml)")
            )

        meta_grid.addWidget(QLabel("Metadata File:"), 0, 0)
        meta_grid.addWidget(self.le_met_file, 0, 1)
        meta_grid.addWidget(btn_browse, 0, 2)
        
        lay.addLayout(meta_grid)
        

        btn_load = QPushButton("Load reflectance")
        btn_load.clicked.connect(self._load_reflectance_clicked)
        lay.addWidget(btn_load)

        hint = QLabel(
            "Builds HoleObject via HoleObject.build_from_parent_dir(dir_path).\n"
            "Important: this does NOT change the active/current dataset (po/ro)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        #lay.addWidget(hint)

        lay.addStretch(1)
        return w
    
    
    
    
    # ----------------------------------------------------------------- NAV

    def _show_menu(self) -> None:
        
        self.stack.setCurrentIndex(0)
        self.btn_back.setEnabled(False)

    def _show_page(self, idx: int) -> None:
        page_names = {
        0: "menu",
        1: "processed", 
        2: "lumo_raw",
        3: "hole",
        4: "raw_manual",
        5: "reflectance"
    }
        logger.info(f"Button clicked: show page {page_names[idx]}")
        self.stack.setCurrentIndex(idx)
        self.btn_back.setEnabled(True)

    def _go_back(self) -> None:
        logger.info("Button clicked: back to menu")
        self._show_menu()

    # -------------------------------------------------------------- LOAD SLOTS

    def _load_processed_clicked(self) -> None:
        jp = (self.le_processed_json.text() or "").strip()
        logger.info(f"Button clicked: Load Processed | path={jp}")
        if not jp:
            self._info("Missing input", "Please select a processed dataset JSON file.")
            return

        try:
            with busy_cursor("Loading processed dataset...", self):
                obj = t.load(Path(jp))
        except Exception as e:
            self._warn("Failed to open processed dataset", str(e))
            logger.error(f"Failed to open processed dataset from {jp}", exc_info=True)
            return

        self.cxt.current = obj
        self.view_flag = "vis"
        logger.info(f"loaded processed data {self.cxt.current.basename}")
        self.accept()

    def _load_lumo_clicked(self) -> None:
        dp = (self.le_raw_dir.text() or "").strip()
        logger.info(f"Button clicked: Load Lumo | path={dp}")
        if not dp:
            self._info("Missing input", "Please select a raw dataset directory.")
            return

        try:
            with busy_cursor("Loading raw dataset...", self):
                obj = t.load(Path(dp))
        except Exception as e:
            self._warn("Failed to open raw dataset", str(e))
            logger.error(f"Failed to open lumo directory from {dp}", exc_info=True)
            return

        self.cxt.current = obj
        self.view_flag = "raw"
        logger.info(f"loaded raw data {self.cxt.current.basename}")
        self.accept()

    def _load_hole_clicked(self) -> None:
        dp = (self.le_hole_dir.text() or "").strip()
        logger.info(f"Button clicked: Load Hole | path={dp}")
        if not dp:
            self._info("Missing input", "Please select a hole directory.")
            return

        try:
            with busy_cursor("Building hole object...", self):
                hole = HoleObject.build_from_parent_dir(Path(dp))
        except Exception as e:
            self._warn("Failed to open hole directory", str(e))
            logger.error(f"Failed to open hole directory from {dp}", exc_info=True)
            return

        # cxt.po, ro and current are not affected by ho update
        self.cxt.ho = hole
        self.view_flag = "hol"
        logger.info(f"loaded hole data {self.cxt.ho.hole_id}")
        self.accept()

    def _load_raw_clicked(self) -> None:
        data_hdr = (self.le_data_hdr.text() or "").strip()
        white_hdr = (self.le_white_hdr.text() or "").strip()
        dark_hdr = (self.le_dark_hdr.text() or "").strip()
        logger.info(f"Button clicked: Load Raw manuaal | paths={data_hdr, white_hdr, dark_hdr}")

        if not (data_hdr and white_hdr and dark_hdr):
            self._info("Missing input", "Please select DATA, WHITE, and DARK header (.hdr) files.")
            return

        meta_xml = (self.le_meta_xml.text() or "").strip()
        meta_path = meta_xml if meta_xml else None

        # Try infer mode
        if self.cb_infer_raw.isChecked():
            try:
                with busy_cursor("Loading raw dataset (infer extension)...", self):
                    raw_obj = RawObject.manual_create_from_multiple_paths(
                        data_hdr, white_hdr, dark_hdr,
                        metadata_path=meta_path
                    )
            except ValueError as e:
                self._warn(
                    "Could not infer extensions",
                    f"{e}\n\nUncheck inference and select the three .raw files explicitly."
                )
                # help the user: auto-switch to explicit mode
                self.cb_infer_raw.setChecked(False)
                logger.error(f"Failed to open raw manually using data header {data_hdr} and inferred paths", exc_info=True)
                return
            except Exception as e:
                self._warn("Failed to open raw dataset", str(e))
                logger.error(f"Failed to open raw manually using data header {data_hdr} and inferred paths", exc_info=True)
                return
            if not self._ensure_required_raw_metadata(raw_obj):
                return
            self.cxt.current = raw_obj
            self.view_flag = "raw"
            logger.info(f"loaded raw data {self.cxt.current.basename}")
            self.accept()
            return

        # Explicit raw mode
        data_raw = (self.le_data_raw.text() or "").strip()
        white_raw = (self.le_white_raw.text() or "").strip()
        dark_raw = (self.le_dark_raw.text() or "").strip()
        logger.info(f"Button clicked: Load Raw manuaal explicit mode | paths={data_hdr, white_hdr, dark_hdr, data_raw, white_raw, dark_raw}")
        if not (data_raw and white_raw and dark_raw):
            self._info(
                "Missing input",
                "Inference is disabled. Please select DATA, WHITE, and DARK raw files."
            )
            return

        try:
            with busy_cursor("Loading mangled raw dataset (explicit raw)...", self):
                raw_obj = RawObject.manual_create_from_critical_paths(
                    data_hdr, data_raw,
                    white_hdr, white_raw,
                    dark_hdr, dark_raw,
                    metadata_path=meta_path
                )
        except Exception as e:
            self._warn("Failed to open raw dataset", str(e))
            logger.error(f"Failed to open raw manually using data header {data_hdr} and explicit paths", exc_info=True)
            return
        
        if not self._ensure_required_raw_metadata(raw_obj):
            return
        
        self.cxt.current = raw_obj
        self.view_flag = "raw"
        logger.info(f"loaded raw data {self.cxt.current.basename}")
        self.accept()
        
        
    def _load_reflectance_clicked(self):
        head = (self.le_ref_envi.text() or "").strip()
        dat = (self.le_dat_envi.text() or "").strip()
        meta_xml = (self.le_meta_xml.text() or "").strip()
        meta_path = meta_xml if meta_xml else None
        logger.info(f"Button clicked: Load Post-processed reflectance | paths={head, dat, meta_path}")
        if not head:
            self._info("Missing input", "Please select a processed dataset JSON file.")
            return

        try:
            with busy_cursor("Loading processed dataset...", self):
                obj = ProcessedObject.load_post_processed_envi(head, dat, meta_path)
        except Exception as e:
            self._warn("Failed to open processed dataset", str(e))
            logger.error(f"Failed to open reflectance using data header {head}", exc_info=True)
            return
        
        if not self._ensure_required_raw_metadata(obj):
            return
        self.cxt.current = obj
        self.view_flag = "vis"
        logger.info(f"loaded processed data {self.cxt.current.basename}")
        self.accept()

    # metadata validation ====================================================


    def _meta_missing(self, meta: dict) -> list[str]:
        """Return list of required keys missing or blank."""
        missing = []
        for k in REQUIRED_META_KEYS:
            v = meta.get(k, "")
            if v is None or str(v).strip() == "":
                missing.append(k)
        return missing    

    def _ensure_required_raw_metadata(self, obj) -> bool:
        """
        Ensure metadata contains required keys.
        Prompts user if missing, and injects results.
    
        Returns True if metadata is satisfied, False if user cancels.
        """
        meta = getattr(obj, "metadata", None)
        if not isinstance(meta, dict):
            meta = {}
    
        missing = self._meta_missing(meta)
        if not missing:
            # already good
            obj.metadata = meta
            return True
    
        # prompt, prefilled with what we have
        dlg = MetadataDialog(meta=meta, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            logging.error("Mandatory metadata not added, loading terminated")
            return False
    
        res = dlg.get_result()
    
        # translate dialog fields -> your canonical metadata keys
        meta["borehole id"] = res.get("hole", "").strip()
        meta["box number"] = res.get("box", "").strip()
        meta["core depth start"] = res.get("depth_from", "").strip()
        meta["core depth stop"] = res.get("depth_to", "").strip()
    
        missing2 = self._meta_missing(meta)
        if missing2:
            QMessageBox.information(
                self,
                "Missing metadata",
                "These fields are required:\n- " + "\n- ".join(missing2)
            )
            logger.info(f"Metadata check | Metadata fields not added")
            return False
    
        obj.metadata = meta
        logger.info(f"Metadata validation: fields_added={missing}")
        return True
        

    # -------------------------------------------------------------- HELPERS

    def _update_mangled_raw_enabled(self) -> None:
        explicit = not self.cb_infer_raw.isChecked()
        for le in (self.le_data_raw, self.le_white_raw, self.le_dark_raw):
            le.setEnabled(explicit)

    def _path_line(self, placeholder: str) -> QLineEdit:
        le = QLineEdit()
        le.setReadOnly(True)
        le.setPlaceholderText(placeholder)
        return le

    def _browse_file_into(self, line_edit: QLineEdit, title: str, filt: str) -> None:
        p, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if p:
            line_edit.setText(p)

    def _browse_dir_into(self, line_edit: QLineEdit, title: str) -> None:
        p = QFileDialog.getExistingDirectory(self, title)
        if p:
            line_edit.setText(p)

    def _warn(self, title: str, msg: str) -> None:
        QMessageBox.warning(self, title, msg)

    def _info(self, title: str, msg: str) -> None:
        QMessageBox.information(self, title, msg)

    def _section_label(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet("font-size: 13px; font-weight: 600;")
        return lab

    def _subsection_label(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet("font-weight: 600; color: #333;")
        return lab

    def _hline(self) -> QFrame:
        ln = QFrame()
        ln.setFrameShape(QFrame.HLine)
        ln.setFrameShadow(QFrame.Sunken)
        return ln

    def accept(self):
        """Called when any load succeeds."""
        logger.info(f"Load dialog accepted | view_flag={self.view_flag}")
        super().accept()
    
    def reject(self):
        """Called when user clicks Cancel or closes dialog."""
        logger.info("Load dialog cancelled")
        super().reject()
# ---------------------------------------------------------------------------
# Standalone demo harness (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QDialog

       

    class DemoWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("LoadDialogue demo")
            self.resize(520, 220)

            self.cxt = CurrentContext()

            layout = QVBoxLayout(self)

            self.info = QLabel(
                "Standalone harness for LoadDialogue.\n\n"
                "After closing the dialog, this will show:\n"
                "- returned view_flag\n"
                "- cxt.active / type(cxt.current)\n"
                "- types of cxt.po / cxt.ro / cxt.ho"
            )
            self.info.setWordWrap(True)

            btn = QPushButton("Open LoadDialogue…")
            btn.clicked.connect(self.open_dialog)

            layout.addWidget(self.info)
            layout.addWidget(btn)

        def open_dialog(self):
            dlg = LoadDialogue(self.cxt, parent=self)
            result = dlg.exec_()

            if result == QDialog.Accepted:
                QMessageBox.information(
                    self,
                    "Dialog accepted",
                    f"view_flag = {dlg.view_flag}\n"
                    f"active = {self.cxt.active}\n\n"
                    f"current = {type(self.cxt.current)}\n"
                    f"po = {type(self.cxt.po)}\n"
                    f"ro = {type(self.cxt.ro)}\n"
                    f"ho = {type(self.cxt.ho)}"
                )
            else:
                QMessageBox.information(self, "Dialog cancelled", "No changes were applied.")

    app = QApplication(sys.argv)
    win = DemoWindow()
    win.show()
    sys.exit(app.exec_())
