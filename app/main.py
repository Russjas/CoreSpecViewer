"""
Entry point and main application window for CoreSpecViewer.

This module defines the `MainRibbonController`, the top-level Qt window that
orchestrates the entire GUI. It owns the global application context
(`CurrentContext`), initialises the ribbon interface, and manages the
stacked set of workflow pages:

    - RawPage          (raw hyperspectral cube viewing)
    - VisualisePage    (processed data, products, clustering, correlation)
    - LibraryPage      (spectral libraries, exemplar selection)
    - HolePage         (multi-box navigation and hole-level operations)

`MainRibbonController` wires every ribbon action to a well-defined controller
method and delegates the actual data operations to the tool layer
(`app.interface.tools`). Each tab switch triggers clean page activation/
teardown so that image tools, selectors and dispatchers never leak between
modes.

This module also provides:
    - File/directory loading for RawObject, ProcessedObject, and HoleObject
    - Centralised saving, “save as”, multi-box saving, and undo/restore
    - Integration of auxiliary windows (CatalogueWindow, InfoTable, Settings)
    - High-level orchestration of masking, cropping, unwrapping, statistics,
      clustering and spectral correlation through the tool dispatcher

Run this module directly via:

    python -m app.main

or call the top-level `main()` function to launch the full CoreSpecViewer GUI.
"""
import sys

import logging
import logging.handlers
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .interface import tools as t
from .models import CurrentContext, HoleObject, RawObject
from .ui.cluster_window import ClusterWindow
from .ui.band_math_dialogue import BandMathsDialog
from .ui.load_dialogue import LoadDialogue
from .ui import (
    AutoSettingsDialog,
    CatalogueWindow,
    Groups,
    GroupedRibbon,
    FlexibleRibbon,
    HolePage,
    InfoTable,
    LibraryPage,
    MetadataDialog,
    LibMetadataDialog,
    RawPage,
    VisualisePage,
    busy_cursor,
    choice_box,
    multi_box,
    two_choice_box,
    WavelengthRangeDialog
)
from .ui.raw_actions import RawActions
from .ui.mask_actions import MaskActions
from .ui.vis_actions import VisActions
from .ui.hole_actions import HoleActions

from .ui.box_ops import BoxOperations
from .ui.display_text import gen_display_text

def setup_logging(log_dir='logs', log_level=logging.INFO):
    """Configure application-wide logging"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    logger = logging.getLogger("app")
    logger.setLevel(log_level)
    
    # File handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / 'corespec.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    except PermissionError:
        # Log file is locked, continue without file logging
        print("Warning: Could not open log file (may be open in another program)")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 

setup_logging()
logger = logging.getLogger(__name__)


class MainRibbonController(QMainWindow):
    """
    Main window that:
      - Hosts the Ribbon (tabs + actions)
      - Hosts a stacked set of Pages (Raw/Mask/Visualise)
      - Delegates every action to the CURRENT page via a clean controller surface
      - Ensures teardown()/activate() on mode switches so tools don't leak
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("CoreSpecViewer")
        
        
        # --- Data shared across modes (filled as user works) ---
        self.cxt = CurrentContext()
        self._catalogue_window = None
        self.cluster_windows: list[ClusterWindow] = []
        
        # --- UI shell: ribbon + stacked pages ---
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)
        # ===== Ribbon operations
        self.ribbon = FlexibleRibbon(self)


        outer.addWidget(self.ribbon, 0)
        #define everpresent actions
        # --- Create actions ---
        self.open_act = QAction("Open", self)
        self.open_act.setShortcut("Ctrl+O")
        self.open_act.triggered.connect(self.load_from_disk)
        self.open_act.setToolTip("Open a raw, processed or hole dataset")
        
        self.save_act = QAction("Save", self)
        self.save_act.setShortcut("Ctrl+S")
        self.save_act.triggered.connect(self.save_clicked)
        self.save_act.setToolTip("Save the current working scan")

        self.save_as_act = QAction("Save As", self)
        self.save_as_act.triggered.connect(self.save_as_clicked)
        self.save_as_act.setToolTip("Save the current working scan in a new location")

        self.undo_act = QAction("Undo", self)
        self.undo_act.setShortcut("Ctrl+Z")
        self.undo_act.triggered.connect(self.undo_unsaved)
        self.undo_act.setToolTip("Removes ALL unsaved data")

        self.multibox_act = QAction("Process Raw Multibox", self)
        self.multibox_act.triggered.connect(self.process_multi_raw)
        self.multibox_act.setToolTip("Select a directory to process All raw data inside")
        
        everpresents = [self.open_act, self.multibox_act, self.save_act, self.save_as_act, self.undo_act]

        self.ribbon.add_global_actions(everpresents)
        #====== non-tab buttons=================
        self.cat_open = QAction('Catalogue Window')
        self.cat_open.triggered.connect(self.show_cat)
        self.info_act = QAction("Info", self)
        self.info_act.setShortcut("Ctrl+I")
        self.info_act.triggered.connect(self.display_info)
        self.settings_act = QAction("Settings", self)
        self.settings_act.triggered.connect(self.on_settings)

        self.ribbon.add_global_actions([self.cat_open, self.info_act, self.settings_act], pos='right')

        # ===== Create all pages==============================================
        self.tabs = QTabWidget(self)
        self.tabs.setTabPosition(QTabWidget.North)   # or South if you prefer
        self.page_list = []
        self.pg_idx_map = {}
        self.raw_page = self.add_page(
            RawPage(self),
            tab_label="Raw",
            key='raw'
        )
        self.vis_page = self.add_page(
            VisualisePage(self),
            tab_label="Visualise",
            key='vis',
            connect_signals={'clusterRequested': self.open_cluster_window}
        )
        self.lib_page = self.add_page(
            LibraryPage(self),
            tab_label="Libraries",
            key='lib'
        )
        self.hol_page = self.add_page(
            HolePage(self),
            tab_label="Hole",
            key='hol',
            connect_signals={'changeView': lambda key: self.choose_view(key, force=True)}
        )
        #============== Action groups==========================================
        self.box_ops = BoxOperations(self.cxt, self)
        self.action_groups = []
        self.raw_actions = self.add_action(RawActions)
        self.mask_actions = self.add_action(MaskActions)
        self.vis_actions = self.add_action(VisActions, box_ops = self.box_ops)
        self.hole_actions = self.add_action(HoleActions, box_ops = self.box_ops)
           
        
        
        #ensure pgs have correct context at start
        self.lib_page._find_default_database()
        self._distribute_context()
                
        self._last_tab_idx = self.tabs.currentIndex()

        self.tabs.currentChanged.connect(self._on_tab_changed)
        outer.addWidget(self.tabs, 1)

        # Initial mode
        self.raw_page.activate()

        # Populate ribbon & connect mode switching
        #self._init_ribbon()

        self.statusBar().showMessage("Ready.")
    # =============== Add page convenience ====================================
    
    def add_page(self, page_instance, tab_label, key, connect_signals=None):
        """
        Registers a specific page instance to self.tabs, self.page_list and self.pg_idx_map.
        Also registers any signals and slots specific to the page instance
        
        """
        # Add to page_list for context distribution
        self.page_list.append(page_instance)
        tab_index = self.tabs.addTab(page_instance, tab_label)
        self.pg_idx_map[key] = tab_index
        page_instance.cxt = self.cxt # -> might be redundant as there is a distribute context call
        # Connect any signals if specified
        if connect_signals:
            for signal_name, callback in connect_signals.items():
                signal = getattr(page_instance, signal_name, None)
                if signal:
                    signal.connect(callback)    
        return page_instance


    def add_action(self, Act, box_ops = None):
        if box_ops is None:
            act_instance = Act(self.cxt, self.ribbon, parent = self)
        else:
            act_instance = Act(self.cxt, self.ribbon, parent = self, box_ops = box_ops)
        self.action_groups.append(act_instance)
        return act_instance


    #======== UI methods ===============================================
    def _clear_all_canvas_refs(self):
        """Clear memmap references from all page canvases before saving."""
        for page in self.page_list:
            # Clear left canvas (SpectralImageCanvas)
            if hasattr(page, 'left_canvas') and hasattr(page.left_canvas, 'clear_memmap_refs'):
                page.left_canvas.clear_memmap_refs()

            # Clear right canvas (ImageCanvas2D)
            if hasattr(page, 'right_canvas') and hasattr(page.right_canvas, 'clear_memmap_refs'):
                page.right_canvas.clear_memmap_refs()  
    
    
    def update_display(self, key = 'mask'):
        p = self.active_page()
        p.update_display(key = key)


    def _on_tab_changed(self, new_idx: int):
        """Handles user-initiated tab changes."""

        # teardown old (the one that just lost focus)
        old_idx = getattr(self, "_last_tab_idx", -1)
        if 0 <= old_idx < self.tabs.count():
            old = self.tabs.widget(old_idx)
            if hasattr(old, "teardown"): old.teardown()

        # activate new
        new = self.tabs.widget(new_idx)
        self._distribute_context()
        if hasattr(new, "activate"): new.activate()
        self._last_tab_idx = new_idx
        self.update_display()


    def _distribute_context(self):
        for pg in self.page_list:
            pg.cxt = self.cxt
        for act in self.action_groups:
            act.cxt = self.cxt
        self.box_ops.cxt = self.cxt

    def active_page(self):
        return self.tabs.currentWidget()

    def choose_view(self, key= 'raw', force = False):

        new_idx =  self.pg_idx_map[key]
        old_idx = getattr(self, "_last_tab_idx", -1)
        if new_idx == old_idx:
            self._distribute_context()

        if old_idx == self.pg_idx_map['hol'] and not force:
                self._distribute_context()
                self.update_display()
                return
        self.tabs.setCurrentIndex(self.pg_idx_map[key])


# convenience method exposed to Action classes

    def refresh(self, view_key: str = None, dataset_key: str = 'mask'):
        """
        Refresh context and UI after data operations.
        
        Args:
            view_key: Optional view to switch to ('raw', 'vis', 'lib', 'hol')
            dataset_key: Dataset key to display (default: 'mask')
        """
        self._distribute_context()
        if view_key:
            self.choose_view(view_key)
        self.update_display(key=dataset_key)

#================= Global actions========================================

    def show_cat(self):
        logger.info('Button clicked: Catalogue pane viewed')
        if self._catalogue_window is None:
            self._catalogue_window = CatalogueWindow(
                parent=self,
                name_filters=["*.json", "*.hdr"],
            )

            self._catalogue_window.fileActivated.connect(
                self.on_catalogue_activated
            )
            self._catalogue_window.dirActivated.connect(
                self.on_catalogue_activated
            )

        self._catalogue_window.show()
        self._catalogue_window.raise_()
        self._catalogue_window.activateWindow()

    def display_info(self):
        logger.info('Button clicked: Metadata viewed')
        self.table_window = InfoTable()
        if self.cxt is not None and self.cxt.current is not None:
            self.table_window.set_from_dict(self.cxt.current.metadata)
        self.table_window.setWindowTitle("Info Table")
        self.table_window.resize(400, 300)
        self.table_window.show()


    def on_settings(self):
        logger.info('Button clicked: Config settings viewed')
        dlg = AutoSettingsDialog(self)
        if dlg.exec_():
            # user clicked Save; propagate lightweight refresh
            self._distribute_context()   # keep pages in sync with any config change
            try:
                self.update_display()    # redraw active page safely
            except Exception:
                pass
            self.statusBar().showMessage("Settings updated.", 3000)

#================= Everpresent actions =====================================
    def on_catalogue_activated(self, path):
        logger.info('Button clicked: Catalogue window viewed')
        if not path:
            return
        with busy_cursor(self, 'Loading....'):
            try:
                loaded_obj = t.load(path)
                if loaded_obj.is_raw:
                    self.cxt.current = loaded_obj
                    self.choose_view('raw')
                    self.update_display()
                    logger.info(f"loaded raw data {self.cxt.current.basename} from catalogue")
                else:
                    self.cxt.current = loaded_obj
                    self.choose_view('vis')
                    self.update_display()
                    logger.info(f"loaded processed data {self.cxt.current.basename} from catalogue")
                return
            except Exception as e:
                
                try:
                    hole = HoleObject.build_from_parent_dir(path)
                    
                    self.cxt.ho = hole
                    self._distribute_context()
                    self.choose_view('hol')
                    self.update_display()
                    logger.info(f"loaded hole {self.cxt.ho.hole_id} from catalogue")
                    return
                except Exception as e:
                    QMessageBox.warning(self, "Open dataset", f"Failed to open hole dataset: {e}")
                    logger.warning(f"failed to open datasets from {path} provided", exc_info=True)
                    return
                    return


    def load_from_disk(self):
        logger.info("load dialogue opened")
        dlg = LoadDialogue(self.cxt, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.choose_view(dlg.view_flag)
            self.update_display()


    def process_multi_raw(self):
        logger.info("Button clicked: multi-box processing")
        multi_box.run_multibox_dialog(self)


    def save_clicked(self):
        logger.info("Button clicked: Save")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            QMessageBox.information(self, "Save", msg)
            return
        if self.cxt.current.has_temps:
            test = two_choice_box('Commit changes before saving?', 'yes', 'no')

            if test == 'left':

                self.cxt.po.commit_temps()
                logger.info(f"Button clicked: Commit temps for {self.cxt.po.basename}")

        wants_prompt = True
        if self.cxt.current.datasets:
            wants_prompt = not any(ds.path.exists() for ds in self.cxt.po.datasets.values())

        if wants_prompt:
            dest = QFileDialog.getExistingDirectory(self, "Choose save folder", str(self.cxt.current.root_dir))
            if not dest:
                return
            self.cxt.current.update_root_dir(dest)  # rewires every dataset path to the chosen folder
            logger.info(f"Save location updated to {self.cxt.current.root_dir}")
        try:
            with busy_cursor('saving...', self):
                self.cxt.current.save_all()
                self.cxt.current.reload_all()
                self.cxt.current.load_thumbs()
                self.update_display()
                logger.info(f"Saved {self.cxt.current.basename}")
        except Exception as e:
            logger.error(f"Failed to save dataset {self.cxt.current.basename}", exc_info=True)
            QMessageBox.warning(self, "Save dataset", f"Failed to save dataset: {e}")
            return


    def save_as_clicked(self):
        logger.info("Button clicked: Save As")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            QMessageBox.information(self, "Save As", msg)
            return
        if self.cxt.current.has_temps:
            test = two_choice_box('Commit changes before saving?', 'yes', 'no')
            if test == 'left':
                self.cxt.po.commit_temps()
                logger.info(f"Button clicked: Commit temps for {self.cxt.po.basename}")
        dest = QFileDialog.getExistingDirectory(self, "Choose save folder", str(self.cxt.current.root_dir))
        if not dest:
            return
        self.cxt.current.update_root_dir(dest)  # rewires every dataset path to the chosen folder
        logger.info(f"Save location updated to {self.cxt.current.root_dir}")
        try:
            with busy_cursor('saving...', self):
                self.cxt.current.build_all_thumbs()
                self.cxt.current.save_all_thumbs()
                self.cxt.po.save_all(new=True)
                logger.info(f"Saved {self.cxt.current.basename}")
        except Exception as e:
            logger.error(f"Failed to save dataset {self.cxt.current.basename}", exc_info=True)
            QMessageBox.warning(self, "Save dataset", f"Failed to save dataset: {e}")
            return


    def undo_unsaved(self):
        logger.info(f"Button clicked: Undo")
        valid_state, msg = self.cxt.requires(self.cxt.SCAN)
        if not valid_state:
            logger.warning(msg)
            QMessageBox.information(self, "Undo", msg)
            return
        self.cxt.current = t.reset(self.cxt.current)
        logger.info(f"{self.cxt.current.basename} temp datasets cleared")
        self._distribute_context()
        self.update_display()


#----------- manage ClusterWindow for interogating cluster centres-------------
    def open_cluster_window(self, cluster_key: str):
        """
        Create a ClusterWindow for the given *CLUSTERS dataset key and
        show it as a standalone window.
    
        Pinned to whatever self.cxt.current is at the moment of opening.
        """
        logger.info(f"Button clicked: View cluster centres")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            QMessageBox.information(self, "Save hole", msg)
            return
        po = self.cxt.current
           
        # NEW: Use factory method instead of direct constructor
        win = ClusterWindow.from_processed_object(
            parent=self,
            cxt=self.cxt,
            po=po,
            cluster_key=cluster_key,
        )
        
        win.setWindowFlag(Qt.Window, True)
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        win.setWindowTitle(gen_display_text(cluster_key))
        self.cluster_windows.append(win)
    
        win.destroyed.connect(
            lambda _obj=None, w=win: self._on_cluster_window_destroyed(w)
        )
    
        # ---- Half-screen-ish sizing ----
        main_geo = self.geometry()
        half_width = max(400, main_geo.width() // 2)
    
        # Resize to half width, full height
        win.resize(half_width, main_geo.height())
    
        # Move it to the left side of the main window
        win.move(main_geo.x(), main_geo.y())
    
        win.activate()
        win.show()
        win.raise_()
    
    def _on_cluster_window_destroyed(self, win: ClusterWindow):
        logger.info("Button clicked: close cluster window")
        try:
            self.cluster_windows.remove(win)
        except ValueError:
            pass
        

def main():
    logger.info("CoreSpecViewer starting...")
    app = QApplication(sys.argv)
    win = MainRibbonController()
    win.showMaximized()  
    sys.exit(app.exec())
    