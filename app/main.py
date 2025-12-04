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
from .ui import (
    AutoSettingsDialog,
    CatalogueWindow,
    Groups,
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

feature_keys = [
    '1400W', '1480W', '1550W', '1760W', '1850W',
    '1900W', '2080W', '2160W', '2200W', '2250W',
    '2290W', '2320W', '2350W', '2390W', '2950W',
    '2950AW', '2830W', '3000W', '3500W', '4000W',
    '4000WIDEW', '4470TRUEW', '4500SW', '4500CW',
    '4670W', '4920W', '4000V_NARROWW', '4000shortW', '2950BW'
]


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
        self.resize(1400, 900)
        
        # --- Data shared across modes (filled as user works) ---
        self.cxt = CurrentContext()
        self._catalogue_window = None
        self.cluster_windows: list[ClusterWindow] = []
        self.legend_mapping_path = None

        # --- UI shell: ribbon + stacked pages ---
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)
        # ===== Ribbon operations
        self.ribbon = Groups(self)


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


        self.raw_page = RawPage(self)
        self.vis_page = VisualisePage(self)
        self.lib_page = LibraryPage(self)
        self.hol_page = HolePage(self)

        self.hol_page.changeView.connect(lambda key: self.choose_view(key, force=True))
        self.vis_page.clusterRequested.connect(self.open_cluster_window)

        self.page_list = [self.raw_page, self.vis_page, self.lib_page, self.hol_page]
        #ensure pgs have correct context at start
        self._distribute_context()
        self.lib_page._find_default_database()

        self.tabs.addTab(self.raw_page, "Raw")
        self.tabs.addTab(self.vis_page, "Visualise")
        self.tabs.addTab(self.lib_page, "Libraries")
        self.tabs.addTab(self.hol_page, "Hole")
        self.pg_idx_map = {'raw': 0, 'vis': 1, 'lib': 2, 'hol': 3}

        self._last_tab_idx = self.tabs.currentIndex()

        self.tabs.currentChanged.connect(self._on_tab_changed)
        outer.addWidget(self.tabs, 1)

        # Initial mode

        self.raw_page.activate()

        # Populate ribbon & connect mode switching
        self._init_ribbon()

        self.statusBar().showMessage("Ready.")

    # =============== Ribbon population ====================
    def _init_ribbon(self):
        """Create actions for each tab; all callbacks route to controller methods."""
        # --- RAW TAB ---
        self.ribbon.add_tab('Raw',[
            ("button", "Auto Crop", self.automatic_crop, "Faster on Raw than Processed data.\nUses image analysis to automatically detect core box - NB. is very flaky"),
            ("button", "Crop",        self.crop_current_image, "Faster on Raw than Processed data.\nManually crop the image"),
            ("button", "Process",     self.process_raw, "Produce a processed dataset from this raw dataset"),
        ])

        # --- MASK TAB ---
        self.ribbon.add_tab('Masking', [
            ("button", "New mask", lambda: self.act_mask_point('new'), "Creates a blank mask,\n then masks by correlation with selected pixel."),
            ("button", "Enhance mask", lambda: self.act_mask_point('enhance'), "Adds to existing mask by correlation with selected pixel"),
            ("button", "Mask line", lambda: self.act_mask_point('line'), "Adds a masked vertical line to existing mask"),
            ("button", "Mask region", self.act_mask_rect, "Adds a masked rectangle to existing mask"),
            ("menu",   "Freehand mask region", [
                ("Mask outside selected", lambda: self.act_mask_polygon(mode = "mask outside"), "With exising mask, masks all pixels outside of selected region"),
                ("Mask inside selected", lambda: self.act_mask_polygon(mode = "mask inside"), "With exising mask, masks all pixels outside of selected region")
            ]),
            ("button", "Despeckle", self.despeck_mask, "Remove speckles from mask"),
            ("button", "Improve", self.act_mask_improve, "Heuristically improves the mask"),
            ("button", "Calc stats", self.act_mask_calc_stats, "Calculates connected components used for downhole unwrapping"),
            ("button", "unwrap preview", self.unwrap, 'Produces "unwrapped" coreboxes by vertical concatenation: Right→Left, Top→Bottom')
        ])

        # --- VISUALISE TAB ---
        self.extract_feature_list = []
        for key in feature_keys:
            self.extract_feature_list.append((key, lambda _, k=key: self.run_feature_extraction(k)))
        self.ribbon.add_tab('Visualise', [
            ("button", "Quick Cluster", self.act_kmeans, "Performs unsupervised k-means clustering"),
            ("menu",   "Correlation", [
                ("MineralMap Pearson (Winner-takes-all)", self.act_vis_correlation, "Performs Pearson correlation against selected collection from the library"),
                ("MineralMap SAM (Winner-takes-all)", self.act_vis_sam, "Performs Spectral Angle Mapping against selected collection from the library"),
                ("MineralMap MSAM (Winner-takes-all)", self.act_vis_msam, "Performs Modified Spectral Angle Mapping against selected collection from the library"),
                ("Multi-range check (Winner-takes-all)", self.act_vis_multirange, "Performs custom multi-window matching"),
                ("select range", self.act_subrange_corr, "Performs correlation on a chosed wavelength range"),
                ("Re-map legends", self._remap_legends)
                
                
               ]),
            ("menu",   "Features", self.extract_feature_list, "Performs Minimum Wavelength Mapping"),
            ("button", "Generate Images", self.gen_images, "Generates full size images of all products and base datasets in an outputs folder"),
            ("menu", "Library building", [
                ("Add spectra", self.act_lib_pix, "Add a single pixel spectra to the current library\n WARNING: This will modify the library on disk, use a back up"),
                ("Add region average", self.act_lib_region, "Add the average spectra of a region to the current library\n WARNING: This will modify the library on disk, use a back u"),
                ])
            ])
        
        # --- HOLE TAB ---
        self.extract_feature_list_multi = []
        for key in feature_keys:
            self.extract_feature_list_multi.append((key, lambda _, k=key: self.run_feature_extraction(k, multi=True)))
        self.ribbon.add_tab('Hole operations',[
                            ("button", "Previous", self.hole_prev_box, "View previous box in hole"),
                            ("button", "Next", self.hole_next_box, "View next box in hole"),
                            ("button", "Return to Raw", self.hole_return_to_raw, "Open the raw dataset to replace this box"),
                            ("button", "Quick Cluster", lambda: self.act_kmeans(multi = True)),
                            ("menu",   "Fullhole Correlations", [
                                ("MineralMap Pearson (Winner-takes-all)", lambda: self.act_vis_correlation(multi=True)),
                                ("MineralMap SAM (Winner-takes-all)", lambda: self.act_vis_sam( multi=True)),
                                ("MineralMap MSAM (Winner-takes-all)", lambda: self.act_vis_msam(multi=True)),
                                ("Multi-range check (Winner-takes-all)", lambda: self.act_vis_multirange(multi = True), "Performs custom multi-window matching"),
                                ("select range", lambda: self.act_subrange_corr(multi = True), "Performs correlation on a chosed wavelength range"),
                               ]),
                            ("menu",   "Fullhole Features", self.extract_feature_list_multi),
                            ("button", "Save All", self.save_all_changes),
                            ("button", "Generate Images", lambda: self.gen_images(multi = True))
                            ])

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
        p = self._active_page()
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

    def _active_page(self):
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


#================= Global actions========================================

    def show_cat(self):
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
        print('info button clicked')
        self.table_window = InfoTable()
        if self.cxt is not None and self.cxt.current is not None:
            self.table_window.set_from_dict(self.cxt.current.metadata)
        self.table_window.setWindowTitle("Info Table")
        self.table_window.resize(400, 300)
        self.table_window.show()


    def on_settings(self):
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
        if not path:
            return
        with busy_cursor(self, 'Loading....'):
            try:
                loaded_obj = t.load(path)
                if loaded_obj.is_raw:
                    self.cxt.current = loaded_obj
                    self.choose_view('raw')
                    self.update_display()
                else:
                    self.cxt.current = loaded_obj
                    self.choose_view('vis')
                    self.update_display()
                return
            except Exception as e:
                print(path,e)
                try:
                    hole = HoleObject.build_from_parent_dir(path)
                    print('hole loaded')
                    self.cxt.ho = hole
                    self._distribute_context()
                    self.choose_view('hol')
                    self.update_display()
                    return
                except Exception as e:
                    print(path,e)
                    return


    def load_from_disk(self):
        '''loads PO or RO only, HO and db are loaded from control panel on
        thei respective games'''
        clicked_button = choice_box( "What would you like to open?", ["Processed dataset", "Raw directory", "Hole directory", "Mangled Dataset"])

        if clicked_button is None:
            return
        
        if clicked_button == 3:
            data_head_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data header file", "", "header files (*.hdr)")
            if not data_head_path:
                return
            white_head_path, _ = QFileDialog.getOpenFileName(
            self, "Open White header file", "", "header files (*.hdr)")
            if not white_head_path:
                return
            dark_head_path, _ = QFileDialog.getOpenFileName(
            self, "Open Dark header file", "", "header files (*.hdr)")
            if not dark_head_path:
                return
            metadata_path, _ = QFileDialog.getOpenFileName(
            self, "Optional lumo metadata", "", "header files (*.xml)")
            try:
                self.cxt.current = RawObject.manual_create_from_multiple_paths(data_head_path, 
                                                                               white_head_path, 
                                                                               dark_head_path, 
                                                                               metadata_path = metadata_path)
                self.choose_view('raw')
                self.update_display()
                return
            except ValueError as e:
                QMessageBox.warning(self, "Open dataset", f"Failed to open dataset: {e}/n manually add raw paths if file names are inconsistent")
                data_raw_path, _ = QFileDialog.getOpenFileName(
                self, "Open Data raw", "", "raw files (*)")
                if not data_raw_path:
                    return
                white_raw_path, _ = QFileDialog.getOpenFileName(
                self, "Open White raw", "", "raw files (*)")
                if not white_raw_path:
                    return
                dark_raw_path, _ = QFileDialog.getOpenFileName(
                self, "Open Dark raw", "", "raw files (*)")
                if not dark_raw_path:
                    return
                try:
                    self.cxt.current = RawObject.manual_create_from_critical_paths(data_head_path,
                                                                                data_raw_path,
                                                                                white_head_path,
                                                                                white_raw_path,
                                                                                dark_head_path,
                                                                                dark_raw_path,
                                                                                metadata_path= metadata_path) 
                    self.choose_view('raw')
                    self.update_display()
                    return
                except ValueError as e:
                    QMessageBox.warning(self, "Open dataset", f"Failed to open dataset: {e}")
                    return
                    
        
        
        
        if clicked_button == 1 or clicked_button == 2:
            path = QFileDialog.getExistingDirectory(
                       self,
                       "Select directory",
                       "",
                       QFileDialog.ShowDirsOnly
                       )
            if not path:
                return
        elif clicked_button == 0:
            path, _ = QFileDialog.getOpenFileName(
            self, "Open JSON Metadata", "", "JSON files (*.json)")
            if not path:
                return
        try:
            with busy_cursor('loading...', self):
                if clicked_button == 2:
                    hole = HoleObject.build_from_parent_dir(path)
                    print('hole loaded')
                    self.cxt.ho = hole
                    self._distribute_context()
                    self.choose_view('hol')
                    self.update_display()
                    return
                else:
                    loaded_obj = t.load(path)
                if loaded_obj.is_raw:
                    self.cxt.current = loaded_obj
                    self.choose_view('raw')
                    self.update_display()
                else:
                    self.cxt.current = loaded_obj
                    self.choose_view('vis')
                    self.update_display()
        except Exception as e:
            QMessageBox.warning(self, "Open dataset", f"Failed to open dataset: {e}")
            return


    def process_multi_raw(self):
        multi_box.run_multibox_dialog(self)


    def gen_images(self, multi = False):
        if self.cxt is None:
            return
        if multi:
            if self.cxt.ho is None:
                return
            for po in self.cxt.ho:
                self.cxt.current.save_all()
                po.export_images()
                self.cxt.current.reload_all()
                self.cxt.current.load_thumbs()
            print('multi')
        if self.cxt.po is None or self.cxt.current.is_raw:
            return
        self.cxt.current.export_images()
        print('single')


    def save_clicked(self):

        if self.cxt.current.is_raw:
            QMessageBox.information(self, "save", "Raw data must be processed prior to saving")
            return
        if self.cxt.current.has_temps:
            test = two_choice_box('Commit changes before saving?', 'yes', 'no')

            if test == 'left':

                self.cxt.po.commit_temps()

        wants_prompt = True
        if self.cxt.current.datasets:
            wants_prompt = not any(ds.path.exists() for ds in self.cxt.po.datasets.values())

        if wants_prompt:
            dest = QFileDialog.getExistingDirectory(self, "Choose save folder", str(self.cxt.current.root_dir))
            if not dest:
                return
            self.cxt.current.update_root_dir(dest)  # rewires every dataset path to the chosen folder
        try:
            with busy_cursor('saving...', self):
                self.cxt.current.save_all()
                self.cxt.current.reload_all()
                self.cxt.current.load_thumbs()
                self.update_display()
        except Exception as e:
           QMessageBox.warning(self, "Save dataset", f"Failed to save dataset: {e}")
           return


    def save_as_clicked(self):
        if self.cxt.po is None:
            QMessageBox.information(self, "save", "Raw data must be processed prior to saving")
            return
        if self.cxt.current.has_temps:
            test = two_choice_box('Commit changes before saving?', 'yes', 'no')
            if test == 'left':
                self.cxt.po.commit_temps()

        dest = QFileDialog.getExistingDirectory(self, "Choose save folder", str(self.cxt.current.root_dir))
        if not dest:
            return
        self.cxt.current.update_root_dir(dest)  # rewires every dataset path to the chosen folder
        try:
            with busy_cursor('saving...', self):
                self.cxt.current.build_all_thumbs()
                self.cxt.current.save_all_thumbs()
                self.cxt.po.save_all(new=True)
        except Exception as e:
            QMessageBox.warning(self, "Save dataset", f"Failed to save dataset: {e}")
            return


    def undo_unsaved(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Undo", "No Current Scan")
            return
        self.cxt.current = t.reset(self.cxt.current)
        self._distribute_context()
        self.update_display()

    # -------- RAW actions --------

    def crop_current_image(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        p = self._active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return

        # Ask the page to collect a rectangle and pass back coords
        def _on_rect(y0, y1, x0, x1):
            try:
                with busy_cursor('cropping...', self):
                    self.cxt.current = t.crop(self.cxt.current, y0, y1, x0, x1)
                    self._distribute_context()
                    self.update_display()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()

    def automatic_crop(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Cropping", "No Current Scan")
            return
        with busy_cursor('cropping...', self):
            self.cxt.current = t.crop_auto(self.cxt.current)
        self._distribute_context()
        self.update_display()


    def process_raw(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        if not self.cxt.current.is_raw:
            QMessageBox.information(self, "Process", "Load a raw dataset first.")
            return
        if (
        not self.cxt.current.metadata.get('borehole id')
        or not self.cxt.current.metadata.get('box number')
        or not self.cxt.current.metadata.get('core depth start')
        or not self.cxt.current.metadata.get('core depth stop')
        ):
            dlg = MetadataDialog(self.cxt.current.metadata, parent=self)
            if dlg.exec() == QDialog.Accepted:
                result = dlg.get_result()
                self.cxt.current.metadata['borehole id'] = result['hole']
                self.cxt.current.metadata['box number'] = result['box']
                self.cxt.current.metadata['core depth start'] = result['depth_from']
                self.cxt.current.metadata['core depth stop'] = result['depth_to']
        try:
            with busy_cursor('processing...', self):

                self.cxt.po = self.cxt.current.process()


        except Exception as e:
            QMessageBox.warning(self, "Process", f"Failed to process/save: {e}")
            return

        self.choose_view('vis')
        self.update_display()
        self.statusBar().showMessage("Processed saved")


    # -------- MASK actions --------

    def act_mask_rect(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Masking", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Mask region", "Open a processed dataset first.")
            return
        p = self._active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return

        def _on_rect(y0, y1, x0, x1):
            try:
                self.cxt.current = t.mask_rect(self.cxt.current, y0, y1, x0, x1 )
                self._distribute_context()
                self.update_display()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()

    def act_mask_point(self, mode):
        if self.cxt.current is None:
            QMessageBox.information(self, "Masking", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Mask region", "Open a processed dataset first.")
            return
        p = self._active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return

        def handle_point_click(y, x):
            try:
                with busy_cursor('trying mask correlation...', self):
                    self.cxt.current = t.mask_point(self.cxt.current, mode, y, x)
                self._distribute_context()
                self.update_display()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_single_click(handle_point_click)


    def act_mask_improve(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Masking", "No Current Scan")
            return
        self.cxt.current = t.improve_mask(self.cxt.current)
        self._distribute_context()
        self.update_display()

    def despeck_mask(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Masking", "No Current Scan")
            return
        self.cxt.current = t.despeckle_mask(self.cxt.current)
        self._distribute_context()
        self.update_display()

    def act_mask_polygon(self, mode = "mask outside"):
        if self.cxt.current is None:
            QMessageBox.information(self, "Masking", "No Current Scan")
            return
        p = self._active_page()
        if not p or not p.dispatcher or self.cxt.current is None:
            return
        def _on_finish(vertices_rc):
            self.cxt.current = t.mask_polygon(self.cxt.current, vertices_rc, mode = mode)
            self._distribute_context()
            self.update_display()
            p.dispatcher.clear_all_temp()
        p.dispatcher.set_polygon(_on_finish, temporary=True)
        p.left_canvas.start_polygon_select()


    def act_mask_calc_stats(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Stats", "Open a processed dataset first.")
            return
        self.cxt.current = t.calc_unwrap_stats(self.cxt.current)

        self._distribute_context()
        self.update_display(key = 'segments')


    def unwrap(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return

        if not self.cxt.current.has('stats'):
            QMessageBox.warning(self, "Warning", "No stats calculated yet.")
            return
        with busy_cursor('unwrapping...', self):
            self.cxt.current = t.unwrapped_output(self.cxt.current)
        self._distribute_context()
        self.update_display(key='DholeAverage')

    # -------- VISUALISE actions --------
    
    def ask_collection_name(self):
        if not self.cxt.library:
            return None
        names = sorted(self.cxt.library.collections.keys())
        if not names:
            QMessageBox.information(self, "No collections", "Create a collection first via 'Add Selected → Collection'.")
            return None
        if len(names) == 1:
            return names[0]
        name, ok = QInputDialog.getItem(self, "Select Collection", "Collections:", names, 0, False)
        return name if ok else None
    
    
    def run_feature_extraction(self, key, multi = False):
        if multi:
            if self.cxt.ho is None: 
                return
            with busy_cursor('feature extraction {key}....', self):
                for po in self.cxt.ho:
                    t.run_feature_extraction(po, key)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                
                self.choose_view('hol')
                self.update_display()
            return
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        print(key)
        with busy_cursor(f'extracting {key}...', self):
            self.cxt.current = t.run_feature_extraction(self.cxt.current, key)
        
        self.choose_view('vis')
        self.update_display()


    def act_vis_correlation(self, multi = False):
        if multi:
            if self.cxt.ho is None: 
                QMessageBox.information(self, "Correlation", "No Hole dataset loaded for multibox operations")
                return
            name = self.ask_collection_name()
            if not name:
                return
            exemplars = self.cxt.library.get_collection_exemplars(name)
            if not exemplars:
                return
            with busy_cursor('correlation...', self):
                for po in self.cxt.ho:
                    t.wta_min_map(po, exemplars, name)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    
            self.choose_view('hol')
            self.update_display()
            return
        #======================================================================
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Correlation", "Open a processed dataset first.")
            return
        name = self.ask_collection_name()
        if not name:
            return
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            return
        with busy_cursor('correlation...', self):
            self.cxt.current = t.wta_min_map(self.cxt.current, exemplars, name)

        self.choose_view('vis')
        self.update_display()


    def act_vis_sam(self, multi = False):
        if multi:
            if self.cxt.ho is None: 
                QMessageBox.information(self, "Correlation", "No Hole dataset loaded for multibox operations")
                return
            name = self.ask_collection_name()
            if not name:
                return
            exemplars = self.cxt.library.get_collection_exemplars(name)
            if not exemplars:
                return
            with busy_cursor('correlation...', self):
                for po in self.cxt.ho:
                    t.wta_min_map_SAM(po, exemplars, name)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    
            self.choose_view('hol')
            self.update_display()
            return
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Correlation", "Open a processed dataset first.")
            return
        name = self.ask_collection_name()
        if not name:
            return
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            return
        with busy_cursor('correlation...', self):
            self.cxt.current = t.wta_min_map_SAM(self.cxt.current, exemplars, name)

        self.choose_view('vis')
        self.update_display()

    def act_vis_multirange(self, multi = False):
        modes = ['pearson', 'sam', 'msam']
        if multi:
            if self.cxt.ho is None: 
                QMessageBox.information(self, "Correlation", "No Hole dataset loaded for multibox operations")
                return
            name = self.ask_collection_name()
            if not name:
                return
            exemplars = self.cxt.library.get_collection_exemplars(name)
            if not exemplars:
                return
            mode, ok = QInputDialog.getItem(self, "Select Match Mode", "Options:", modes, 0, False)
            if not ok or not mode:
                return
            with busy_cursor('correlation...', self):
                for po in self.cxt.ho:
                    t.wta_multi_range_minmap(po, exemplars, name, mode=mode)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    
            self.choose_view('hol')
            self.update_display()
            return
        
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Correlation", "Open a processed dataset first.")
            return
        name = self.ask_collection_name()
        if not name:
            return
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            return
        
        mode, ok = QInputDialog.getItem(self, "Select Match Mode", "Options:", modes, 0, False)
        if not ok or not mode:
            return
        with busy_cursor('correlation...', self):
            self.cxt.current = t.wta_multi_range_minmap(self.cxt.current, exemplars, name, mode=mode)

        self.choose_view('vis')
        self.update_display()
        
        
    def act_subrange_corr(self, multi = False):
        modes = ['pearson', 'sam', 'msam']
        if multi:
            if self.cxt.ho is None: 
                QMessageBox.information(self, "Correlation", "No Hole dataset loaded for multibox operations")
                return
            name = self.ask_collection_name()
            if not name:
                return
            exemplars = self.cxt.library.get_collection_exemplars(name)
            if not exemplars:
                return
            mode, ok = QInputDialog.getItem(self, "Select Match Mode", "Options:", modes, 0, False)
            if not ok or not mode:
                return
            ok, start_nm, stop_nm = WavelengthRangeDialog.get_range(
                parent=self,
                start_default=0,
                stop_default=20000,
            )
            if not ok:
                return
            with busy_cursor('correlation...', self):
                for po in self.cxt.ho:
                    try:
                        t.wta_min_map_user_defined(po, exemplars, name, [start_nm, stop_nm], mode=mode)
                        po.commit_temps()
                        po.save_all()
                        po.reload_all()
                        po.load_thumbs()
                        print('done one')
                    except ValueError:
                        continue
                    
            self.choose_view('hol')
            self.update_display()
            return
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Correlation", "Open a processed dataset first.")
            return
        name = self.ask_collection_name()
        if not name:
            return
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            return
        
        mode, ok = QInputDialog.getItem(self, "Select Match Mode", "Options:", modes, 0, False)
        if not ok or not mode:
            return
        ok, start_nm, stop_nm = WavelengthRangeDialog.get_range(
            parent=self,
            start_default=0,
            stop_default=20000,
        )
        if not ok:
            return
        with busy_cursor('correlation...', self):
            try:
                self.cxt.current = t.wta_min_map_user_defined(self.cxt.current, exemplars, name, [start_nm, stop_nm], mode=mode)
            except Exception as e:
                QMessageBox.warning(self, "Failed operation", f"Failed to use band range: {e}")
                return

        self.choose_view('vis')
        self.update_display()

    def act_vis_msam(self, multi = False):
        if multi:
            if self.cxt.ho is None: 
                QMessageBox.information(self, "Correlation", "No Hole dataset loaded for multibox operations")
                return
            name = self.ask_collection_name()
            if not name:
                return
            exemplars = self.cxt.library.get_collection_exemplars(name)
            if not exemplars:
                return
            with busy_cursor('correlation...', self):
                for po in self.cxt.ho:
                    t.wta_min_map_MSAM(po, exemplars, name)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    
            self.choose_view('hol')
            self.update_display()
            return
        print('this was called')
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Correlation", "Open a processed dataset first.")
            return
        name = self.ask_collection_name()
        if not name:
            return
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            return
        with busy_cursor('correlation...', self):
            self.cxt.current = t.wta_min_map_MSAM(self.cxt.current, exemplars, name)

        self.choose_view('vis')
        self.update_display()


    def act_kmeans(self, multi = False):
        if multi:
            if self.cxt.ho is None: 
                return
            clusters, ok1 = QInputDialog.getInt(self, "KMeans Clustering",
                "Enter number of clusters:",value=5, min=1, max=50)
            if not ok1:
                return
            iters, ok2 = QInputDialog.getInt(self, "KMeans Clustering",
                "Enter number of iterations:", value=50, min=1, max=1000)
            if not ok2:
                return
        
            with busy_cursor('clustering...', self):
                for po in self.cxt.ho:
                    t.kmeans_caller(po, clusters, iters)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    self.update_display()
            self.choose_view('hol')
            self.update_display()
            return
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        clusters, ok1 = QInputDialog.getInt(self, "KMeans Clustering",
            "Enter number of clusters:",value=5, min=1, max=50)
        if not ok1:
            return
        iters, ok2 = QInputDialog.getInt(self, "KMeans Clustering",
            "Enter number of iterations:", value=50, min=1, max=1000)
        if not ok2:
            return
    
        with busy_cursor('clustering...', self):
            self.cxt.current = t.kmeans_caller(self.cxt.current, clusters, iters)
        self.choose_view('vis')
    
        self.update_display(key=f'kmeans-{clusters}-{iters}INDEX')

    def _remap_legends(self):
        if self.legend_mapping_path is None:
            QMessageBox.information(
            self,
            "Legend remapping",
            "Legend remapping groups detailed spectral hits into "
            "interpretable mineral classes.\n\n"
            "You can choose *any* JSON file that defines these classes.\n"
            "A recommended default lives in the 'resources' folder.\n\n"
            "Please select a remapping file now."
        )

        # Default directory for the QFileDialog
        

            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select legend mapping JSON",
                ".",
                "JSON files (*.json)"
            )
            if not path:
                return  # user cancelled — do nothing safely
            self.legend_mapping_path = path
        
        if self.cxt.current is None or self.cxt.current.is_raw:
            QMessageBox.warning(self, "Open dataset", f"There is no processed dataset loaded")
        try:
            self.cxt.current = t.clean_legends(self.cxt.current, self.legend_mapping_path)
        except Exception as e:
            QMessageBox.warning(self, "Failed operation", f"Failed to remap legends: {e}")
            return
        self._distribute_context()
        self.update_display()
        
    def act_lib_pix(self):
        """Add a single pixel spectrum to the current library."""
        if self.cxt.current is None:
            QMessageBox.information(self, "Add to Library", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Add to Library", "Open a processed dataset first.")
            return
        if not self.cxt.library or not self.cxt.library.is_open():
            QMessageBox.warning(self, "Add to Library", "No library database is open.")
            return
        
        p = self._active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return
        
        def handle_point_click(y, x):
            try:
                spectrum = self.cxt.current.savgol[int(y), int(x), :]
                wavelengths_nm = self.cxt.current.bands
                print(type(spectrum), type(wavelengths_nm), 'types in handle point click')
                
                # Ask for metadata
                dlg = LibMetadataDialog(parent=self)
                if dlg.exec() != QDialog.Accepted:
                    return  # user cancelled
                
                metadata = dlg.get_metadata()
                name = metadata.get("Name", "").strip()
                if not name:
                    QMessageBox.warning(
                        self, 
                        "Add to Library", 
                        "Name is a mandatory field"
                    )
                    return
                
                metadata['SampleNum'] = f"Hole: {self.cxt.current.metadata.get('borehole id', 'Unknown')} Box: {self.cxt.current.metadata.get('box number', 'Unknown')} Pixel: ({int(y)}, {int(x)})"
                with busy_cursor('Adding to library...', self):
                    sample_id = self.cxt.library.add_sample(
                        name=name,
                        wavelengths_nm=wavelengths_nm,
                        reflectance=spectrum,
                        metadata=metadata
                    )
                
                self.statusBar().showMessage(
                    f"Added spectrum '{name}' to library (ID: {sample_id})", 
                    5000
                )
                
            except Exception as e:
                QMessageBox.warning(
                    self, 
                    "Add to Library", 
                    f"Failed to add spectrum to library: {e}"
                )
            finally:
                p.dispatcher.clear_all_temp()
        
        p.dispatcher.set_single_click(handle_point_click)
        self.statusBar().showMessage("Click a pixel to add its spectrum to the library...")

        
    def act_lib_region(self):
        """Add the average spectrum of a region to the current library."""
        if self.cxt.current is None:
            QMessageBox.information(self, "Add to Library", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Add to Library", "Open a processed dataset first.")
            return
        if not self.cxt.library or not self.cxt.library.is_open():
            QMessageBox.warning(self, "Add to Library", "No library database is open.")
            return
        
        p = self._active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return
        
        def _on_rect(y0, y1, x0, x1):
            try:
                # Extract region and compute average spectrum
                region = self.cxt.current.savgol[y0:y1, x0:x1, :]
                
                # Use mask if available to exclude masked pixels
                if self.cxt.current.has('mask'):
                    mask_region = self.cxt.current.mask[y0:y1, x0:x1]
                    valid_pixels = region[mask_region == 0]
                    if valid_pixels.size == 0:
                        QMessageBox.warning(
                            self, 
                            "Add to Library", 
                            "Selected region contains no valid (unmasked) pixels."
                        )
                        return
                    avg_spectrum = valid_pixels.mean(axis=0)
                    pixel_count = len(valid_pixels)
                else:
                    avg_spectrum = region.reshape(-1, region.shape[-1]).mean(axis=0)
                    pixel_count = (y1 - y0) * (x1 - x0)
                
                wavelengths_nm = self.cxt.current.bands
                
                dlg = LibMetadataDialog(parent=self)
                
                if dlg.exec() != QDialog.Accepted:
                    return  # user cancelled
                
                metadata = dlg.get_metadata()
                name = metadata.get("Name", "").strip()
                if not name:
                    QMessageBox.warning(
                        self, 
                        "Add to Library", 
                        "Name is a mandatory field"
                    )
                    return
                metadata['SampleNum'] = f"Hole: {self.cxt.current.metadata.get('borehole id', 'Unknown')} Box: {self.cxt.current.metadata.get('box number', 'Unknown')} Region: ({y0}-{y1},{x0}-{x1})"
                
                
                with busy_cursor('Adding to library...', self):
                    sample_id = self.cxt.library.add_sample(
                        name=name,
                        wavelengths_nm=wavelengths_nm,
                        reflectance=avg_spectrum,
                        metadata=metadata
                    )
                    self.choose_view('lib')
                    self.update_display()
                
                self.statusBar().showMessage(
                    f"Added averaged spectrum '{name}' ({pixel_count} pixels) to library (ID: {sample_id})", 
                    5000
                )
                
            except Exception as e:
                QMessageBox.warning(
                    self, 
                    "Add to Library", 
                    f"Failed to add spectrum to library: {e}"
                )
            finally:
                p.dispatcher.clear_all_temp()
        
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()
        self.statusBar().showMessage("Draw a rectangle to average and add to library...")
        
        
            # --- HOLE actions ---
    def hole_next_box(self):
        if self.cxt.ho is None or self.cxt.current is None:
            return
        if self.cxt.current.is_raw:
            return
        try:
            box_num = int(self.cxt.current.metadata.get("box number"))
        except Exception:
            box_num = None
        if box_num is not None:
            try:
                self.cxt.current = self.cxt.ho[box_num+1]
                self._distribute_context()
                self.update_display()
            except KeyError:
                return

    def hole_prev_box(self):
        if self.cxt.ho is None or self.cxt.current is None:
            return
        if self.cxt.current.is_raw:
            return
        try:
            box_num = int(self.cxt.current.metadata.get("box number"))
        except Exception:
            box_num = None
        if box_num is not None:
            try:
                self.cxt.current = self.cxt.ho[box_num-1]
                self._distribute_context()
                self.update_display()
            except KeyError:
                return

    def hole_return_to_raw(self):
        if self.cxt.ho is None or self.cxt.current is None:
            return
        if self.cxt.current.is_raw:
            return
        path = QFileDialog.getExistingDirectory(
                   self,
                   "Select directory",
                   "",
                   QFileDialog.ShowDirsOnly
                   )
        if not path:
            return
        box_num = None
        try:
            box_num = int(self.cxt.current.metadata.get("box number"))
        except Exception:
            box_num = None

        try:
            with busy_cursor('loading...', self):
                loaded_obj = t.load(path)
                if  not loaded_obj.is_raw:
                    QMessageBox.warning(self, "Return to Raw",
                        "Selected path is not a raw Lumo directory.")
                    return
                new_po = loaded_obj.process()
                new_po.update_root_dir(self.cxt.current.root_dir)
                new_po.build_all_thumbs()
                new_po.save_all_thumbs()
                if new_po.metadata.get("borehole id") != self.cxt.ho.hole_id:
                    QMessageBox.warning(self, "Return to Raw",
                        "The new loaded scan is from a different hole")
                    return
                if box_num not in self.cxt.ho.boxes:
                    QMessageBox.warning(self, "Return to Raw",
                             f"Box {box_num} not found in current hole.")
                    return
                self.cxt.ho.boxes[box_num] = new_po
                self.cxt.ho.hole_meta[box_num] = new_po.metadata

                self.cxt.current = new_po

                self.choose_view('vis')
                self.update_display()

        except Exception as e:
            QMessageBox.warning(self, "Open dataset", f"Failed to open dataset: {e}")
            return

    def save_all_changes(self):
        if self.cxt.ho is None or self.cxt.current is None:
            return

        if self.cxt is not None and self.cxt.ho is not None:
            with busy_cursor('Saving.....', self):
                print('saving')
                for po in self.cxt.ho:
                    if po.has_temps:
                        print(po.metadata['box number'])
                        po.commit_temps()
                        po.save_all()
                        print('saved all, reloading')
                        po.reload_all()
                        po.load_thumbs()
                self.choose_view('hol')
                self.update_display()

#----------- manage ClusterWindow for interogating cluster centres-------------
    def open_cluster_window(self, cluster_key: str):
        """
        Create a ClusterWindow for the given *CLUSTERS dataset key and
        show it as a standalone window.
    
        Pinned to whatever self.cxt.current is at the moment of opening.
        """
        po = self.cxt.current
        if po is None or getattr(po, "is_raw", False):
            QMessageBox.information(
                self,
                "No processed box",
                "You need a processed box selected before inspecting clusters.",
            )
            return
    
        win = ClusterWindow(
            parent=self,
            cxt=self.cxt,
            po=po,
            cluster_key=cluster_key,
        )
        win.setWindowFlag(Qt.Window, True)
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        win.setWindowTitle(cluster_key)
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
        try:
            self.cluster_windows.remove(win)
        except ValueError:
            pass
    
    
def main():
    app = QApplication(sys.argv)
    win = MainRibbonController()
    win.showMaximized()  
    sys.exit(app.exec())
