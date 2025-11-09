# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:58:41 2025

@author: russj
"""
import sys
# controller.py
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QStackedWidget, QFileDialog, QMessageBox, QApplication, QInputDialog,
    QDialog, QAction, QTabWidget,QSizePolicy
)

from ribbon import Ribbon
from pages import RawPage, VisualisePage, LibraryPage
from util_windows import MetadataDialog, two_choice_box, InfoTable, busy_cursor

from objects import RawObject, ProcessedObject
import tools as t
from tools import load, crop, reset, mask_rect, mask_point, improve_mask
from PyQt5.QtGui import QIcon
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
        self.current_obj = None


        # --- UI shell: ribbon + stacked pages ---
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)
        # ===== Ribbon operations
        self.ribbon = Ribbon(self)
        
       
        outer.addWidget(self.ribbon, 0)
        #define everpresent actions
        # --- Create actions ---
        self.open_act = QAction("Open", self)
        self.open_act.setShortcut("Ctrl+O")
        self.open_act.triggered.connect(self.load_from_disk)

        self.save_act = QAction("Save", self)
        self.save_act.setShortcut("Ctrl+S")
        self.save_act.triggered.connect(self.save_clicked)
        
        self.save_as_act = QAction("Save As", self)
        self.save_as_act.triggered.connect(self.save_as_clicked)
        
        self.undo_act = QAction("Undo", self)
        self.undo_act.setShortcut("Ctrl+Z")
        self.undo_act.triggered.connect(self.undo_unsaved)

        everpresents = [self.open_act, self.save_act, self.save_as_act, self.undo_act]

        self.ribbon.add_global_actions(everpresents)
        
        self.info_act = QAction("Info", self)
        self.info_act.setShortcut("Ctrl+I")
        self.info_act.triggered.connect(self.display_info)
        self.ribbon.add_global_actions([self.info_act], pos = 'right')

        # ===== Page operations
        self.tabs = QTabWidget(self)
        self.tabs.setTabPosition(QTabWidget.East)   # or South if you prefer
         

        self.raw_page = RawPage(self)
        self.vis_page = VisualisePage(self)
        self.lib_page = LibraryPage(self)

        self.tabs.addTab(self.raw_page, "Raw")
        self.tabs.addTab(self.vis_page, "Visualise")
        self.tabs.addTab(self.lib_page, "Libraries")
        
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
            ("button", "Auto Crop", self.automatic_crop),
            ("button", "Crop",        self.crop_current_image),
            
            ("button", "Process",     self.process_raw),
        ])

        # --- MASK TAB ---
        self.ribbon.add_tab('Masking', [
            ("button", "New mask", lambda: self.act_mask_point('new')),
            ("button", "Enhance mask", lambda: self.act_mask_point('enhance')),
            ("button", "Mask line", lambda: self.act_mask_point('line')),
            ("button", "Mask region", self.act_mask_rect),
            ("button", "Improve", self.act_mask_improve),
            
            ("button", "Calc stats", self.act_mask_calc_stats),
            ("button", "unwrap preview", self.unwrap)
            
                   
        ])

        # --- VISUALISE TAB ---
        self.extract_feature_list = []
        for key in feature_keys:
            
            self.extract_feature_list.append((key, lambda _, k=key: self.run_feature_extraction(k)))
        
        
        
        
        self.ribbon.add_tab('Visualise', [
            
            ("menu",   "Correlation", [
                ("Pearson (template)", lambda: self.act_vis_correlation("pearson")),
                ("Other…",            lambda: self.act_vis_correlation("other")),
            ]),
            ("menu",   "Features", self.extract_feature_list),
            ])
    
    def update_display(self, key = 'mask'):
        p = self._active_page()
        if hasattr(p, "cache"):
            p.add_to_cache(key)
        p.update(key = key)    
    
    def set_current_conditions(self):
        for pg in (self.raw_page, self.vis_page, self.lib_page):
            pg.current_obj = self.current_obj
        idx_map = {"raw": 0, "mask": 1, "visualise": 2}
        
        if self.current_obj.is_raw:
            idx = 0
        else:
            idx = 1
        old_page = self._active_page()
        if old_page:
            old_page.teardown()
        self.tabs.setCurrentIndex(idx)

        new_page = self._active_page()
        if new_page:
            new_page.activate()
    
    
    def _active_page(self):
        return self.tabs.currentWidget()

    def _mode(self) -> str:
        """Return logical mode name based on current tab index."""
        idx = self.tabs.currentIndex()
        return ("raw", "mask", "visualise")[idx]

    def display_info(self):
        print('info button clicked')
        self.table_window = InfoTable()
        self.table_window.set_from_dict(self.current_obj.metadata)
        self.table_window.setWindowTitle("Info Table")
        self.table_window.resize(400, 300)
        self.table_window.show()
    # -------- RAW actions --------
    def load_from_disk(self):
        
        clicked_button = two_choice_box( "What would you like to open?", "Processed dataset", "Raw directory")
        
        if clicked_button == 'cancel':
            return
        if clicked_button == 'right':
            path = QFileDialog.getExistingDirectory(
                       self,
                       "Select directory",
                       "",                       
                       QFileDialog.ShowDirsOnly  
                       )  
            if not path:
                return
        elif clicked_button == 'left':
            path, _ = QFileDialog.getOpenFileName(
            self, "Open JSON Metadata", "", "JSON files (*.json)")
            if not path:
                return
        try:
            with busy_cursor('loading...', self):
                self.current_obj = load(path)
        except Exception as e:
            QMessageBox.warning(self, "Open dataset", f"Failed to open dataset: {e}")
            return

               
        self.set_current_conditions()
        self.update_display()

    def crop_current_image(self):
        p = self._active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return

        # Ask the page to collect a rectangle and pass back coords
        def _on_rect(y0, y1, x0, x1):
            try:
                self.current_obj = crop(self.current_obj, y0, y1, x0, x1)   
                self.set_current_conditions()
                self.update_display()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()

    def automatic_crop(self):
        if not self.current_obj.is_raw:
            QMessageBox.information(self, "Auto crop", "Auto crop only works on raw data (for now)")
            return
        with busy_cursor('cropping...', self):
            self.current_obj = t.crop_auto(self.current_obj)
        self.set_current_conditions()
        self.update_display()

    def undo_unsaved(self):
        self.current_obj = reset(self.current_obj)
        self.set_current_conditions()
        self.update_display()
        
    def save_clicked(self):
        if self.current_obj.is_raw:
            QMessageBox.information(self, "save", "Raw data must be processed prior to saving")
            return
        if self.current_obj.has_temps:
            test = two_choice_box('Commit changes before saving?', 'yes', 'no')
            if test == 'left':
                self.current_obj.commit_temps()
                
        wants_prompt = True
        if self.current_obj.datasets:
            wants_prompt = not any(ds.path.exists() for ds in self.current_obj.datasets.values())
    
        if wants_prompt:
            dest = QFileDialog.getExistingDirectory(self, "Choose save folder", str(self.current_obj.root_dir))
            if not dest:
                return
            self.current_obj.update_root_dir(dest)  # rewires every dataset path to the chosen folder
        try:
            with busy_cursor('saving...', self):
                self.current_obj.save_all()
        except Exception as e:
            QMessageBox.warning(self, "Save dataset", f"Failed to save dataset: {e}")
            return
    def save_as_clicked(self):
        if self.current_obj.is_raw:
            QMessageBox.information(self, "save", "Raw data must be processed prior to saving")
            return
        if self.current_obj.has_temps:
            test = two_choice_box('Commit changes before saving?', 'yes', 'no')
            if test == 'left':
                self.current_obj.commit_temps()
                
        dest = QFileDialog.getExistingDirectory(self, "Choose save folder", str(self.current_obj.root_dir))
        if not dest:
            return
        self.current_obj.update_root_dir(dest)  # rewires every dataset path to the chosen folder
        try:
            with busy_cursor('saving...', self):
                self.current_obj.save_all()
        except Exception as e:
            QMessageBox.warning(self, "Save dataset", f"Failed to save dataset: {e}")
            return
        
    
    def process_raw(self):
        if not self.current_obj.is_raw:
            QMessageBox.information(self, "Process", "Load a raw dataset first.")
            return
        if (
        not self.current_obj.metadata.get('borehole id')
        or not self.current_obj.metadata.get('box number')
        or not self.current_obj.metadata.get('core depth start')
        or not self.current_obj.metadata.get('core depth stop')
        ):
            dlg = MetadataDialog(self.raw.metadata, parent=self)
            if dlg.exec() == QDialog.Accepted:
                result = dlg.get_result()
                self.current_obj.metadata['borehole id'] = result['hole']
                self.current_obj.metadata['box number'] = result['box']
                self.current_obj.metadata['core depth start'] = result['depth_from']
                self.current_obj.metadata['core depth stop'] = result['depth_to']
        
        try:
            with busy_cursor('processing...', self):
                self.current_obj = self.current_obj.process()  # placeholder
            
        except Exception as e:
            QMessageBox.warning(self, "Process", f"Failed to process/save: {e}")
            return

        self.set_current_conditions()
        self.update_display()
        self.statusBar().showMessage("Processed saved")
        

    # -------- MASK actions --------

    def act_mask_rect(self):
        if self.current_obj.is_raw:
            QMessageBox.information(self, "Mask region", "Open a processed dataset first.")
            return
        p = self._active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return

        def _on_rect(y0, y1, x0, x1):
            try:
                self.current_obj = mask_rect(self.current_obj, y0, y1, x0, x1 )
                self.set_current_conditions()
                self.update_display()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()

    def act_mask_point(self, mode):
        if self.current_obj.is_raw:
            QMessageBox.information(self, "Mask region", "Open a processed dataset first.")
            return
        p = self._active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return
        
        def handle_point_click(y, x):
            try:
                with busy_cursor('trying mask correlation...', self):
                    self.current_obj = mask_point(self.current_obj, mode, y, x)
                self.set_current_conditions()
                self.update_display()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_single_click(handle_point_click)
        
        
    def act_mask_improve(self):
        self.current_obj = improve_mask(self.current_obj)
        self.set_current_conditions()
        self.update_display()
       
    
    def act_mask_calc_stats(self):
        if self.current_obj.is_raw:
            QMessageBox.information(self, "Stats", "Open a processed dataset first.")
            return
        self.current_obj = t.calc_unwrap_stats(self.current_obj)
        
        self.set_current_conditions()
        self.update_display(key = 'segments')
        
        
    def unwrap(self):
        
        if not self.current_obj.has('stats'):
            QMessageBox.warning(self, "Warning", "No stats calculated yet.")
            return
        with busy_cursor('unwrapping...', self):
            self.current_obj = t.unwrapped_output(self.current_obj)
        self.set_current_conditions()
        self.update_display(key='DholeAverage')

    # -------- VISUALISE actions --------
    def run_feature_extraction(self, key):
        print(key)
        with busy_cursor(f'extracting {key}...', self):
            self.current_obj = t.run_feature_extraction(self.current_obj, key)
        self.set_current_conditions()
        self.update_display()
        
        
    def act_vis_correlation(self, kind: str):
        if not self.po:
            QMessageBox.information(self, "Correlation", "Open a processed dataset first.")
            return
        p = self._active_page()
        # TODO: run correlation → show in right_canvas, update table
        QMessageBox.information(self, "Correlation", f"Ran correlation: {kind}")

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainRibbonController()
    win.show()
    sys.exit(app.exec()) 