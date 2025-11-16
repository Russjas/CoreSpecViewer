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
    QDialog, QAction, QTabWidget,QSizePolicy,QStyle
)

from ribbon import Ribbon, Groups
from pages import RawPage, VisualisePage, LibraryPage, AutoSettingsDialog  
from HolePage import HolePage
from util_windows import MetadataDialog, two_choice_box, InfoTable, busy_cursor

from objects import RawObject, ProcessedObject
import tools as t
from tools import load, crop, reset, mask_rect, mask_point, improve_mask
from PyQt5.QtGui import QIcon
import multi_box
from context import CurrentContext
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
        self.open_act = QAction("Open single scan", self)
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
        
        self.multibox_act = QAction("Process Raw Multibox", self)
        self.multibox_act.triggered.connect(self.process_multi_raw)
        everpresents = [self.open_act, self.multibox_act, self.save_act, self.save_as_act, self.undo_act]

        self.ribbon.add_global_actions(everpresents)
        #====== non-tab buttons=================
        self.info_act = QAction("Info", self)
        self.info_act.setShortcut("Ctrl+I")
        self.info_act.triggered.connect(self.display_info)
        self.settings_act = QAction("Settings", self)
        self.settings_act.triggered.connect(self.on_settings)

        self.ribbon.add_global_actions([self.info_act, self.settings_act], pos='right')

        # ===== Create all pages==============================================
        self.tabs = QTabWidget(self)
        self.tabs.setTabPosition(QTabWidget.East)   # or South if you prefer
         

        self.raw_page = RawPage(self)
        self.vis_page = VisualisePage(self)
        self.lib_page = LibraryPage(self)
        self.hol_page = HolePage(self)
        
        self.hol_page.changeView.connect(lambda key: self.choose_view(key, force=True))
        
        self.page_list = [self.raw_page, self.vis_page, self.lib_page, self.hol_page]

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
            ("button", "Freehand mask region", self.act_mask_polygon),
            ("button", "Improve", self.act_mask_improve),
            ("button", "Calc stats", self.act_mask_calc_stats),
            ("button", "unwrap preview", self.unwrap)
        ])

        # --- VISUALISE TAB ---
        self.extract_feature_list = []
        for key in feature_keys:
            self.extract_feature_list.append((key, lambda _, k=key: self.run_feature_extraction(k)))
        self.ribbon.add_tab('Visualise', [
            ("button", "Quick Cluster", self.act_kmeans),
            ("menu",   "Correlation", [
                ("MineralMap (Winner-takes-all)", lambda: self.act_vis_correlation("gpt vectors")),
                ("Other…",            lambda: self.act_vis_correlation("other")),
            ]),
            ("menu",   "Features", self.extract_feature_list),
            ])
        # --- HOLE TAB ---
        self.ribbon.add_tab('Hole operations',[
                            ("button", "Next", self.hole_next_box),
                            ("button", "Return to Raw", self.hole_return_to_raw),
                            
                            ])
        
    #======== UI methods ===============================================
    def update_display(self, key = 'mask'):
        p = self._active_page()
        if hasattr(p, "cache"):
            p.add_to_cache(key)
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
        
        
    def load_from_disk(self):
        '''loads PO or RO only, HO and db are loaded from control panel on
        thei respective games'''
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
                loaded_obj = load(path)
                
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
    
    def save_clicked_temp(self):#TODO if the save error doesnt recur this can be deleted

        print('Why no changes?')
        self.cxt.current.print_refs('before clear canvas')
        #self._clear_all_canvas_refs()
        self.cxt.current.print_refs('after clear canvas current')
        self.cxt.po.print_refs('after clear canvas po')
        #self.ref_hold = self.cxt.current.datasets['savgol']._memmap_ref
        print('COMMITING')
        self.cxt.po.commit_temps()
        print('SAVING')
        self.cxt.current.save_all()
        self._distribute_context()
        self.update_display()
        

        
    def save_clicked(self):
         
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "save", "Raw data must be processed prior to saving")
            return
        if self.cxt.current.has_temps:
            test = two_choice_box('Commit changes before saving?', 'yes', 'no')
            
            if test == 'left':
                
                self.cxt.po.commit_temps()
                
                self.cxt.po.build_all_thumbs()
                
                self.cxt.po.save_all_thumbs()
                
             
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
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        self.cxt.current = reset(self.cxt.current)
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
                self.cxt.current = crop(self.cxt.current, y0, y1, x0, x1)   
                self._distribute_context()
                self.update_display()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()
        
    def automatic_crop(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
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
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Mask region", "Open a processed dataset first.")
            return
        p = self._active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return

        def _on_rect(y0, y1, x0, x1):
            try:
                self.cxt.current = mask_rect(self.cxt.current, y0, y1, x0, x1 )
                self._distribute_context()
                self.update_display()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()

    def act_mask_point(self, mode):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
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
                    self.cxt.current = mask_point(self.cxt.current, mode, y, x)
                self._distribute_context()
                self.update_display()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_single_click(handle_point_click)
        
        
    def act_mask_improve(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        self.cxt.current = improve_mask(self.cxt.current)
        self._distribute_context()
        self.update_display()
       
    def act_mask_polygon(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        p = self._active_page()
        if not p or not p.dispatcher or self.cxt.current is None:
            return
    
        def _on_finish(vertices_rc):
            
            self.cxt.current = t.mask_polygon(self.cxt.current, vertices_rc)
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
    def run_feature_extraction(self, key):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        print(key)
        with busy_cursor(f'extracting {key}...', self):
            self.cxt.current = t.run_feature_extraction(self.cxt.current, key)
        self._distribute_context()
        self.update_display()
        
        
    def act_vis_correlation(self, kind: str):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        if self.cxt.current.is_raw:
            QMessageBox.information(self, "Correlation", "Open a processed dataset first.")
            return
        exemplars, coll_name = self.lib_page.get_collection_exemplars()
        if not exemplars or not coll_name:
            return
        with busy_cursor('correlation...', self):
            self.cxt.current = t.wta_min_map(self.cxt.current, exemplars, coll_name, kind)
        
        self.choose_view('vis')
        self.update_display()

    def act_kmeans(self):
        if self.cxt.current is None:
            QMessageBox.information(self, "Correlation", "No Current Scan")
            return
        # Prompt for clusters
        clusters, ok1 = QInputDialog.getInt(
            self,
            "KMeans Clustering",
            "Enter number of clusters:",
            value=5,          # default
            min=1,
            max=50
        )
        
        # If cancelled, abort
        if not ok1:
            return
        # Prompt for iterations
        iters, ok2 = QInputDialog.getInt(
            self,
            "KMeans Clustering",
            "Enter number of iterations:",
            value=50,         # default
            min=1,
            max=1000
        )
        
        if not ok2:
            return
        
        with busy_cursor('clustering...', self):
            self.cxt.current = t.kmeans_caller(self.cxt.current, clusters, iters)
        self.choose_view('vis')
        
        self.update_display(key=f'kmeans-{clusters}-{iters}INDEX')

    def _clear_all_canvas_refs(self):
        """Clear memmap references from all page canvases before saving."""
        
        
        for page in self.page_list:
            # Clear left canvas (SpectralImageCanvas)
            if hasattr(page, 'left_canvas') and hasattr(page.left_canvas, 'clear_memmap_refs'):
                page.left_canvas.clear_memmap_refs()
            
            # Clear right canvas (ImageCanvas2D)
            if hasattr(page, 'right_canvas') and hasattr(page.right_canvas, 'clear_memmap_refs'):
                page.right_canvas.clear_memmap_refs()

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
                loaded_obj = load(path)
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
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainRibbonController()
    win.show()
    sys.exit(app.exec()) 