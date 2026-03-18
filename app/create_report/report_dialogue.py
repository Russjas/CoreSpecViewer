"""
Dialog for selecting which dataset keys to include in a hole-level PDF booklet.
 
Allows the user to select from datasets that are present in ALL boxes within
the hole, ensuring consistent output across the entire booklet.
 
Extended version supports downhole product dataset plots via tabbed interface.
"""
import logging
from pathlib import Path
 
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QDialogButtonBox,
    QMessageBox,
    QSpinBox,
    QTabWidget,
    QWidget,
    QCheckBox
)
 
logger = logging.getLogger(__name__)
 
 
class PDFExportDialog(QDialog):
    """
    Dialog for selecting dataset keys to include in PDF booklet.
    
    Presents two tabs:
    1. Box Images: 2D dataset keys that are available in ALL boxes
    2. Downhole Plots: 1D product datasets for downhole plotting
    
    Allows multi-selection in both tabs and returns the selected keys on accept.
    """
    
    def __init__(self, cxt, parent=None):
        """
        Parameters
        ----------
        cxt : CurrentContext
            Current application context containing the loaded hole
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.cxt = cxt
        self.setWindowTitle("Export Hole PDF Booklet")
        self.selected_keys = []
        self.selected_product_keys = []
        self.boxes_per_page = 2
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabs = QTabWidget(self)
        
        # ====================================================================
        # TAB 1: Box Images
        # ====================================================================
        box_tab = QWidget()
        box_layout = QVBoxLayout(box_tab)
        
        # Instructions for box images
        box_instructions = QLabel(
            "Select dataset keys to include in the PDF booklet.\n"
            "Each selected key will generate one overview page plus box detail pages.\n"
            "Only datasets present in ALL boxes are shown."
        )
        box_instructions.setWordWrap(True)
        box_layout.addWidget(box_instructions)
        
        # Boxes per page selector
        boxes_layout = QHBoxLayout()
        boxes_label = QLabel("Boxes per page:")
        boxes_layout.addWidget(boxes_label)
        
        self.boxes_spinbox = QSpinBox(self)
        self.boxes_spinbox.setMinimum(2)
        self.boxes_spinbox.setMaximum(3)
        self.boxes_spinbox.setValue(2)  # Default
        self.boxes_spinbox.setToolTip("Number of boxes to display per landscape page")
        boxes_layout.addWidget(self.boxes_spinbox)
        boxes_layout.addStretch()
        
        box_layout.addLayout(boxes_layout)
        
        # List widget for box dataset multi-selection
        self.box_list_widget = QListWidget(self)
        self.box_list_widget.setSelectionMode(QListWidget.MultiSelection)
        
        # Populate with available box keys
        available_keys = self._get_keys_in_all_boxes()
        for key in available_keys:
            item = QListWidgetItem(key)
            self.box_list_widget.addItem(item)
        
        box_layout.addWidget(self.box_list_widget)
        
        # Selection buttons for box tab
        box_btn_layout = QHBoxLayout()
        
        box_select_all_btn = QPushButton("Select All")
        box_select_all_btn.clicked.connect(self.select_all_boxes)
        box_btn_layout.addWidget(box_select_all_btn)
        
        box_deselect_all_btn = QPushButton("Deselect All")
        box_deselect_all_btn.clicked.connect(self.deselect_all_boxes)
        box_btn_layout.addWidget(box_deselect_all_btn)
        
        box_layout.addLayout(box_btn_layout)
        
        self.tabs.addTab(box_tab, "Box Images")
        
        # ====================================================================
        # TAB 2: Downhole Plots
        # ====================================================================
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        
        # Instructions for downhole plots
        plot_instructions = QLabel(
            "Select downhole product datasets to plot.\n"
            "Each selected dataset will generate one downhole plot page.\n"
            "Plots show depth vs. feature values for the entire hole. \n"
            "NB: Plots use the last selected resampling window"
        )
        plot_instructions.setWordWrap(True)
        plot_layout.addWidget(plot_instructions)
        
        # List widget for product dataset multi-selection
        self.product_list_widget = QListWidget(self)
        self.product_list_widget.setSelectionMode(QListWidget.MultiSelection)
        
        # Populate with available product datasets
        available_products = self._get_plottable_product_keys()
        for key in available_products:
            from ..ui.display_text import gen_display_text
            display_name = gen_display_text(key)
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, key)  # Store actual key
            self.product_list_widget.addItem(item)
        
        plot_layout.addWidget(self.product_list_widget)
        
        # Selection buttons for product tab
        product_btn_layout = QHBoxLayout()
        
        product_select_all_btn = QPushButton("Select All")
        product_select_all_btn.clicked.connect(self.select_all_products)
        product_btn_layout.addWidget(product_select_all_btn)
        
        product_deselect_all_btn = QPushButton("Deselect All")
        product_deselect_all_btn.clicked.connect(self.deselect_all_products)
        product_btn_layout.addWidget(product_deselect_all_btn)
        
        plot_layout.addLayout(product_btn_layout)
        
        # Info label for plot count
        self.plot_info_label = QLabel("")
        self.plot_info_label.setStyleSheet("color: #666; font-style: italic;")
        plot_layout.addWidget(self.plot_info_label)
        
        self.tabs.addTab(plot_tab, "Downhole Plots")
        
        # Update info when selection changes
        self.product_list_widget.itemSelectionChanged.connect(self._update_plot_info)
        
        main_layout.addWidget(self.tabs)
        
        # ====================================================================
        # Dialog buttons
        # ====================================================================
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        self.setMinimumWidth(500)
        self.setMinimumHeight(600)
        
        # Initialize info label
        self._update_plot_info()
    
    def _get_keys_in_all_boxes(self) -> list[str]:
        """Find dataset keys that are present in ALL boxes."""
        valid_state, msg = self.cxt.requires(self.cxt.HOLE)
        if not valid_state:
            return []
        
        hole = self.cxt.ho
        if not hole.boxes:
            return []
        
        # Start with keys from first box
        first_box = list(hole.boxes.values())[0]
        all_keys = set(first_box.datasets.keys()) | set(first_box.temp_datasets.keys())
        
        # Intersect with keys from all other boxes
        for po in hole:
            box_keys = set(po.datasets.keys()) | set(po.temp_datasets.keys())
            all_keys &= box_keys
        
        # Filter out un-needed keys
        excluded_keys = {'metadata', 'bands', 'savgol', 'stats', 'cropped', 'savgol_cr'}
        visualizable_keys = [
            k for k in all_keys 
            if k not in excluded_keys 
            and not k.endswith("LEGEND") 
            and not k.lower().startswith("prof")
        ]
        
        return sorted(visualizable_keys)
    
    def _get_plottable_product_keys(self) -> list[str]:
        """
        Get product datasets suitable for downhole plotting.
        
        Returns keys from HoleObject.product_datasets that can be plotted,
        excluding LEGEND and CLUSTERS datasets which are metadata.
        """
        valid_state, msg = self.cxt.requires(self.cxt.HOLE)
        if not valid_state:
            return []
        
        hole = self.cxt.ho
        if not hole.product_datasets:
            return []
        
        # Exclude non-plottable suffixes
        excluded_suffixes = ('LEGEND', 'CLUSTERS')
        plottable = [
            k for k in hole.product_datasets.keys()
            if not any(k.endswith(s) for s in excluded_suffixes)
        ]
        
        return sorted(plottable)
    
    def select_all_boxes(self):
        """Select all items in the box list."""
        for i in range(self.box_list_widget.count()):
            self.box_list_widget.item(i).setSelected(True)
    
    def deselect_all_boxes(self):
        """Deselect all items in the box list."""
        self.box_list_widget.clearSelection()
    
    def select_all_products(self):
        """Select all items in the product list."""
        for i in range(self.product_list_widget.count()):
            self.product_list_widget.item(i).setSelected(True)
    
    def deselect_all_products(self):
        """Deselect all items in the product list."""
        self.product_list_widget.clearSelection()
    
    def _update_plot_info(self):
        """Update the info label showing how many plots are selected."""
        selected_count = len(self.product_list_widget.selectedItems())
        if selected_count == 0:
            self.plot_info_label.setText("No downhole plots selected")
        elif selected_count == 1:
            self.plot_info_label.setText("1 downhole plot will be generated")
        else:
            self.plot_info_label.setText(f"{selected_count} downhole plots will be generated")
    
    def accept(self):
        """Override accept to validate selection, get output path, and generate PDF."""
        # Get selections from both tabs
        box_selected_items = self.box_list_widget.selectedItems()
        product_selected_items = self.product_list_widget.selectedItems()
        
        # At least one selection required
        if not box_selected_items and not product_selected_items:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select at least one box dataset or downhole plot to include in the PDF."
            )
            return
        
        # Extract selected keys
        self.selected_keys = [item.text() for item in box_selected_items]
        self.selected_product_keys = [item.data(Qt.UserRole) for item in product_selected_items]
        self.boxes_per_page = self.boxes_spinbox.value()
        
        logger.info(f"User selected {len(self.selected_keys)} box keys for PDF export: {self.selected_keys}")
        logger.info(f"User selected {len(self.selected_product_keys)} product keys for plots: {self.selected_product_keys}")
        logger.info(f"Boxes per page: {self.boxes_per_page}")
        
        # Ask where to save the PDF
        hole = self.cxt.ho
        default_filename = f"{hole.hole_id}_booklet.pdf"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF Booklet",
            str(hole.root_dir / default_filename),
            "PDF Files (*.pdf)"
        )
        
        if not output_path:
            logger.info("PDF export cancelled - no output path selected")
            return
        
        # Generate the PDF
        try:
            from .pdf_booklet import create_hole_pdf_booklet
            from ..ui.util_windows import busy_cursor
            
            with busy_cursor('Generating PDF booklet...', self):
                pdf_path = create_hole_pdf_booklet(
                    hole=hole,
                    selected_keys=self.selected_keys,
                    output_path=output_path,
                    boxes_per_page=self.boxes_per_page,
                    selected_product_keys=self.selected_product_keys
                )
            
            # Build success message
            sections = []
            if self.selected_keys:
                sections.append(f"{len(self.selected_keys)} box dataset(s)")
            if self.selected_product_keys:
                sections.append(f"{len(self.selected_product_keys)} downhole plot(s)")
            
            message = f"PDF booklet created successfully with:\n"
            message += "\n".join(f"  • {s}" for s in sections)
            message += f"\n\nSaved to:\n{pdf_path}"
            
            QMessageBox.information(
                self,
                "Export Complete",
                message
            )
            logger.info(f"PDF booklet exported to {pdf_path}")
            super().accept()
            
        except Exception as e:
            logger.error(f"Failed to create PDF booklet: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to create PDF booklet:\n\n{str(e)}"
            )
            
    
