"""
PDF Booklet Export for HoleObject.
 
Creates an A4 PDF booklet with:
- Title page with hole metadata (portrait)
- TWO boxes per landscape page per selected dataset key (stacked vertically)
- Overview pages at the end showing concatenated images for each feature (portrait)
 
Uses reportlab for PDF generation.
"""
from io import BytesIO
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
 
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
import matplotlib
from matplotlib.figure import Figure

from ..spectral_ops.visualisation import mk_thumb, DISPLAY_RANGE
from ..models.hole_object import HoleObject
from ..ui.display_text import gen_display_text

logger = logging.getLogger(__name__)

def format_missing_boxes(present_boxes: list[int], first_box: int, last_box: int) -> str:
    """Format missing boxes into compact range notation."""
    if not present_boxes:
        if first_box == last_box:
            return f"{first_box} missing"
        return f"{first_box}-{last_box} missing"
    
    present_set = set(present_boxes)
    missing = [n for n in range(first_box, last_box + 1) if n not in present_set]
    
    if not missing:
        return "All boxes present"
    
    ranges = []
    start = prev = None
    
    for n in missing:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = n
    
    if start is not None:
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    
    return ", ".join(ranges) + " missing"


def create_hole_pdf_booklet(
    hole: HoleObject,
    selected_keys: list[str],
    output_path: Path | str,
    boxes_per_page = 3,
    include_downhole_plots: bool = True,  # NEW
    selected_product_keys: list[str] = None  # NEW
) -> Path:
    """
    Create a PDF booklet for a drill hole with selected dataset visualizations.
    
    Parameters
    ----------
    hole : HoleObject
        The hole object containing all boxes
    selected_keys : list[str]
        Dataset keys to include (must be present in all boxes)
    output_path : Path | str
        Path where the PDF should be saved
        
    Returns
    -------
    Path
        Path to the created PDF file
        
    Raises
    ------
    ValueError
        If selected_keys contains keys not present in all boxes
        
    Notes
    -----
    PDF structure:
    1. Title page with hole metadata (portrait)
    2. TWO boxes per page per selected key (landscape, stacked vertically)
    3. Overview pages showing all boxes concatenated for each key (portrait)
    4. Downhole plots (NEW - optional)
    """
    output_path = Path(output_path)
    
    # Validate that all keys are present in all boxes
    for key in selected_keys:
        if not hole.check_for_all_keys(key):
            raise ValueError(f"Key '{key}' is not present in all boxes")
    
    logger.info(f"Creating PDF booklet for hole {hole.hole_id} with {len(selected_keys)} keys")
    
    # Create canvas for manual page building
    c = pdf_canvas.Canvas(str(output_path), pagesize=A4)
    
    # ========================================================================
    # 1. TITLE PAGE (Portrait)
    # ========================================================================
    _build_title_page(c, hole, selected_keys)
    c.showPage()
    # ========================================================================
    # 2. OVERVIEW PAGES (Portrait)
    # ========================================================================
    logger.info(f"Generating {len(selected_keys)} overview pages...")
    for key in selected_keys:
        c.setPageSize(A4)
        _build_overview_page(c, hole, key)
        c.showPage()
    # ========================================================================
    # 4. Hole plots (Portrait - 2 or 3 boxes per page)
    # ========================================================================

    if include_downhole_plots and selected_product_keys:
        _build_downhole_plots_section(c, hole, selected_product_keys)
    # ========================================================================
    # 3. BOX PAGES (Landscape - 2 or 3 boxes per page)
    # ========================================================================
    num_pages = (hole.num_box * len(selected_keys) + 1) // 3  # Ceiling division
    logger.info(f"Generating {num_pages} box pages (3 boxes per page)...")
    if boxes_per_page not in [2, 3]:
        boxes_per_page = 2
    if boxes_per_page ==2:  
        for key in selected_keys:
            boxes_list = list(hole)

            # Process boxes in pairs
            for i in range(0, len(boxes_list), 2):
                c.setPageSize(landscape(A4))
                
                # Get first box (top half)
                po1 = boxes_list[i]
                
                # Get second box if it exists (bottom half)
                po2 = boxes_list[i + 1] if i + 1 < len(boxes_list) else None
                
                _build_double_box_page(c, po1, po2, key)
                c.showPage()
    else:    
        for key in selected_keys:
            boxes_list = list(hole)
            
            # Process boxes in pairs
            for i in range(0, len(boxes_list), 3):
                c.setPageSize(landscape(A4))
                
                # Get first box
                po1 = boxes_list[i]
                
                # Get second box if it exists
                po2 = boxes_list[i + 1] if i + 1 < len(boxes_list) else None
                # Get third box if it exists
                po3 = boxes_list[i + 2] if i + 2 < len(boxes_list) else None
                _build_triple_box_page(c, po1, po2, po3, key)
                c.showPage()

    # Save PDF
    c.save()
    logger.info(f"PDF booklet created successfully: {output_path}")
    return output_path


def _build_title_page(c, hole, selected_keys):
    """Build title page on canvas (portrait)."""
    width, height = A4
    margin = 1.5 * cm
    
    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height - 2*inch, f"Drill Hole Report: {hole.hole_id}")
    
    # Metadata
    c.setFont("Helvetica", 11)
    y = height - 2.8*inch
    
    # Collect metadata
    starts, stops = [], []
    for meta in hole.hole_meta.values():
        try:
            s = float(meta.get("core depth start", "nan"))
            if np.isfinite(s): starts.append(s)
        except: pass
        try:
            e = float(meta.get("core depth stop", "nan"))
            if np.isfinite(e): stops.append(e)
        except: pass
    
    # Build metadata lines
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Hole ID:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + 100, y, str(hole.hole_id))
    y -= 20
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Number of Boxes:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + 100, y, str(hole.num_box))
    y -= 20
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Box Range:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + 100, y, f"{hole.first_box} to {hole.last_box}")
    y -= 20
    
    if starts and stops:
        dmin, dmax = min(starts), max(stops)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "Depth Range:")
        c.setFont("Helvetica", 11)
        c.drawString(margin + 100, y, f"{dmin:.2f}m to {dmax:.2f}m")
        y -= 20
        
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "Total Depth:")
        c.setFont("Helvetica", 11)
        c.drawString(margin + 100, y, f"{dmax - dmin:.2f}m")
        y -= 20
    
    # Missing boxes
    present_boxes = sorted([int(i) for i in hole.boxes.keys()])
    missing_text = format_missing_boxes(present_boxes, hole.first_box, hole.last_box)
    if missing_text != "All boxes present":
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "Missing boxes:")
        c.setFont("Helvetica", 11)
        c.drawString(margin + 100, y, missing_text)
        y -= 20
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Report Generated:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + 100, y, datetime.now().strftime('%Y-%m-%d %H:%M'))
    y -= 20
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Datasets Included:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + 100, y, str(len(selected_keys)))
    y -= 30
    
    # Dataset list
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Dataset List:")
    y -= 20
    c.setFont("Helvetica", 10)
    for key in selected_keys:
        c.drawString(margin + 20, y, gen_display_text(key))
        y -= 15


def _build_double_box_page(c, po1, po2, key):
    """Build a landscape page with two boxes stacked vertically."""
    width, height = landscape(A4)
    margin = 1.0 * cm  # Slightly smaller margin for more space
    
    # Calculate available space for each half
    divider_y = height / 2
    legend_space = 50  # Space reserved for legend at bottom of each half
    
    # ========================================================================
    # TOP HALF - First Box
    # ========================================================================
    _build_single_box_segment(c, po1, key, 
                          x_margin=margin, 
                          y_top=height - margin, 
                          y_bottom=divider_y + 5,  # Small gap between halves
                          width=width,
                          legend_space=legend_space)
    
    # ========================================================================
    # BOTTOM HALF - Second Box (if exists)
    # ========================================================================
    if po2 is not None:
        _build_single_box_segment(c, po2, key, 
                              x_margin=margin, 
                              y_top=divider_y - 5,  # Small gap between halves
                              y_bottom=margin,
                              width=width,
                              legend_space=legend_space)
    else:
        # If odd number of boxes, leave bottom half empty or add a note
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(margin, divider_y / 2, "(Last page - no additional box)")

def _build_triple_box_page(c, po1, po2, po3, key):
    """Build a landscape page with three boxes stacked vertically."""
    width, height = landscape(A4)
    margin = 1.0 * cm
    
    # Calculate available space and dividers
    in_margin_height = height - (2 * margin)
    third = in_margin_height / 3
    gap = 3  # Small gap between segments
    
    div_1 = height - margin - third  # Bottom of top third
    div_2 = div_1 - third  # Bottom of middle third
    
    legend_space = 50  # Space reserved for legend at bottom of each segment
    
    # ========================================================================
    # TOP THIRD - First Box
    # ========================================================================
    _build_single_box_segment(c, po1, key, 
                          x_margin=margin, 
                          y_top=height - margin, 
                          y_bottom=div_1 + gap,
                          width=width,
                          legend_space=legend_space)
    
    # ========================================================================
    # MIDDLE THIRD - Second Box (if exists)
    # ========================================================================
    if po2 is not None:
        _build_single_box_segment(c, po2, key, 
                              x_margin=margin, 
                              y_top=div_1 - gap,
                              y_bottom=div_2 + gap,
                              width=width,
                              legend_space=legend_space)
    
    # ========================================================================
    # BOTTOM THIRD - Third Box (if exists)
    # ========================================================================
    if po3 is not None:
        _build_single_box_segment(c, po3, key, 
                              x_margin=margin, 
                              y_top=div_2 - gap,
                              y_bottom=margin,
                              width=width,
                              legend_space=legend_space)



def _build_single_box_segment(c, po, key, x_margin, y_top, y_bottom, width, legend_space):
    """Build a single box visualization in a designated vertical region."""
    
    box_num = po.metadata.get('box number', 'Unknown')
    box_depth_start = po.metadata.get('core depth start', 'N/A')
    box_depth_stop = po.metadata.get('core depth stop', 'N/A')
    
    # Heading at top of this section
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x_margin, y_top - 15, f"Box {box_num} - {gen_display_text(key)}")
    
    # Metadata
    c.setFont("Helvetica", 8)
    c.drawString(x_margin, y_top - 28, f"Depth: {box_depth_start}m to {box_depth_stop}m")
    
    # Image area
    try:
        img_pil = _generate_box_image(po, key)
        
        img_buffer = BytesIO()
        img_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Available space for image (leave room for heading, metadata, and legend)
        available_width = width - 2 * x_margin
        available_height = (y_top - y_bottom) - 35 - legend_space  # 35 for heading/metadata
        
        img_width, img_height = img_pil.size
        scale = min(available_width / img_width, available_height / img_height, 1.0)
        
        display_width = img_width * scale
        display_height = img_height * scale
        
        # Center the image horizontally
        x = (width - display_width) / 2
        y = y_top - 35 - display_height  # y represents bottom left corner of image?
        legend_top_y = y - 2
        c.drawImage(ImageReader(img_buffer), x, y, width=display_width, height=display_height)
        
        # Draw legend or colorbar below the image
        if key.endswith("INDEX"):
            legend_key = key.replace("INDEX", "LEGEND")
            legend_ds = po.temp_datasets.get(legend_key) or po.datasets.get(legend_key)
            if legend_ds and legend_ds.data:
                _draw_legend_compact(c, legend_ds.data, x, legend_top_y, display_width)
        else:
            ds = po.temp_datasets.get(key) or po.datasets.get(key)
            if ds and ds.data is not None and ds.data.ndim==2:
                _draw_colorbar_compact(c, ds.data, po.mask, x, legend_top_y, display_width)
        
        logger.debug(f"Added image for Box {box_num}, key={key}")
        
    except Exception as e:
        logger.error(f"Failed to generate image for Box {box_num}, key={key}: {e}", exc_info=True)
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(x_margin, (y_top + y_bottom) / 2, f"Error generating image: {e}")


def _build_overview_page(c, hole, key):
    """Build overview page on canvas (portrait)."""
    width, height = A4
    margin = 1.5 * cm
    
    # Heading
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - margin - 20, f"Overview: {gen_display_text(key)} - All Boxes")
    
    # Concatenated image
    try:
        concat_img = _generate_concatenated_overview(hole, key)
        
        img_buffer = BytesIO()
        concat_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        available_width = width - 2 * margin
        available_height = height - 2 * margin - 0.8*inch
        
        img_width, img_height = concat_img.size
        scale = min(available_width / img_width, available_height / img_height, 1.0)
        
        display_width = img_width * scale
        display_height = img_height * scale
        
        # Center horizontally
        x = (width - display_width) / 2
        y = height - margin - 0.8*inch - display_height

        c.drawImage(ImageReader(img_buffer), x, y, width=display_width, height=display_height)
        if key.endswith("INDEX"):
            # Get legend from first box (they should all be the same)
            legend_key = key.replace("INDEX", "LEGEND")
            first_box = list(hole.boxes.values())[0]
            legend_ds = first_box.temp_datasets.get(legend_key) or first_box.datasets.get(legend_key)
            if legend_ds and legend_ds.data:
                _draw_legend(c, legend_ds.data, concat_img, x, y, display_width)
        else:
            # For colorbar, get data from first box
            first_box = list(hole.boxes.values())[0]
            ds = first_box.temp_datasets.get(key) or first_box.datasets.get(key)
            mask = first_box.mask if first_box.has('mask') else None
            if ds and ds.data is not None:
                _draw_colorbar(c, ds.data, mask, x, y, display_width)
        logger.debug(f"Added overview for key={key}")
        
    except Exception as e:
        logger.error(f"Failed to generate overview for key={key}: {e}", exc_info=True)
        c.setFont("Helvetica-Oblique", 11)
        c.drawString(margin, height/2, f"Error generating overview: {e}")


def _generate_box_image(po, key: str) -> Image.Image:
    """Generate full-resolution image for box page."""
    ds = po.temp_datasets.get(key) or po.datasets.get(key)
    if ds is None:
        raise ValueError(f"Dataset key '{key}' not found in box {po.basename}")
    
    if ds.ext == ".npy" and getattr(ds.data, "ndim", 0) > 1:
        if key == "mask":
            return mk_thumb(ds.data, resize=False)
        elif key.endswith("INDEX"):
            return mk_thumb(ds.data, mask=po.mask, index_mode=True, resize=False)
        else:
            return mk_thumb(ds.data, mask=po.mask, resize=False)
    elif ds.ext == ".npz":
        return mk_thumb(ds.data.data, mask=ds.data.mask, resize=False)
    else:
        raise ValueError(f"Cannot generate image for dataset type {ds.ext}")
    

def _generate_concatenated_overview(hole: HoleObject, key: str) -> Image.Image:
    """
    Generate a concatenated overview image for a specific key across all boxes.
    
    All boxes are concatenated vertically in box number order using cached thumbnails.
    """
    images = []
    
    # Get thumbnail for each box
    for po in hole:
        ds = po.temp_datasets.get(key) or po.datasets.get(key)
        if ds is None or ds.thumb is None:
            logger.warning(f"Skipping box {po.basename} - no thumbnail for key '{key}'")
            continue
        images.append(ds.thumb)
    
    if not images:
        raise ValueError(f"No thumbnails available for key '{key}'. Run 'Generate Images' first.")
    
    # Concatenate vertically
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    max_width = max(widths)
    total_height = sum(heights)
    
    concatenated = Image.new('RGB', (max_width, total_height))
    
    y_offset = 0
    for img in images:
        x_offset = (max_width - img.width) // 2
        concatenated.paste(img, (x_offset, y_offset))
        y_offset += img.height
    
    return concatenated


def _draw_legend_compact(c, legend: list[dict], x_start, y_bottom, width):
    """Draw compact horizontal legend for INDEX datasets below image in tight space."""
       
    # Get colors from tab20
    cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]
    
    # Build index to label mapping
    idx_to_label = {}
    for row in legend or []:
        try:
            idx = int(row.get("index"))
            lab = str(row.get("label", f"class {idx}"))
            idx_to_label[idx] = lab
        except:
            continue
    
    if not idx_to_label:
        return
    
    # Compact layout parameters for half-page
    box_size = 8  # Smaller boxes
    text_gap = 2
    
    # Calculate item width based on longest label
    c.setFont("Helvetica", 6)  # Smaller font
    max_text_width = max([c.stringWidth(label, "Helvetica", 6) for label in idx_to_label.values()])
    item_width = box_size + text_gap + max_text_width + 8
    items_per_row = max(1, int(width / item_width))
    row_height = 12  # Tighter vertical spacing
    
    x_start_pos = x_start
    y_start_pos = y_bottom-15
    
    row = 0
    col = 0
    
    for idx in sorted(idx_to_label.keys()):
        # Calculate position
        x = x_start_pos + (col * item_width)
        y = y_start_pos - (row * row_height)
        
        # Draw color box
        color = cmap(idx % 20)[:3]
        c.setFillColorRGB(color[0], color[1], color[2])
        c.rect(x, y, box_size, box_size, fill=1, stroke=1)
        
        # Draw label
        c.setFillColorRGB(0, 0, 0)
        label = idx_to_label[idx]
        c.drawString(x + box_size + text_gap, y + 1, label)
        
        # Move to next position
        col += 1
        if col >= items_per_row:
            col = 0
            row += 1


def _draw_colorbar_compact(c, data, mask, x_start, y_bottom, width):
    """Draw compact horizontal colorbar for continuous data below image in tight space."""
    # Determine stretch range (same logic as mk_thumb)
    if mask is not None:
        a = np.ma.masked_array(data, mask=mask).astype(float)
    else:
        a = np.ma.array(data, dtype=float)
    
    
    data_min = np.nanmin(a)
    data_max = np.nanmax(a)
    
    # Detect range type
    if data_min >= 0 and np.sum((a.compressed() >= 0) & (a.compressed() <= 1)) >= 0.95 * a.compressed().size:
        amin, amax = 0.0, 1.0
    else:
        in_range = False
        compressed = a.compressed()
        for range_min, range_max in DISPLAY_RANGE.values():
            valid_data = compressed[(compressed >= range_min) & (compressed <= range_max)]
            if valid_data.size > 0.7 * compressed.size:
                amin, amax = range_min, range_max
                in_range = True
                break
        if not in_range:
            amin, amax = data_min, data_max
    
    # Draw compact horizontal colorbar
    my_map = matplotlib.colormaps['viridis']
    bar_height = 10  # Smaller height
    bar_width = min(width * 0.6, 250)  # Scale with image width, max 250
    
    x_bar = x_start
    y_pos = y_bottom - 25
    num_steps = 100
    step_width = bar_width / num_steps
    
    for i in range(num_steps):
        val = i / num_steps
        color = my_map(val)[:3]
        c.setFillColorRGB(color[0], color[1], color[2])
        c.rect(x_bar + i * step_width, y_pos, step_width, bar_height, fill=1, stroke=0)
    
    # Draw border
    c.setStrokeColorRGB(0, 0, 0)
    c.rect(x_bar, y_pos, bar_width, bar_height, fill=0, stroke=1)
    
    # Draw labels below the colorbar
    c.setFont("Helvetica", 6)  # Smaller font
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x_bar, y_pos - 10, f"{amin:.0f}")
    c.drawRightString(x_bar + bar_width, y_pos - 10, f"{amax:.0f}")


# Keep original legend/colorbar functions for overview pages
def _draw_legend(c, legend: list[dict], img_data, x_start, y_bottom, width):
    """Draw horizontal legend for INDEX datasets below image, wrapping to multiple rows."""
       
    # Get colors from tab20
    cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]
    
    # Build index to label mapping
    idx_to_label = {}
    for row in legend or []:
        try:
            idx = int(row.get("index"))
            lab = str(row.get("label", f"class {idx}"))
            idx_to_label[idx] = lab
        except:
            continue
    
    if not idx_to_label:
        return
    
    # Layout parameters
    box_size = 12
    text_gap = 3
    
    # Calculate item width based on longest label
    c.setFont("Helvetica", 7)
    max_text_width = max([c.stringWidth(label, "Helvetica", 7) for label in idx_to_label.values()])
    item_width = box_size + text_gap + max_text_width + 10
    items_per_row = max(1, int(width / item_width))
    row_height = 18
    
    x_start_pos = x_start
    y_start_pos = y_bottom - 30
    
    row = 0
    col = 0
    
    for idx in sorted(idx_to_label.keys()):
        # Calculate position
        x = x_start_pos + (col * item_width)
        y = y_start_pos - (row * row_height)
        
        # Draw color box
        color = cmap(idx % 20)[:3]
        c.setFillColorRGB(color[0], color[1], color[2])
        c.rect(x, y, box_size, box_size, fill=1, stroke=1)
        
        # Draw label
        c.setFillColorRGB(0, 0, 0)
        label = idx_to_label[idx]
        c.drawString(x + box_size + text_gap, y + 2, label)
        
        # Move to next position
        col += 1
        if col >= items_per_row:
            col = 0
            row += 1


def _draw_colorbar(c, data, mask, x_start, y_bottom, width):
    """Draw horizontal colorbar for continuous data below image."""
    # Determine stretch range (same logic as mk_thumb)
    if mask is not None:
        a = np.ma.masked_array(data, mask=mask).astype(float)
    else:
        a = np.ma.array(data, dtype=float)
    
    data_min = np.nanmin(a)
    data_max = np.nanmax(a)
    
    # Detect range type
    if data_min >= 0 and np.sum((a.compressed() >= 0) & (a.compressed() <= 1)) >= 0.95 * a.compressed().size:
        amin, amax = 0.0, 1.0
    else:
        in_range = False
        compressed = a.compressed()
        for range_min, range_max in DISPLAY_RANGE.values():
            valid_data = compressed[(compressed >= range_min) & (compressed <= range_max)]
            if valid_data.size > 0.7 * compressed.size:
                amin, amax = range_min, range_max
                in_range = True
                break
        if not in_range:
            amin, amax = data_min, data_max
    
    # Draw horizontal colorbar
    my_map = matplotlib.colormaps['viridis']
    bar_height = 15
    bar_width = 300
    
    x_bar = x_start + 20
    y_pos = y_bottom - 35
    num_steps = 100
    step_width = bar_width / num_steps
    
    for i in range(num_steps):
        val = i / num_steps
        color = my_map(val)[:3]
        c.setFillColorRGB(color[0], color[1], color[2])
        c.rect(x_bar + i * step_width, y_pos, step_width, bar_height, fill=1, stroke=0)
    
    # Draw border
    c.setStrokeColorRGB(0, 0, 0)
    c.rect(x_bar, y_pos, bar_width, bar_height, fill=0, stroke=1)
    
    # Draw labels below the colorbar
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x_bar, y_pos - 12, f"{amin:.0f}")
    c.drawRightString(x_bar + bar_width, y_pos - 12, f"{amax:.0f}")


def _render_downhole_plot_to_buffer(
    depths: np.ndarray,
    values: np.ndarray,
    plot_type: str,
    legend: list[dict] = None,
    title: str = "",
    figsize: tuple = (8, 10)
) -> BytesIO:
    """
    Render downhole plot to PNG buffer.
    
    Uses Figure directly without backend manipulation - works with
    whatever backend is already loaded (Qt5Agg in this application).
    """
    # Create figure - uses current backend's renderer automatically
    fig = Figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111)
    
    # Ensure depth is ascending
    if depths[0] > depths[-1]:
        depths = depths[::-1]
        if values.ndim == 1:
            values = values[::-1]
        else:
            values = values[::-1, :]
    
    if plot_type == 'continuous':
        ax.plot(values, depths, 'o-', markersize=3)
        ax.set_xlabel(title)
        ax.set_ylabel("Depth (m)")
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    
    elif plot_type == 'discrete':
        cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]
        
        index_to_color = {}
        legend_handles = []
        legend_labels = []
        
        for i, entry in enumerate(legend):
            mineral_id = int(entry["index"])
            color = cmap(mineral_id % 20)
            index_to_color[i] = color
            
            from matplotlib.patches import Patch
            legend_handles.append(Patch(facecolor=color))
            legend_labels.append(entry["label"])
        
        # No data color
        index_to_color[-1] = (1.0, 1.0, 1.0, 1.0)
        legend_handles.append(Patch(facecolor=(1.0, 1.0, 1.0, 1.0)))
        legend_labels.append("No Dominant / Gap")
        
        # Draw bars
        width = 0.1
        H = values.shape[0]
        for i in range(H):
            idx = values[i]
            z_top = depths[i]
            z_bottom = depths[i+1] if i + 1 < H else depths[-1] + (depths[-1] - depths[-2])
            color = index_to_color.get(idx, (0.5, 0.5, 0.5, 1.0))
            
            ax.barh(
                y=z_top,
                width=width,
                height=z_bottom - z_top,
                left=0,
                align='edge',
                color=color,
                edgecolor='none'
            )
        
        ax.set_ylim(depths.min(), depths.max())
        ax.invert_yaxis()
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel(title)
        ax.set_xlim(0.0, width)
        ax.set_xticks([])
        
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8
        )
    
    elif plot_type == 'fractions':
        H, C = values.shape
        K = C - 1
        
        col_sums = np.sum(values, axis=0)
        cols_to_plot = [i for i in range(C) if col_sums[i] > 0]
        
        frac_use = values[:, cols_to_plot]
        cum = np.cumsum(frac_use, axis=1)
        left = np.hstack([np.zeros((H, 1)), cum[:, :-1]])
        right = cum
        
        cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]
        
        for band_idx, col_idx in enumerate(cols_to_plot):
            if col_idx < K:
                cid = int(legend[col_idx]["index"])
                name = str(legend[col_idx]["label"])
                color = cmap(cid % 20)
            else:
                name = "Unclassified"
                color = (0.7, 0.7, 0.7, 1.0)
            
            ax.fill_betweenx(
                depths,
                left[:, band_idx],
                right[:, band_idx],
                step="pre",
                facecolor=color,
                edgecolor="none",
                label=name
            )
        
        ax.set_ylim(depths.min(), depths.max())
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Fraction of row width")
        ax.set_ylabel("Depth (m)")
        ax.grid(True, axis="x", alpha=0.2)
        
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8
        )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    # Save to buffer - Figure.savefig() works with current backend
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    
    # Clean up
    fig.clear()
    del fig
    
    return buf


def _build_downhole_plots_section(c, hole, selected_product_keys):
    """
    Add downhole plot pages for selected product datasets.
    
    Uses HoleObject.step_product_dataset() to get resampled data,
    then renders plots using the same logic as ImageCanvas2D.
    """
    from reportlab.lib.utils import ImageReader
    
    for key in selected_product_keys:
        try:
            # Use existing resampling logic
            depths, values, dominant = hole.step_product_dataset(key)
        except ValueError as e:
            logger.warning(f"Cannot plot {key}: {e}")
            continue
        
        # Determine plot type from suffix (same as hole_page.show_downhole)
        if key.endswith("FRACTIONS"):
            plot_type = 'fractions'
            legend_key = key.replace("FRACTIONS", "LEGEND")
            legend = hole.product_datasets[legend_key].data
            
        elif key.endswith("DOM-MIN"):
            plot_type = 'discrete'
            legend_key = key.replace("DOM-MIN", "LEGEND")
            legend = hole.product_datasets[legend_key].data
            values = dominant  # Use dominant indices, not fractions
            
        elif key.endswith("INDEX"):
            plot_type = 'discrete'
            legend_key = key.replace("INDEX", "LEGEND")
            legend = hole.product_datasets[legend_key].data
            
        else:
            plot_type = 'continuous'
            legend = None
        
        # Render plot
        img_buffer = _render_downhole_plot_to_buffer(
            depths=depths,
            values=values,
            plot_type=plot_type,
            legend=legend,
            title=gen_display_text(key),
            figsize=(8, 11) if plot_type == 'discrete' else (7, 10)
        )
        
        # Add page
        c.setPageSize(A4)  # Portrait
        width, height = A4
        
        # Draw title
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width/2, height - 30, f"Downhole Plot: {hole.hole_id}")
        
        # Draw plot image
        img_reader = ImageReader(img_buffer)
        
        # Calculate dimensions to fit page with margins
        margin = 40
        available_width = width - 2*margin
        available_height = height - 80  # Space for title
        
        c.drawImage(
            img_reader,
            margin,
            margin,
            width=available_width,
            height=available_height,
            preserveAspectRatio=True,
            anchor='c'
        )
        
        c.showPage()


def create_po_pdf_booklet(
    po,
    output_path: Path | str,
    include_metadata: bool = True
) -> Path:
    """
    Create a PDF booklet for a single ProcessedObject with selected dataset visualizations.
    
    Parameters
    ----------
    po : ProcessedObject
        The ProcessedObject containing the datasets
    output_path : Path | str
        Path where the PDF should be saved
    include_metadata : bool, default=True
        Whether to include a title page with metadata
        
    Returns
    -------
    Path
        Path to the created PDF file
        
    Notes
    -----
    PDF structure:
    1. Optional title page with metadata (portrait)
    2. Landscape pages with 2 datasets per page (stacked vertically)
    """
    output_path = Path(output_path) / f"{po.basename}.pdf"
    exclude_keys = ["savgol", "cropped", "savgol_cr", "metadata", "stats", "bands", "display"]
    selected_keys = [key for key in po.datasets.keys() if key not in exclude_keys
                     and not key.endswith("LEGEND")
                     and not key.endswith("CLUSTERS")]
    
    logger.info(f"Creating PDF booklet for {po.basename} with {len(selected_keys)} keys")
    
    # Create canvas
    c = pdf_canvas.Canvas(str(output_path), pagesize=A4)
    
    # ========================================================================
    # 1. TITLE PAGE (Portrait) - Optional
    # ========================================================================
    if include_metadata:
        _build_po_title_page(c, po, selected_keys)
        c.showPage()
    
    # ========================================================================
    # 2. DATASET PAGES (Landscape - 2 datasets per page, stacked vertically)
    # ========================================================================
    logger.info(f"Generating dataset pages (2 per page)...")
    
    width_landscape, height_landscape = landscape(A4)
    margin = 1.0 * cm
    divider_y = height_landscape / 2
    legend_space = 50
    
    # Process datasets in pairs
    for i in range(0, len(selected_keys), 2):
        c.setPageSize(landscape(A4))
        
        key1 = selected_keys[i]
        key2 = selected_keys[i + 1] if i + 1 < len(selected_keys) else None
        
        # Top half - use existing function with key1
        _build_single_box_segment(c, po, key1, 
                                 x_margin=margin, 
                                 y_top=height_landscape - margin, 
                                 y_bottom=divider_y + 5,
                                 width=width_landscape,
                                 legend_space=legend_space)
        
        # Bottom half - use existing function with key2
        if key2 is not None:
            _build_single_box_segment(c, po, key2, 
                                     x_margin=margin, 
                                     y_top=divider_y - 5,
                                     y_bottom=margin,
                                     width=width_landscape,
                                     legend_space=legend_space)
        
        c.showPage()
    
    # Save PDF
    c.save()
    logger.info(f"PDF booklet created successfully: {output_path}")
    return output_path

def _build_po_title_page(c, po, selected_keys):
    """Build title page for a single ProcessedObject (portrait)."""
    width, height = A4
    margin = 1.5 * cm
    
    # Title
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width/2, height - 2*inch, f"Core Box Report: {po.basename}")
    
    # Metadata
    y = height - 2.8*inch
    
    # Get metadata if available
    metadata = po.metadata if hasattr(po, 'metadata') else {}
    
    # Basename
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Box ID:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + 100, y, str(po.basename))
    y -= 20
    
    # Core depth information if available
    if metadata:
        depth_start = metadata.get("core depth start", "N/A")
        depth_stop = metadata.get("core depth stop", "N/A")
        
        if depth_start != "N/A" and depth_stop != "N/A":
            try:
                ds = float(depth_start)
                de = float(depth_stop)
                
                c.setFont("Helvetica-Bold", 11)
                c.drawString(margin, y, "Depth Range:")
                c.setFont("Helvetica", 11)
                c.drawString(margin + 100, y, f"{ds:.2f}m to {de:.2f}m")
                y -= 20
                
                c.setFont("Helvetica-Bold", 11)
                c.drawString(margin, y, "Box Length:")
                c.setFont("Helvetica", 11)
                c.drawString(margin + 100, y, f"{de - ds:.2f}m")
                y -= 20
            except (ValueError, TypeError):
                pass
        
        # Sample date if available
        sample_date = metadata.get("sample date", None)
        if sample_date:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, "Sample Date:")
            c.setFont("Helvetica", 11)
            c.drawString(margin + 100, y, str(sample_date))
            y -= 20
    
    # Number of datasets
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Available Datasets:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + 100, y, str(len(po.datasets) + len(po.temp_datasets)))
    y -= 20
    
    # Report generation time
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Report Generated:")
    c.setFont("Helvetica", 11)
    c.drawString(margin + 100, y, datetime.now().strftime('%Y-%m-%d %H:%M'))
    y -= 30
    
    # Selected datasets list with wrapping
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Included Datasets:")
    y -= 20
    
    c.setFont("Helvetica", 10)
    max_width = width - 2 * margin - 20  # Available width for text
    
    for i, key in enumerate(selected_keys, 1):
        display_name = gen_display_text(key)
        full_text = f"{i}. {display_name}"
        
        # Check if text fits on one line
        text_width = c.stringWidth(full_text, "Helvetica", 10)
        
        if text_width <= max_width:
            # Fits on one line
            c.drawString(margin + 20, y, full_text)
            y -= 15
        else:
            # Need to wrap - split at reasonable point
            # Simple approach: just truncate with ellipsis
            while c.stringWidth(full_text + "...", "Helvetica", 10) > max_width and len(full_text) > 10:
                full_text = full_text[:-1]
            c.drawString(margin + 20, y, full_text + "...")
            y -= 15
        
        if y < margin + 50:  # Avoid going off page
            c.drawString(margin + 20, y, "... (additional datasets not shown)")
            break