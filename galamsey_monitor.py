import io
import os
import sys
import tempfile
import threading
import traceback

import cv2
import ee
import folium
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from PyQt6.QtCore import (QDate, QMetaObject, QObject, Qt, QUrl,
                          pyqtSignal, Q_ARG)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (QApplication, QDateEdit, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMessageBox, QProgressDialog,
                             QPushButton, QSizePolicy, QSlider, QSpinBox,
                             QSplitter, QStyle, QTextEdit, QVBoxLayout,
                             QWidget)


# --- Worker Signals ---
class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    frame_processed = pyqtSignal(int, int)


# --- Cloud Masking Function (SCL) ---
def mask_s2_clouds_scl(image):
    scl = image.select('SCL')
    # Keep clear (4), water (6), and snow/ice (11).
    # Mask out saturated/defective (0), dark area pixels (1), cloud shadows (3),
    # vegetation (5 -> not for cloud mask), not vegetated (5 -> not for cloud mask),
    # clouds medium probability (8), clouds high probability (9), cirrus (10).
    # SCL values to keep: 2 (Dense Dark Vegetation), 4 (Bare Soils),
    # 5 (Vegetation), 6 (Water), 7 (Unclassified), 11 (Snow/Ice)
    # SCL values to mask (clouds, shadows, etc.): 0, 1, 3, 8, 9, 10
    mask = scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(
        scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000).copyProperties(
        image, ["system:time_start"])


# --- GEE Single Frame Processing Logic ---
def process_single_gee_frame(aoi_rectangle_coords, period1_start,
                             period1_end, period2_start, period2_end,
                             threshold_val, dimensions_val, project_id,
                             progress_emitter=None, output_type='image'):
    aoi = ee.Geometry.Rectangle(aoi_rectangle_coords)

    def calculate_ndvi(image):
        return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    rgb_viz_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3,
                      'gamma': 1.4}
    loss_viz_params = {'palette': ['red']}

    if progress_emitter:
        progress_emitter(
            f"Frame: Processing Baseline ({period1_start}-{period1_end})...")
    collection_p1_base = s2_sr.filterBounds(aoi).filterDate(
        period1_start, period1_end).map(mask_s2_clouds_scl)
    count1 = collection_p1_base.size().getInfo()
    if count1 == 0:
        if progress_emitter:
            progress_emitter(
                "Frame: No suitable cloud-free images for Baseline period.")
        return None
    median_ndvi_p1 = collection_p1_base.map(calculate_ndvi).select(
        'NDVI').median()

    if progress_emitter:
        progress_emitter(
            f"Frame: Processing Period 2 ({period2_start}-{period2_end})...")
    collection_p2_base = s2_sr.filterBounds(aoi).filterDate(
        period2_start, period2_end).map(mask_s2_clouds_scl)
    count2 = collection_p2_base.size().getInfo()
    if count2 == 0:
        if progress_emitter:
            progress_emitter(
                "Frame: No suitable cloud-free images for Period 2.")
        return None
    median_ndvi_p2 = collection_p2_base.map(calculate_ndvi).select(
        'NDVI').median()
    median_rgb_p2 = collection_p2_base.select(['B4', 'B3', 'B2']).median()

    if progress_emitter:
        progress_emitter("Frame: Calculating NDVI change...")
    ndvi_change = median_ndvi_p2.subtract(median_ndvi_p1).rename(
        'NDVI_Change')
    loss_mask = ndvi_change.lt(threshold_val).selfMask()

    if progress_emitter:
        progress_emitter("Frame: Creating visual layers...")
    background_layer = median_rgb_p2.visualize(**rgb_viz_params)
    loss_overlay = loss_mask.visualize(**loss_viz_params)
    final_image_viz = ee.ImageCollection(
        [background_layer, loss_overlay]).mosaic().clip(aoi)

    if output_type == 'image':
        if progress_emitter:
            progress_emitter("Frame: Generating thumbnail for image output...")

        if isinstance(dimensions_val, tuple):
            dims_str = f"{dimensions_val[0]}x{dimensions_val[1]}"
        elif isinstance(dimensions_val, str) and 'x' in dimensions_val:
            dims_str = dimensions_val
        elif isinstance(dimensions_val, int):
            dims_str = str(dimensions_val)
        else:
            if progress_emitter:
                progress_emitter(
                    "Warning: Invalid dimensions_val for GEE thumbnail. "
                    "Defaulting to 512.")
            dims_str = "512"

        thumb_url = None
        try:
            thumb_params = {'region': aoi.getInfo()['coordinates'],
                            'dimensions': dims_str, 'format': 'png'}
            thumb_url = final_image_viz.getThumbURL(thumb_params)
        except ee.EEException as thumb_e:
            err_msg = str(thumb_e)
            if "No valid pixels" in err_msg or \
                    "Image.select: Pattern 'constant' did not match any bands." in err_msg:
                if progress_emitter:
                    progress_emitter(
                        f"Frame: Thumbnail error ({thumb_e}). "
                        f"Trying background only for image.")
                try:
                    thumb_url = background_layer.getThumbURL(thumb_params)
                except Exception as bg_thumb_e:
                    if progress_emitter:
                        progress_emitter(
                            "Frame: Error getting background thumbnail for "
                            f"image: {bg_thumb_e}")
                    return None
            else:
                if progress_emitter:
                    progress_emitter(
                        f"Frame: GEE error generating thumbnail: {thumb_e}")
                return None
        if not thumb_url:
            if progress_emitter:
                progress_emitter(
                    "Frame: Failed to generate thumbnail URL for image.")
            return None

        if progress_emitter:
            progress_emitter(f"Frame: Downloading image ({dims_str})...")
        response = requests.get(thumb_url, timeout=60)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    elif output_type == 'map_id':
        if progress_emitter:
            progress_emitter("Frame: Generating MapID for interactive map...")
        try:
            map_id_dict = ee.data.getMapId({
                'image': final_image_viz,
                'region': aoi.getInfo()['coordinates']
            })
            bounds_coords = aoi.bounds(maxError=1).getInfo()['coordinates'][0]
            return {'map_id_dict': map_id_dict, 'aoi_bounds': bounds_coords}
        except Exception as e:
            if progress_emitter:
                progress_emitter(f"Error getting MapID: {e}")
            raise
    else:
        raise ValueError(
            f"Invalid output_type specified for GEE processing: {output_type}")


# --- GEE Single Analysis Worker ---
class GEEWorker(QObject):
    def __init__(self, aoi_rectangle_coords, start1, end1, start2, end2,
                 threshold, project_id='galamsey-monitor'):
        super().__init__()
        self.signals = WorkerSignals()
        self.aoi_rectangle_coords = aoi_rectangle_coords
        self.start1, self.end1 = start1, end1
        self.start2, self.end2 = start2, end2
        self.threshold = threshold
        self.project_id = project_id
        self.is_cancelled = False

    def run(self):
        try:
            ee.Initialize(project=self.project_id)
        except Exception:
            # Initialization might have happened already or failed;
            # subsequent GEE calls will raise errors if not initialized.
            pass

        try:
            analysis_data = process_single_gee_frame(
                self.aoi_rectangle_coords, self.start1, self.end1,
                self.start2, self.end2, self.threshold, "600x400",
                self.project_id, self.signals.progress.emit,
                output_type='map_id'
            )
            if self.is_cancelled:
                self.signals.error.emit(
                    "Single analysis was cancelled during processing.")
                return
            if analysis_data:
                self.signals.finished.emit(analysis_data)
            else:
                self.signals.error.emit(
                    "Failed to generate map data from GEE "
                    "(e.g., no images found).")
        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(
                f"Error in GEEWorker (map_id): {e}\nTrace: {tb_str}")


# --- Time-Lapse Generation Worker (with Overlays) ---
class TimeLapseWorker(QObject):
    def __init__(self, aoi_rectangle_coords, baseline_start_date,
                 baseline_end_date, timelapse_start_year,
                 timelapse_end_year, threshold,
                 output_dimensions=(1280, 720),
                 project_id='galamsey-monitor',
                 output_video_path="galamsey_timelapse.mp4", fps=1,
                 aoi_name="Default AOI", raw_input_coords=None):
        super().__init__()
        self.signals = WorkerSignals()
        self.aoi_rectangle_coords = aoi_rectangle_coords
        self.baseline_start_date = baseline_start_date
        self.baseline_end_date = baseline_end_date
        self.timelapse_start_year = timelapse_start_year
        self.timelapse_end_year = timelapse_end_year
        self.threshold = threshold
        self.output_dimensions = output_dimensions
        self.project_id = project_id
        self.output_video_path = output_video_path
        self.fps = fps
        self.is_cancelled = False
        self.aoi_name = aoi_name
        self.raw_input_coords = raw_input_coords if raw_input_coords else [
            0.0, 0.0, 0.0, 0.0]

    def _get_font(self, path, size, default_message_emitter):
        try:
            return ImageFont.truetype(path, size)
        except IOError:
            default_message_emitter(
                f"Warning: Font '{path}' not found. Using default PIL font.")
            return ImageFont.load_default()

    def _get_text_size(self, draw_instance, text, font_instance):
        try:  # Modern PIL
            bbox = draw_instance.textbbox((0, 0), text, font=font_instance)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:  # Older PIL
            return draw_instance.textsize(text, font=font_instance)

    def _draw_text_with_background(self, draw, text_content, font, position,
                                   text_color, bg_color, padding):
        text_w, text_h = self._get_text_size(draw, text_content, font)
        bg_x0, bg_y0 = position
        bg_x1 = bg_x0 + text_w + 2 * padding
        bg_y1 = bg_y0 + text_h + 2 * padding
        draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=bg_color)
        draw.text((bg_x0 + padding, bg_y0 + padding), text_content,
                  font=font, fill=text_color)
        return bg_x0, bg_y0, bg_x1, bg_y1  # Return bounding box of background

    def run(self):
        try:
            ee.Initialize(project=self.project_id)
        except Exception:
            # GEE might already be initialized or init might fail.
            # Subsequent calls will raise errors if not initialized.
            pass

        video_output_width, video_output_height = self.output_dimensions
        video_frame_size = (video_output_width, video_output_height)
        self.signals.progress.emit(
            f"Video frame size set to: {video_output_width}x{video_output_height}")

        text_color = (255, 255, 255)  # White
        bg_color_semi_transparent = (0, 0, 0, 150)  # Black, semi-transparent
        text_padding = 8
        font_path = "arial.ttf"

        font_title = self._get_font(font_path, 28, self.signals.progress.emit)
        font_coords_header = self._get_font(font_path, 18, self.signals.progress.emit)
        font_coords_value = self._get_font(font_path, 16, self.signals.progress.emit)
        font_bottom_info = self._get_font(font_path, 14, self.signals.progress.emit)
        font_year_info = self._get_font(font_path, 22, self.signals.progress.emit)

        video_writer_initialized = False
        out_video = None
        total_frames = self.timelapse_end_year - self.timelapse_start_year + 1
        if total_frames <= 0:
            self.signals.error.emit(
                "No frames requested (start year not before end year).")
            return

        try:
            for i, year in enumerate(
                    range(self.timelapse_start_year,
                          self.timelapse_end_year + 1)):
                if self.is_cancelled:
                    self.signals.progress.emit(
                        "Time-lapse generation cancelled by user.")
                    break
                self.signals.frame_processed.emit(i + 1, total_frames)
                self.signals.progress.emit(
                    f"Processing GEE data for year: {year} ({i + 1}/{total_frames})")

                gee_pil_image = process_single_gee_frame(
                    self.aoi_rectangle_coords, self.baseline_start_date,
                    self.baseline_end_date, f"{year}-01-01", f"{year}-12-31",
                    self.threshold, self.output_dimensions, self.project_id,
                    self.signals.progress.emit, output_type='image'
                )

                current_frame_rgba: Image.Image
                if gee_pil_image:
                    if gee_pil_image.size != video_frame_size:
                        self.signals.progress.emit(
                            f"Resizing GEE image from {gee_pil_image.size} "
                            f"to {video_frame_size}")
                        gee_pil_image = gee_pil_image.resize(
                            video_frame_size, Image.Resampling.LANCZOS)
                    current_frame_rgba = gee_pil_image.convert('RGBA') \
                        if gee_pil_image.mode != 'RGBA' else gee_pil_image.copy()
                else:
                    current_frame_rgba = Image.new('RGBA', video_frame_size,
                                                   (0, 0, 0, 255)) # Solid black
                    draw_temp = ImageDraw.Draw(current_frame_rgba)
                    no_data_text = f"NO DATA FOR {year}"
                    text_w, text_h = self._get_text_size(draw_temp,
                                                         no_data_text, font_title)
                    draw_temp.text(
                        ((video_output_width - text_w) / 2,
                         (video_output_height - text_h) / 2),
                        no_data_text, font=font_title, fill=text_color
                    )

                draw = ImageDraw.Draw(current_frame_rgba)

                # --- Coordinates (Top-Left) ---
                coord_texts = [
                    ("Top-left-corner:", font_coords_header, 3),
                    (f"  Lat 1: {self.raw_input_coords[0]:.6f}", font_coords_value, 3),
                    (f"  Lon 1: {self.raw_input_coords[1]:.6f}", font_coords_value, 8),
                    ("Bottom-right-corner:", font_coords_header, 3),
                    (f"  Lat 2: {self.raw_input_coords[2]:.6f}", font_coords_value, 3),
                    (f"  Lon 2: {self.raw_input_coords[3]:.6f}", font_coords_value, 0),
                ]
                max_coord_w = 0
                coords_block_total_h = 0
                coord_line_heights = []

                for text_content, font, _ in coord_texts:
                    # Use a consistent line height metric based on font
                    _, font_line_h = self._get_text_size(draw, "Ay", font)
                    coord_line_heights.append(font_line_h)
                    text_w, _ = self._get_text_size(draw, text_content, font)
                    max_coord_w = max(max_coord_w, text_w)

                for idx, (_, _, spacing) in enumerate(coord_texts):
                    coords_block_total_h += coord_line_heights[idx] + spacing
                if coord_texts: # Remove last spacing if any texts
                    coords_block_total_h -= coord_texts[-1][2]

                coords_bg_x0 = text_padding
                coords_bg_y0 = text_padding # Align with top
                coords_bg_width = max_coord_w + 2 * text_padding
                coords_bg_height_with_padding = coords_block_total_h + 2 * text_padding

                # Draw Coordinates Background
                draw.rectangle(
                    [coords_bg_x0, coords_bg_y0,
                     coords_bg_x0 + coords_bg_width,
                     coords_bg_y0 + coords_bg_height_with_padding],
                    fill=bg_color_semi_transparent
                )
                # Draw Coordinates Text
                current_y_for_coord_text = coords_bg_y0 + text_padding
                for idx, (text_content, font, spacing) in enumerate(coord_texts):
                    draw.text((coords_bg_x0 + text_padding, current_y_for_coord_text),
                              text_content, font=font, fill=text_color)
                    current_y_for_coord_text += coord_line_heights[idx] + spacing

                coords_bg_x1_edge = coords_bg_x0 + coords_bg_width # Right edge of coords block

                # --- AOI Name (Positioned to the right of Coordinates block) ---
                aoi_text_content = self.aoi_name
                aoi_w, aoi_h = self._get_text_size(draw, aoi_text_content, font_title)
                aoi_bg_total_width = aoi_w + 2 * text_padding

                # Attempt to center AOI title in the space right of the coordinates block
                # or place it directly after if space is limited.
                available_width_for_aoi = video_output_width - coords_bg_x1_edge - (2 * text_padding)

                if aoi_bg_total_width < available_width_for_aoi:
                    aoi_pos_x = coords_bg_x1_edge + text_padding + \
                                (available_width_for_aoi - aoi_bg_total_width) / 2
                else: # Not enough space to center, or coord block is too wide
                    aoi_pos_x = coords_bg_x1_edge + text_padding

                aoi_pos_y = text_padding # Align with top, same as coordinates block

                self._draw_text_with_background(
                    draw, aoi_text_content, font_title, (aoi_pos_x, aoi_pos_y),
                    text_color, bg_color_semi_transparent, text_padding
                )

                # --- Year (Bottom-Right) ---
                year_text = f"YEAR: {year}"
                year_w, year_h = self._get_text_size(draw, year_text, font_year_info)
                # Position from bottom-right corner
                year_bg_x0 = video_output_width - (year_w + 2 * text_padding) - text_padding
                year_bg_y0 = video_output_height - (year_h + 2 * text_padding) - text_padding
                self._draw_text_with_background(
                    draw, year_text, font_year_info, (year_bg_x0, year_bg_y0),
                    text_color, bg_color_semi_transparent, text_padding
                )

                # --- Bottom-Left Info (Loss, Copyright, Rights) ---
                bl_texts = [
                    ("Red shows potential Vegetative Loss", font_bottom_info, 3),
                    ("Copyright KilTech Ent 2025", font_bottom_info, 3),
                    ("All rights reserved.", font_bottom_info, 0)
                ]
                max_bl_w = 0
                bl_block_total_h = 0
                bl_line_heights = []

                for text_content, font, _ in bl_texts:
                    _, font_line_h = self._get_text_size(draw, "Ay", font)
                    bl_line_heights.append(font_line_h)
                    text_w, _ = self._get_text_size(draw, text_content, font)
                    max_bl_w = max(max_bl_w, text_w)

                for idx, (_, _, spacing) in enumerate(bl_texts):
                    bl_block_total_h += bl_line_heights[idx] + spacing
                if bl_texts: # Remove last spacing
                    bl_block_total_h -= bl_texts[-1][2]

                bl_bg_x0 = text_padding
                bl_bg_height_with_padding = bl_block_total_h + 2 * text_padding
                bl_bg_y0 = video_output_height - bl_bg_height_with_padding - text_padding # From bottom
                bl_bg_width = max_bl_w + 2 * text_padding

                draw.rectangle(
                    [bl_bg_x0, bl_bg_y0,
                     bl_bg_x0 + bl_bg_width,
                     bl_bg_y0 + bl_bg_height_with_padding],
                    fill=bg_color_semi_transparent
                )

                current_y_bl_text = bl_bg_y0 + text_padding
                for idx, (text_content, font, spacing) in enumerate(bl_texts):
                    draw.text((bl_bg_x0 + text_padding, current_y_bl_text),
                              text_content, font=font, fill=text_color)
                    current_y_bl_text += bl_line_heights[idx] + spacing

                # Convert to RGB for OpenCV
                final_frame_rgb = current_frame_rgba.convert('RGB')
                frame_cv = cv2.cvtColor(np.array(final_frame_rgb),
                                        cv2.COLOR_RGB2BGR)

                if not video_writer_initialized:
                    self.signals.progress.emit("Initializing video writer...")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_dir = os.path.dirname(self.output_video_path)
                    if video_dir and not os.path.exists(video_dir):
                        os.makedirs(video_dir, exist_ok=True)
                    out_video = cv2.VideoWriter(self.output_video_path, fourcc,
                                                self.fps, video_frame_size)
                    if not out_video.isOpened():
                        self.signals.error.emit(
                            f"FATAL: Could not open video writer for "
                            f"{self.output_video_path}.")
                        return
                    video_writer_initialized = True

                out_video.write(frame_cv)
                self.signals.progress.emit(f"Added frame for {year} to video.")

            if video_writer_initialized and out_video:
                self.signals.progress.emit("Finalizing video...")
                out_video.release()
                self.signals.progress.emit(
                    f"Video compilation complete: {self.output_video_path}")
                self.signals.finished.emit(self.output_video_path)
            elif not video_writer_initialized and total_frames > 0:
                self.signals.error.emit(
                    "No valid frames were processed to create the video.")

        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(
                f"Time-lapse overlay/compilation error: {e}\nTrace: {tb_str}")
            if video_writer_initialized and out_video:
                out_video.release()

# --- Main Application Window ---
class GalamseyMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galamsey Monitor")
        self.setGeometry(10, 40, 1400, 850)

        self.worker_thread = None
        self.worker = None
        self.timelapse_worker_thread = None
        self.timelapse_worker = None
        self.progress_dialog = None
        self.project_id = 'galamsey-monitor'
        self.map_html_temp_dir = None
        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self.final_video_path = None

        self._setup_ui()
        self._connect_signals()
        self.init_gee_check()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Single Period Analysis Group ---
        single_analysis_group = QGroupBox("Single Period Analysis")
        single_analysis_form = QFormLayout()
        self.aoi_name_input = QLineEdit()
        self.aoi_name_input.setPlaceholderText("e.g. Anyinam")
        self.aoi_name_input.setToolTip(
            "Descriptive name for AOI (Mandatory for Video).")
        single_analysis_form.addRow("AOI:", self.aoi_name_input)

        self.coord_input = QLineEdit("6.401452, -0.594587, 6.355603, -0.496084")
        self.coord_input.setToolTip(
            "Coords: Lat1, Lon1, Lat2, Lon2 (e.g., 6.3,-1.8,6.4,-1.7)")
        single_analysis_form.addRow("AOI Coords (Lat1,Lon1,Lat2,Lon2):",
                                    self.coord_input)

        self.date1_start = QDateEdit(QDate(2020, 1, 1))
        self.date1_end = QDateEdit(QDate(2020, 12, 31))
        self.date2_start = QDateEdit(QDate(2025, 1, 1))
        self.date2_end = QDateEdit(QDate.currentDate())
        for dt_edit in [self.date1_start, self.date1_end, self.date2_start,
                        self.date2_end]:
            dt_edit.setCalendarPopup(True)
            dt_edit.setDisplayFormat("dd-MM-yyyy")
        single_analysis_form.addRow("Period 1 Start (Baseline):",
                                    self.date1_start)
        single_analysis_form.addRow("Period 1 End (Baseline):", self.date1_end)
        single_analysis_form.addRow("Period 2 Start (Comparison):",
                                    self.date2_start)
        single_analysis_form.addRow("Period 2 End (Comparison):",
                                    self.date2_end)

        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(-1.0, 0.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setValue(-0.20)
        single_analysis_form.addRow("NDVI Change Threshold:",
                                    self.threshold_input)

        self.analyze_button = QPushButton("Analyze Single Period")
        single_analysis_form.addRow(self.analyze_button)
        single_analysis_group.setLayout(single_analysis_form)
        single_analysis_group.setMinimumWidth(450)

        # --- Interactive Map Group ---
        map_group = QGroupBox(
            "Interactive Map Preview (Red shows potential vegetation loss)")
        map_v_layout = QVBoxLayout()
        self.map_view = QWebEngineView()
        self.map_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        self.map_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.map_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.map_view.setHtml(
            "<html><body style='display:flex;justify-content:center;"
            "align-items:center;height:100%;font-family:sans-serif;"
            "color:grey;'><p>Map will appear here after analysis.</p>"
            "</body></html>")
        self.map_view.setMinimumSize(400, 300)
        map_v_layout.addWidget(self.map_view)
        map_group.setLayout(map_v_layout)

        # --- Top Horizontal Splitter ---
        top_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_h_splitter.addWidget(single_analysis_group)
        top_h_splitter.addWidget(map_group)
        top_h_splitter.setSizes([400, 700])

        # --- Time-Lapse Video Group ---
        timelapse_video_group = QGroupBox("Time-Lapse Video (720p)")
        main_timelapse_h_layout = QHBoxLayout()

        # Timelapse Controls Panel (Left)
        timelapse_controls_panel = QWidget()
        left_v_layout = QVBoxLayout(timelapse_controls_panel)
        left_v_layout.setContentsMargins(0, 5, 5, 5)
        timelapse_form = QFormLayout()
        self.timelapse_start_year_input = QSpinBox()
        self.timelapse_start_year_input.setRange(2000,
                                                 QDate.currentDate().year())
        self.timelapse_start_year_input.setValue(2021)
        timelapse_form.addRow("Time-Lapse Start Year:",
                              self.timelapse_start_year_input)
        self.timelapse_end_year_input = QSpinBox()
        self.timelapse_end_year_input.setRange(2000,
                                               QDate.currentDate().year() + 5)
        self.timelapse_end_year_input.setValue(QDate.currentDate().year())
        timelapse_form.addRow("Time-Lapse End Year:",
                              self.timelapse_end_year_input)
        self.timelapse_fps_input = QSpinBox()
        self.timelapse_fps_input.setRange(1, 30)
        self.timelapse_fps_input.setValue(1)
        timelapse_form.addRow("Video FPS:", self.timelapse_fps_input)
        left_v_layout.addLayout(timelapse_form)
        self.generate_timelapse_button = QPushButton(
            "Generate & Load Time-Lapse Video")
        left_v_layout.addWidget(self.generate_timelapse_button)
        left_v_layout.addStretch(1)
        timelapse_controls_panel.setMinimumWidth(220)
        timelapse_controls_panel.setMaximumWidth(300)

        # Video Display Panel (Right)
        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.video_widget.setMinimumHeight(360)  # Approx for 720p aspect

        video_display_panel = QWidget()
        right_v_layout = QVBoxLayout(video_display_panel)
        right_v_layout.setContentsMargins(0, 0, 0, 0)
        right_v_layout.addWidget(self.video_widget, 1)  # Video takes most space

        video_player_controls_layout = QHBoxLayout()
        self.play_button = QPushButton()
        self.play_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.setEnabled(False)
        video_player_controls_layout.addWidget(self.play_button)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setEnabled(False)
        video_player_controls_layout.addWidget(self.position_slider, 1)

        self.year_label_for_slider = QLabel("Year: -")
        video_player_controls_layout.addWidget(self.year_label_for_slider)
        right_v_layout.addLayout(video_player_controls_layout)

        main_timelapse_h_layout.addWidget(timelapse_controls_panel)
        main_timelapse_h_layout.addWidget(video_display_panel, 1)
        timelapse_video_group.setLayout(main_timelapse_h_layout)

        # --- Status Log Group ---
        status_group = QGroupBox("Status Log")
        status_v_layout = QVBoxLayout()
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMinimumHeight(80)
        self.status_log.setMaximumHeight(150)  # Limit height
        self.status_log.setSizePolicy(QSizePolicy.Policy.Preferred,
                                      QSizePolicy.Policy.Expanding)
        status_v_layout.addWidget(self.status_log)
        status_group.setLayout(status_v_layout)

        # --- Main Vertical Splitter ---
        main_v_splitter = QSplitter(Qt.Orientation.Vertical)
        main_v_splitter.addWidget(top_h_splitter)
        main_v_splitter.addWidget(timelapse_video_group)
        main_v_splitter.addWidget(status_group)
        main_v_splitter.setSizes([300, 470, 80])
        main_layout.addWidget(main_v_splitter)

    def _connect_signals(self):
        self.analyze_button.clicked.connect(self.run_single_analysis)
        self.generate_timelapse_button.clicked.connect(
            self.run_timelapse_generation)
        self.play_button.clicked.connect(self.play_video)
        self.position_slider.sliderMoved.connect(
            self.set_video_position_from_slider)
        self.media_player.playbackStateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.video_position_changed)
        self.media_player.durationChanged.connect(self.video_duration_changed)
        self.media_player.errorOccurred.connect(self.handle_media_player_error)

    def init_gee_check(self):
        self.log_status("Attempting GEE initialization...")
        try:
            ee.Initialize(project=self.project_id)
            self.log_status(
                f"GEE initialized successfully (Project: {self.project_id}).")
            # Perform a simple GEE op to confirm connectivity and permissions
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(
                1).size().getInfo()
            self.log_status("GEE test query successful.")
        except Exception as e:
            self.handle_gee_init_error(e)

    def handle_gee_init_error(self, e):
        msg = (f"GEE Initialization Error: {e}\n\n"
               "Please ensure:\n"
               "1. You have run 'earthengine authenticate' and authenticated.\n"
               "2. Active internet connection.\n"
               f"3. GEE Project ID ('{self.project_id}') is correct and API enabled.\n"
               "4. Necessary permissions for the GEE project.\n\n"
               "Restart application after verifying.")
        self.log_status(msg.replace("\n\n", "\n").replace("\n", "\nStatus: "))
        QMessageBox.critical(self, "GEE Initialization Error", msg)
        self.analyze_button.setEnabled(False)
        self.generate_timelapse_button.setEnabled(False)

    def log_status(self, message):
        self.status_log.append(message)
        QApplication.processEvents()  # Ensure GUI updates

    def setup_progress_dialog(self, title="Processing..."):
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.close()
        self.progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumDuration(0)  # Show immediately
        self.progress_dialog.setValue(0)
        return self.progress_dialog

    def _parse_coordinates(self):
        coords_text = self.coord_input.text().strip()
        coords_str_list = coords_text.split(',')
        if len(coords_str_list) != 4:
            raise ValueError(
                "Coordinate input requires 4 numbers (Lat1, Lon1, Lat2, Lon2) "
                "separated by commas.")
        try:
            raw_coords_float = [float(c.strip()) for c in coords_str_list]
        except ValueError:
            raise ValueError("Invalid number format in coordinates.")

        lat1, lon1, lat2, lon2 = raw_coords_float
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
            raise ValueError("Latitudes must be between -90 and 90.")
        if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            raise ValueError("Longitudes must be between -180 and 180.")

        # For ee.Geometry.Rectangle: [xmin, ymin, xmax, ymax]
        aoi_ee_rect_coords = [min(lon1, lon2), min(lat1, lat2),
                              max(lon1, lon2), max(lat1, lat2)]
        return raw_coords_float, aoi_ee_rect_coords

    def run_single_analysis(self):
        self.log_status("Starting single period analysis (interactive map)...")
        aoi_name = self.aoi_name_input.text().strip()
        if aoi_name:
            self.log_status(f"AOI Name: {aoi_name}")

        self.analyze_button.setEnabled(False)
        self.generate_timelapse_button.setEnabled(False)
        self.map_view.setHtml(
            "<html><body><h2>Processing analysis...</h2>"
            "<p>Map will load shortly.</p></body></html>")
        try:
            _, aoi_ee_rect_coords = self._parse_coordinates()
            start1 = self.date1_start.date().toString("yyyy-MM-dd")
            end1 = self.date1_end.date().toString("yyyy-MM-dd")
            start2 = self.date2_start.date().toString("yyyy-MM-dd")
            end2 = self.date2_end.date().toString("yyyy-MM-dd")
            if (self.date1_start.date() >= self.date1_end.date() or
                    self.date2_start.date() >= self.date2_end.date() or
                    self.date1_end.date() >= self.date2_start.date()):
                raise ValueError(
                    "Date ranges are invalid or overlapping. Ensure P1_End < "
                    "P2_Start, and P1_Start < P1_End, P2_Start < P2_End.")
        except ValueError as ve:
            self.log_status(f"Input Error: {ve}")
            QMessageBox.warning(self, "Input Error", str(ve))
            self.map_view.setHtml(
                f"<html><body><h2>Input Error:</h2><pre>{ve}</pre></body></html>")
            self.analyze_button.setEnabled(True)
            self.generate_timelapse_button.setEnabled(True)
            return

        pd = self.setup_progress_dialog("Single Analysis in Progress...")
        pd.setRange(0, 0)  # Indeterminate progress
        pd.canceled.connect(self.cancel_single_analysis)
        pd.show()

        self.worker = GEEWorker(aoi_ee_rect_coords, start1, end1, start2, end2,
                                self.threshold_input.value(),
                                project_id=self.project_id)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker.signals.finished.connect(self.on_single_analysis_complete_map)
        self.worker.signals.error.connect(self.on_single_analysis_error_map)
        self.worker.signals.progress.connect(self.update_progress_label_only)
        self.worker_thread.start()

    def update_progress_label_only(self, message):
        self.log_status(message)
        if self.progress_dialog and self.progress_dialog.isVisible():
            QMetaObject.invokeMethod(self.progress_dialog, "setLabelText",
                                     Qt.ConnectionType.QueuedConnection,
                                     Q_ARG(str, message))

    def on_single_analysis_complete_map(self, analysis_data):
        self.log_status("Single analysis (map data) received.")
        if self.progress_dialog:
            self.progress_dialog.close()

        map_id_dict = analysis_data.get('map_id_dict')
        aoi_bounds_gee = analysis_data.get('aoi_bounds')

        if not map_id_dict or not aoi_bounds_gee:
            self.log_status("Error: MapID or AOI bounds missing from GEE result.")
            self.map_view.setHtml(
                "<html><body><h2>Error:</h2>"
                "<p>Could not retrieve map data from GEE.</p></body></html>")
            self.analyze_button.setEnabled(True)
            self.generate_timelapse_button.setEnabled(True)
            return

        try:
            lons = [p[0] for p in aoi_bounds_gee]
            lats = [p[1] for p in aoi_bounds_gee]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2

            folium_map = folium.Map(location=[center_lat, center_lon],
                                    zoom_start=12, tiles=None)
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                attr='Google', name='Google Hybrid', overlay=False,
                control=True, show=True
            ).add_to(folium_map)
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                attr='Google', name='Google Roadmap', overlay=False,
                control=True
            ).add_to(folium_map)
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine Analysis',
                name='GEE Analysis Layer', overlay=True, control=True, show=True
            ).add_to(folium_map)
            folium.LayerControl().add_to(folium_map)

            if self.map_html_temp_dir:
                try:
                    self.map_html_temp_dir.cleanup()
                except Exception as e_clean:
                    self.log_status(
                        f"Note: Error cleaning up previous map temp dir: {e_clean}")

            self.map_html_temp_dir = tempfile.TemporaryDirectory(
                prefix="galamsey_map_")
            map_path = os.path.join(self.map_html_temp_dir.name, "map.html")

            folium_map.save(map_path)
            self.log_status(f"Interactive map saved to temp file: {map_path}")
            self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(map_path)))
            self.log_status("Interactive map loaded into view.")

        except Exception as e:
            self.log_status(
                f"Error creating/displaying Folium map: {e}\n{traceback.format_exc()}")
            self.map_view.setHtml(
                f"<html><body><h2>Map Display Error:</h2><pre>{e}</pre></body></html>")

        self.analyze_button.setEnabled(True)
        self.generate_timelapse_button.setEnabled(True)

    def on_single_analysis_error_map(self, error_message):
        self.log_status(f"Single Analysis (map) Error: {error_message}")
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Single Analysis Error", error_message)
        self.map_view.setHtml(
            f"<html><body><h2>Analysis Failed:</h2><pre>{error_message}</pre></body></html>")
        self.analyze_button.setEnabled(True)
        self.generate_timelapse_button.setEnabled(True)

    def cancel_single_analysis(self):
        self.log_status("Attempting to cancel single analysis...")
        if self.worker:
            self.worker.is_cancelled = True
        if self.progress_dialog:
            self.progress_dialog.close()
        self.map_view.setHtml(
            "<html><body style='display:flex;justify-content:center;"
            "align-items:center;height:100%;font-family:sans-serif;"
            "color:grey;'><p>Analysis Cancelled. Map will appear here.</p>"
            "</body></html>")
        self.analyze_button.setEnabled(True)
        self.generate_timelapse_button.setEnabled(True)

    def run_timelapse_generation(self):
        self.log_status("Starting 720p time-lapse video generation...")
        aoi_name_from_input = self.aoi_name_input.text().strip()
        if not aoi_name_from_input:
            QMessageBox.warning(self, "Input Error",
                                "AOI name is mandatory for video generation.")
            return

        temp_sanitized_aoi_name = "".join(
            c if c.isalnum() or c == '_' else '_' for c in aoi_name_from_input)
        final_sanitized_aoi_name = "_".join(
            s for s in temp_sanitized_aoi_name.split('_') if s).strip('_')

        if not final_sanitized_aoi_name:
            QMessageBox.warning(self, "Input Error",
                                f"AOI name '{aoi_name_from_input}' is invalid "
                                "for a filename. Please use alphanumeric characters.")
            return
        self.log_status(f"Using AOI Name for Timelapse: {final_sanitized_aoi_name}")

        self.analyze_button.setEnabled(False)
        self.generate_timelapse_button.setEnabled(False)
        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self._update_year_label_for_slider(0)  # Reset year label

        try:
            raw_coords, aoi_ee_rect = self._parse_coordinates()
            b_start = self.date1_start.date().toString("yyyy-MM-dd")
            b_end = self.date1_end.date().toString("yyyy-MM-dd")
            tl_start_y = self.timelapse_start_year_input.value()
            tl_end_y = self.timelapse_end_year_input.value()

            if self.date1_start.date() >= self.date1_end.date():
                raise ValueError("Baseline date range invalid.")
            if tl_start_y > tl_end_y:
                raise ValueError("Timelapse year range invalid.")
            if QDate(tl_start_y, 1, 1) <= self.date1_end.date():
                raise ValueError("Timelapse start year must be after baseline end.")

            coord_fn_str = '_'.join(
                [str(c).replace('.', 'p').replace('-', 'm') for c in raw_coords])
            vid_fname_base = (f"{final_sanitized_aoi_name}_{coord_fn_str}_"
                              f"{tl_start_y}-{tl_end_y}_720p_timelapse.mp4")

            videos_dir = os.path.join(os.getcwd(), "videos")
            os.makedirs(videos_dir, exist_ok=True)
            self.final_video_path = os.path.join(videos_dir, vid_fname_base)
            self.log_status(f"Video will be saved to: {self.final_video_path}")

        except ValueError as ve:
            self.log_status(f"Input Error (Timelapse Setup): {ve}")
            QMessageBox.warning(self, "Input Error", str(ve))
            self.analyze_button.setEnabled(True)
            self.generate_timelapse_button.setEnabled(True)
            return

        pd = self.setup_progress_dialog("Generating 720p Time-Lapse Video...")
        total_frames = tl_end_y - tl_start_y + 1
        pd.setRange(0, total_frames if total_frames > 0 else 1)
        pd.canceled.connect(self.cancel_timelapse_generation)
        pd.show()

        self.timelapse_worker = TimeLapseWorker(
            aoi_rectangle_coords=aoi_ee_rect,
            baseline_start_date=b_start, baseline_end_date=b_end,
            timelapse_start_year=tl_start_y, timelapse_end_year=tl_end_y,
            threshold=self.threshold_input.value(),
            output_dimensions=(1280, 720),  # 720p
            project_id=self.project_id,
            output_video_path=self.final_video_path,
            fps=self.timelapse_fps_input.value(),
            aoi_name=final_sanitized_aoi_name, raw_input_coords=raw_coords
        )
        self.timelapse_worker_thread = threading.Thread(
            target=self.timelapse_worker.run, daemon=True)
        self.timelapse_worker.signals.finished.connect(self.on_timelapse_complete)
        self.timelapse_worker.signals.error.connect(self.on_timelapse_error)
        self.timelapse_worker.signals.progress.connect(
            self.update_progress_label_only)
        self.timelapse_worker.signals.frame_processed.connect(
            self.update_timelapse_progress_value)
        self.timelapse_worker_thread.start()

    def update_timelapse_progress_value(self, current_frame, total_frames):
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.setRange(0, total_frames)
            self.progress_dialog.setValue(current_frame)

    def on_timelapse_complete(self, video_path):
        self.log_status(f"720p Time-Lapse video generated: {video_path}")
        if self.progress_dialog:
            self.progress_dialog.close()

        self.active_timelapse_start_year = self.timelapse_start_year_input.value()
        self.active_timelapse_end_year = self.timelapse_end_year_input.value()
        self.active_timelapse_fps = self.timelapse_fps_input.value()

        self.media_player.setSource(QUrl.fromLocalFile(video_path))
        self.play_button.setEnabled(True)
        self.position_slider.setEnabled(True)
        self._update_year_label_for_slider(0)  # Reset to start
        QMessageBox.information(self, "Time-Lapse Ready",
                                f"720p Time-Lapse video loaded.\nSaved: {video_path}")
        self.analyze_button.setEnabled(True)
        self.generate_timelapse_button.setEnabled(True)

    def on_timelapse_error(self, error_message):
        self.log_status(f"Time-Lapse Generation Error: {error_message}")
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Time-Lapse Error", error_message)
        self.analyze_button.setEnabled(True)
        self.generate_timelapse_button.setEnabled(True)
        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self._update_year_label_for_slider(0)

    def cancel_timelapse_generation(self):
        self.log_status("Cancelling time-lapse generation...")
        if self.timelapse_worker:
            self.timelapse_worker.is_cancelled = True
        if self.progress_dialog:
            self.progress_dialog.close()
        self.analyze_button.setEnabled(True)
        self.generate_timelapse_button.setEnabled(True)
        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self._update_year_label_for_slider(0)

    def play_video(self):
        if self.media_player.source().isEmpty():
            self.log_status("No video loaded to play.")
            return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.log_status("Video paused.")
        else:
            self.media_player.play()
            self.log_status("Video playing.")

    def media_state_changed(self, state: QMediaPlayer.PlaybackState):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

        # If stopped at the end, keep the last year displayed
        if (state == QMediaPlayer.PlaybackState.StoppedState and
                self.active_timelapse_start_year is not None and
                self.media_player.position() == self.media_player.duration() and
                self.media_player.duration() > 0):
            self._update_year_label_for_slider(self.media_player.duration())

    def _update_year_label_for_slider(self, position_ms):
        if (self.active_timelapse_start_year is not None and
                self.active_timelapse_end_year is not None and
                self.active_timelapse_fps is not None and
                self.active_timelapse_fps > 0):

            ms_per_frame = 1000.0 / self.active_timelapse_fps
            if ms_per_frame <= 0:
                self.year_label_for_slider.setText("Year: - (FPS Error)")
                return

            current_frame_float = position_ms / ms_per_frame
            num_total_frames = (self.active_timelapse_end_year -
                                self.active_timelapse_start_year + 1)

            current_frame_idx = int(current_frame_float)
            # Clamp index to be within [0, num_total_frames - 1]
            current_frame_idx = max(0, min(current_frame_idx,
                                           num_total_frames - 1))

            current_year = self.active_timelapse_start_year + current_frame_idx
            self.year_label_for_slider.setText(f"Year: {current_year}")
        else:
            self.year_label_for_slider.setText("Year: -")

    def video_position_changed(self, position_ms):  # milliseconds
        self.position_slider.setValue(position_ms)
        self._update_year_label_for_slider(position_ms)

    def video_duration_changed(self, duration_ms):  # milliseconds
        self.position_slider.setRange(0, duration_ms)
        if duration_ms == 0:  # Video unloaded or invalid
            self.active_timelapse_start_year = None
            self.active_timelapse_end_year = None
            self.active_timelapse_fps = None
            self._update_year_label_for_slider(0)
            self.play_button.setEnabled(False)
            self.position_slider.setEnabled(False)

    def set_video_position_from_slider(self, position_ms):
        self.media_player.setPosition(position_ms)
        self._update_year_label_for_slider(position_ms)  # Update immediately

    def handle_media_player_error(self):
        self.play_button.setEnabled(False)
        self.position_slider.setEnabled(False)
        error_string = self.media_player.errorString()
        self.log_status(f"Media Player Error: {error_string}")
        QMessageBox.critical(self, "Media Player Error",
                             "An error occurred with the media player: "
                             f"{error_string}")
        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self._update_year_label_for_slider(0)

    def closeEvent(self, event):
        self.log_status("Closing application...")
        if self.worker_thread and self.worker_thread.is_alive():
            self.log_status("Cancelling active single analysis worker...")
            self.cancel_single_analysis()
            self.worker_thread.join(timeout=1)  # Wait briefly
        if self.timelapse_worker_thread and self.timelapse_worker_thread.is_alive():
            self.log_status("Cancelling active time-lapse worker...")
            self.cancel_timelapse_generation()
            self.timelapse_worker_thread.join(timeout=1)  # Wait briefly

        self.media_player.stop()
        if self.map_html_temp_dir:
            try:
                self.map_html_temp_dir.cleanup()
                self.log_status("Cleaned up temporary map HTML directory.")
            except Exception as e:
                self.log_status(
                    f"Error cleaning up map HTML temp dir on close: {e}")
        self.log_status("Cleanup complete. Exiting.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("GalamseyMonitorApp")
    # For a more modern look, consider Fusion style:
    # app.setStyle("Fusion")
    monitor_app = GalamseyMonitorApp()
    monitor_app.show()
    sys.exit(app.exec())