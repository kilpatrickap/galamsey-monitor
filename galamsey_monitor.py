"""
Galamsey Monitor Application

This application allows users to monitor potential galamsey (illegal small-scale mining)
activities by analyzing changes in vegetation cover using Google Earth Engine (GEE)
Sentinel-2 satellite imagery. It provides:
1.  Single Period Analysis: Compares two user-defined periods and displays
    potential vegetation loss on an interactive map.
2.  Time-Lapse Video Generation: Creates a 720p time-lapse video showing
    vegetation loss overlay for a range of years.

Key Features:
-   Cloud masking for Sentinel-2 imagery.
-   NDVI (Normalized Difference Vegetation Index) calculation.
-   Interactive map preview using Folium.
-   Time-lapse video generation with text overlays (AOI name, coordinates, year).
-   GUI built with PyQt6.
-   Multithreading for GEE processing to keep the GUI responsive.
"""
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
from PyQt6.QtGui import QImage, QPixmap # QPixmap currently unused, consider removal if not planned
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


class WorkerSignals(QObject):
    """
    Defines signals available from a running worker thread.
    Supported signals are:
    - finished: Emitted when the worker has finished processing.
                Carries an object payload (result of processing).
    - error: Emitted when an error occurs. Carries a string payload (error message).
    - progress: Emitted to update progress. Carries a string payload (status message).
    - frame_processed: Emitted during time-lapse generation after each frame.
                       Carries (current_frame_number, total_frames).
    """
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    frame_processed = pyqtSignal(int, int)


def mask_s2_clouds_scl(image: ee.Image) -> ee.Image:
    """
    Masks clouds in a Sentinel-2 image using the Scene Classification Layer (SCL).

    Args:
        image: An ee.Image (Sentinel-2) with an 'SCL' band.

    Returns:
        An ee.Image with clouds and cloud shadows masked out.
    """
    scl = image.select('SCL')
    # Values to mask (clouds, shadows, etc.): 0, 1, 3, 8, 9, 10
    # These correspond to:
    # 0: NO_DATA (defective/saturated)
    # 1: DARK_AREA_PIXELS
    # 3: CLOUD_SHADOWS
    # 8: CLOUD_MEDIUM_PROBABILITY
    # 9: CLOUD_HIGH_PROBABILITY
    # 10: THIN_CIRRUS
    mask = scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(
        scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000).copyProperties(
        image, ["system:time_start"])


def process_single_gee_frame(aoi_rectangle_coords: list,
                             period1_start: str, period1_end: str,
                             period2_start: str, period2_end: str,
                             threshold_val: float, dimensions_val,
                             project_id: str, progress_emitter=None,
                             output_type: str = 'image') -> Image.Image | dict | None:
    """
    Processes GEE data for a single frame or map ID analysis.

    Calculates NDVI change between two periods and identifies potential
    vegetation loss based on a threshold.

    Args:
        aoi_rectangle_coords: Coordinates for the Area of Interest [xmin, ymin, xmax, ymax].
        period1_start: Start date for baseline period (YYYY-MM-DD).
        period1_end: End date for baseline period (YYYY-MM-DD).
        period2_start: Start date for comparison period (YYYY-MM-DD).
        period2_end: End date for comparison period (YYYY-MM-DD).
        threshold_val: NDVI change threshold for detecting loss.
        dimensions_val: Dimensions for the output thumbnail (int for square,
                        tuple (width, height), or string "WIDTHxHEIGHT").
        project_id: Google Earth Engine project ID.
        progress_emitter: Optional callable to emit progress messages.
        output_type: 'image' for PIL Image, 'map_id' for GEE MapID dictionary.

    Returns:
        A PIL Image if output_type is 'image', a dictionary containing
        MapID and AOI bounds if output_type is 'map_id', or None on failure.
    """
    aoi = ee.Geometry.Rectangle(aoi_rectangle_coords)

    def calculate_ndvi(image: ee.Image) -> ee.Image:
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
    if collection_p1_base.size().getInfo() == 0:
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
    if collection_p2_base.size().getInfo() == 0:
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
                            "Frame: Error getting background thumbnail: "
                            f"{bg_thumb_e}")
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
            map_id_dict_result = ee.data.getMapId({
                'image': final_image_viz,
                'region': aoi.getInfo()['coordinates']
            })
            bounds_coords = aoi.bounds(maxError=1).getInfo()['coordinates'][0]
            return {'map_id_dict': map_id_dict_result, 'aoi_bounds': bounds_coords}
        except Exception as e:
            if progress_emitter:
                progress_emitter(f"Error getting MapID: {e}")
            raise
    else:
        raise ValueError(
            f"Invalid output_type specified for GEE processing: {output_type}")


class GEEWorker(QObject):
    """
    Worker thread for performing a single GEE analysis to generate MapID data.
    """
    def __init__(self, aoi_rectangle_coords: list, start1: str, end1: str,
                 start2: str, end2: str, threshold: float,
                 project_id: str = 'galamsey-monitor'):
        super().__init__()
        self.signals = WorkerSignals()
        self.aoi_rectangle_coords = aoi_rectangle_coords
        self.start1, self.end1 = start1, end1
        self.start2, self.end2 = start2, end2
        self.threshold = threshold
        self.project_id = project_id
        self.is_cancelled = False

    def run(self):
        """Executes the GEE processing task."""
        try:
            ee.Initialize(project=self.project_id)
        except Exception:
            pass # GEE might be already initialized.

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
                    "Failed to generate map data from GEE.")
        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(
                f"Error in GEEWorker (map_id): {e}\nTrace: {tb_str}")


class TimeLapseWorker(QObject):
    """
    Worker thread for generating a time-lapse video with text overlays.
    """
    def __init__(self, aoi_rectangle_coords: list, baseline_start_date: str,
                 baseline_end_date: str, timelapse_start_year: int,
                 timelapse_end_year: int, threshold: float,
                 output_dimensions: tuple = (1280, 720),
                 project_id: str = 'galamsey-monitor',
                 output_video_path: str = "galamsey_timelapse.mp4", fps: int = 1,
                 aoi_name: str = "Default AOI", raw_input_coords: list = None):
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

    def _get_font(self, path: str, size: int,
                  default_message_emitter) -> ImageFont.FreeTypeFont:
        """Safely loads a font, falling back to default if not found."""
        try:
            return ImageFont.truetype(path, size)
        except IOError:
            if default_message_emitter:
                default_message_emitter(
                    f"Warning: Font '{path}' not found. Using default PIL font.")
            return ImageFont.load_default()

    def _get_text_size(self, draw_instance: ImageDraw.ImageDraw, text: str,
                       font_instance: ImageFont.FreeTypeFont) -> tuple[int, int]:
        """Gets text dimensions, compatible with older/newer PIL versions."""
        try:
            bbox = draw_instance.textbbox((0, 0), text, font=font_instance)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]  # width, height
        except AttributeError:
            return draw_instance.textsize(text, font=font_instance)

    def _draw_text_with_background(self, draw: ImageDraw.ImageDraw,
                                   text_content: str,
                                   font: ImageFont.FreeTypeFont,
                                   position: tuple[int, int],
                                   text_color: tuple, bg_color: tuple,
                                   padding: int) -> tuple[int, int, int, int]:
        """Draws text with a semi-transparent background."""
        text_w, text_h = self._get_text_size(draw, text_content, font)
        bg_x0, bg_y0 = position
        bg_x1 = bg_x0 + text_w + 2 * padding
        bg_y1 = bg_y0 + text_h + 2 * padding
        draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=bg_color)
        draw.text((bg_x0 + padding, bg_y0 + padding), text_content,
                  font=font, fill=text_color)
        return bg_x0, bg_y0, bg_x1, bg_y1 # Returns background bounding box

    def run(self):
        """Generates the time-lapse video frames and compiles them."""
        try:
            ee.Initialize(project=self.project_id)
        except Exception:
            pass

        video_w, video_h = self.output_dimensions
        video_frame_size = (video_w, video_h)
        self.signals.progress.emit(
            f"Video frame size set to: {video_w}x{video_h}")

        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0, 150)  # Black, semi-transparent
        padding = 8
        font_path = "arial.ttf" # Consider making this configurable or bundling

        font_title = self._get_font(font_path, 28, self.signals.progress.emit)
        font_hdr = self._get_font(font_path, 18, self.signals.progress.emit)
        font_val = self._get_font(font_path, 16, self.signals.progress.emit)
        font_info = self._get_font(font_path, 14, self.signals.progress.emit)
        font_year = self._get_font(font_path, 22, self.signals.progress.emit)

        video_writer = None
        total_frames_count = self.timelapse_end_year - self.timelapse_start_year + 1
        if total_frames_count <= 0:
            self.signals.error.emit("No frames requested.")
            return

        try:
            for i, year_val in enumerate(
                    range(self.timelapse_start_year, self.timelapse_end_year + 1)):
                if self.is_cancelled:
                    self.signals.progress.emit("Time-lapse cancelled.")
                    break
                self.signals.frame_processed.emit(i + 1, total_frames_count)
                self.signals.progress.emit(
                    f"Processing GEE data for {year_val} ({i + 1}/{total_frames_count})")

                gee_img = process_single_gee_frame(
                    self.aoi_rectangle_coords, self.baseline_start_date,
                    self.baseline_end_date, f"{year_val}-01-01", f"{year_val}-12-31",
                    self.threshold, self.output_dimensions, self.project_id,
                    self.signals.progress.emit, output_type='image'
                )

                frame_rgba: Image.Image
                if gee_img:
                    if gee_img.size != video_frame_size:
                        self.signals.progress.emit(
                            f"Resizing GEE image from {gee_img.size} to {video_frame_size}")
                        gee_img = gee_img.resize(video_frame_size, Image.Resampling.LANCZOS)
                    frame_rgba = gee_img.convert('RGBA') if gee_img.mode != 'RGBA' else gee_img.copy()
                else:
                    frame_rgba = Image.new('RGBA', video_frame_size, (0,0,0,255))
                    draw_temp = ImageDraw.Draw(frame_rgba)
                    no_data_msg = f"NO DATA FOR {year_val}"
                    txt_w, txt_h = self._get_text_size(draw_temp, no_data_msg, font_title)
                    draw_temp.text(((video_w - txt_w) / 2, (video_h - txt_h) / 2),
                                   no_data_msg, font=font_title, fill=text_color)

                draw_on_frame = ImageDraw.Draw(frame_rgba)

                # Coordinates (Top-Left)
                coord_items = [
                    ("Top-left-corner:", font_hdr, 3),
                    (f"  Lat 1: {self.raw_input_coords[0]:.6f}", font_val, 3),
                    (f"  Lon 1: {self.raw_input_coords[1]:.6f}", font_val, 8),
                    ("Bottom-right-corner:", font_hdr, 3),
                    (f"  Lat 2: {self.raw_input_coords[2]:.6f}", font_val, 3),
                    (f"  Lon 2: {self.raw_input_coords[3]:.6f}", font_val, 0),
                ]
                max_w_coord = 0
                total_h_coord_block = 0
                line_hs_coord = [self._get_text_size(draw_on_frame, "Ay", f[1])[1] for f in coord_items]
                for idx, (txt, fnt, _) in enumerate(coord_items):
                    max_w_coord = max(max_w_coord, self._get_text_size(draw_on_frame, txt, fnt)[0])
                    total_h_coord_block += line_hs_coord[idx] + coord_items[idx][2]
                if coord_items: total_h_coord_block -= coord_items[-1][2]

                coord_bg_x0, coord_bg_y0 = padding, padding
                coord_bg_w = max_w_coord + 2 * padding
                coord_bg_h = total_h_coord_block + 2 * padding
                draw_on_frame.rectangle(
                    [coord_bg_x0, coord_bg_y0, coord_bg_x0 + coord_bg_w, coord_bg_y0 + coord_bg_h],
                    fill=bg_color)
                curr_y_coord = coord_bg_y0 + padding
                for idx, (txt, fnt, space) in enumerate(coord_items):
                    draw_on_frame.text((coord_bg_x0 + padding, curr_y_coord), txt, font=fnt, fill=text_color)
                    curr_y_coord += line_hs_coord[idx] + space
                coord_bg_x1 = coord_bg_x0 + coord_bg_w

                # AOI Name (Right of Coordinates, or centered if space)
                aoi_txt = self.aoi_name
                aoi_w, _ = self._get_text_size(draw_on_frame, aoi_txt, font_title)
                aoi_bg_w_total = aoi_w + 2 * padding
                rem_w = video_w - coord_bg_x1 - (2 * padding)
                aoi_x = coord_bg_x1 + padding + (rem_w - aoi_bg_w_total) / 2 \
                    if aoi_bg_w_total < rem_w else coord_bg_x1 + padding
                self._draw_text_with_background(
                    draw_on_frame, aoi_txt, font_title, (aoi_x, padding),
                    text_color, bg_color, padding)

                # Year (Bottom-Right)
                year_txt_str = f"YEAR: {year_val}"
                yr_w, yr_h = self._get_text_size(draw_on_frame, year_txt_str, font_year)
                yr_x = video_w - (yr_w + 2 * padding) - padding
                yr_y = video_h - (yr_h + 2 * padding) - padding
                self._draw_text_with_background(
                    draw_on_frame, year_txt_str, font_year, (yr_x, yr_y),
                    text_color, bg_color, padding)

                # Bottom-Left Info
                bl_info_items = [
                    ("Red shows potential Vegetative Loss", font_info, 3),
                    ("Copyright KilTech Ent 2025", font_info, 3),
                    ("All rights reserved.", font_info, 0)
                ]
                max_w_bl = 0
                total_h_bl_block = 0
                line_hs_bl = [self._get_text_size(draw_on_frame, "Ay", f[1])[1] for f in bl_info_items]
                for idx, (txt, fnt, _) in enumerate(bl_info_items):
                    max_w_bl = max(max_w_bl, self._get_text_size(draw_on_frame, txt, fnt)[0])
                    total_h_bl_block += line_hs_bl[idx] + bl_info_items[idx][2]
                if bl_info_items: total_h_bl_block -= bl_info_items[-1][2]

                bl_bg_x0 = padding
                bl_bg_h = total_h_bl_block + 2 * padding
                bl_bg_y0 = video_h - bl_bg_h - padding
                bl_bg_w = max_w_bl + 2 * padding
                draw_on_frame.rectangle(
                    [bl_bg_x0, bl_bg_y0, bl_bg_x0 + bl_bg_w, bl_bg_y0 + bl_bg_h],
                    fill=bg_color)
                curr_y_bl = bl_bg_y0 + padding
                for idx, (txt, fnt, space) in enumerate(bl_info_items):
                    draw_on_frame.text((bl_bg_x0 + padding, curr_y_bl), txt, font=fnt, fill=text_color)
                    curr_y_bl += line_hs_bl[idx] + space

                final_rgb = frame_rgba.convert('RGB')
                frame_for_cv = cv2.cvtColor(np.array(final_rgb), cv2.COLOR_RGB2BGR)

                if not video_writer:
                    self.signals.progress.emit("Initializing video writer...")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_dir = os.path.dirname(self.output_video_path)
                    if vid_dir and not os.path.exists(vid_dir):
                        os.makedirs(vid_dir, exist_ok=True)
                    video_writer = cv2.VideoWriter(
                        self.output_video_path, fourcc, self.fps, video_frame_size)
                    if not video_writer.isOpened():
                        self.signals.error.emit(f"Could not open video writer for {self.output_video_path}.")
                        return
                video_writer.write(frame_for_cv)
                self.signals.progress.emit(f"Added frame for {year_val} to video.")

            if video_writer:
                self.signals.progress.emit("Finalizing video...")
                video_writer.release()
                self.signals.progress.emit(f"Video compilation complete: {self.output_video_path}")
                self.signals.finished.emit(self.output_video_path)
            elif total_frames_count > 0 : # Only error if frames were expected
                self.signals.error.emit("No frames processed to create video.")

        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(f"Time-lapse error: {e}\nTrace: {tb_str}")
            if video_writer:
                video_writer.release()


class GalamseyMonitorApp(QWidget):
    """
    Main application window for the Galamsey Monitor.
    Handles UI setup, user interactions, and orchestrates GEE processing.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galamsey Monitor")
        # Consider making window size adaptive or configurable
        self.setGeometry(10, 40, 1400, 850)

        self.worker_thread = None
        self.gee_worker = None # Renamed from self.worker for clarity
        self.timelapse_worker_thread = None
        self.timelapse_worker = None
        self.progress_dialog = None
        self.project_id = 'galamsey-monitor' # Make configurable if needed
        self.map_html_temp_dir = None
        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self.final_video_path = None # Stores path of last generated video

        self._setup_ui()
        self._connect_signals()
        self.init_gee_check()

    def _setup_ui(self):
        """Initializes and lays out the UI components."""
        main_layout = QVBoxLayout(self)

        # Single Period Analysis Group
        single_analysis_gb = QGroupBox("Single Period Analysis")
        single_form = QFormLayout()
        self.aoi_name_input = QLineEdit("Anyinam") # Default example
        self.aoi_name_input.setToolTip("Descriptive name for AOI (for Video).")
        single_form.addRow("AOI Name:", self.aoi_name_input)
        self.coord_input = QLineEdit("6.401452,-0.594587,6.355603,-0.496084")
        self.coord_input.setToolTip("Lat1,Lon1,Lat2,Lon2 (e.g. 6.3,-1.8,6.4,-1.7)")
        single_form.addRow("AOI Coords:", self.coord_input)
        self.date1_start = QDateEdit(QDate(2020, 1, 1))
        self.date1_end = QDateEdit(QDate(2020, 12, 31))
        self.date2_start = QDateEdit(QDate(2025, 1, 1))
        self.date2_end = QDateEdit(QDate.currentDate())
        for dt_edit in [self.date1_start, self.date1_end, self.date2_start, self.date2_end]:
            dt_edit.setCalendarPopup(True)
            dt_edit.setDisplayFormat("dd-MM-yyyy")
        single_form.addRow("Baseline Start:", self.date1_start)
        single_form.addRow("Baseline End:", self.date1_end)
        single_form.addRow("Comparison Start:", self.date2_start)
        single_form.addRow("Comparison End:", self.date2_end)
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(-1.0, 0.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setValue(-0.20)
        single_form.addRow("NDVI Change Threshold:", self.threshold_input)
        self.analyze_button = QPushButton("Analyze Single Period")
        single_form.addRow(self.analyze_button)
        single_analysis_gb.setLayout(single_form)
        single_analysis_gb.setMinimumWidth(450)

        # Interactive Map Group
        map_gb = QGroupBox("Interactive Map Preview")
        map_layout = QVBoxLayout()
        self.map_view = QWebEngineView()
        # Enable necessary WebEngine settings
        ws = self.map_view.settings()
        ws.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        ws.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        ws.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.map_view.setHtml(
            "<body style='display:flex;justify-content:center;align-items:center;"
            "height:100%;font-family:sans-serif;color:grey;'>"
            "<p>Map will appear here after analysis.</p></body>"
        )
        self.map_view.setMinimumSize(400, 300)
        map_layout.addWidget(self.map_view)
        map_gb.setLayout(map_layout)

        # Top Horizontal Splitter
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(single_analysis_gb)
        top_splitter.addWidget(map_gb)
        top_splitter.setSizes([450, 950]) # Give more space to map initially

        # Time-Lapse Video Group
        timelapse_gb = QGroupBox("Time-Lapse Video (720p)")
        timelapse_main_layout = QHBoxLayout()

        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 5, 5, 5)
        timelapse_form_layout = QFormLayout()
        self.tl_start_year_input = QSpinBox()
        self.tl_start_year_input.setRange(2000, QDate.currentDate().year())
        self.tl_start_year_input.setValue(QDate.currentDate().year() - 3) # Default 3 years ago
        timelapse_form_layout.addRow("Start Year:", self.tl_start_year_input)
        self.tl_end_year_input = QSpinBox()
        self.tl_end_year_input.setRange(2000, QDate.currentDate().year() + 5)
        self.tl_end_year_input.setValue(QDate.currentDate().year())
        timelapse_form_layout.addRow("End Year:", self.tl_end_year_input)
        self.tl_fps_input = QSpinBox()
        self.tl_fps_input.setRange(1, 30)
        self.tl_fps_input.setValue(1) # Default 1 FPS
        timelapse_form_layout.addRow("Video FPS:", self.tl_fps_input)
        controls_layout.addLayout(timelapse_form_layout)
        self.generate_tl_button = QPushButton("Generate & Load Time-Lapse")
        controls_layout.addWidget(self.generate_tl_button)
        controls_layout.addStretch(1)
        controls_panel.setMinimumWidth(220)
        controls_panel.setMaximumWidth(300)

        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setMinimumHeight(360) # Min height for 16:9 aspect

        video_display_widget = QWidget()
        video_display_layout = QVBoxLayout(video_display_widget)
        video_display_layout.setContentsMargins(0,0,0,0)
        video_display_layout.addWidget(self.video_widget, 1)

        player_controls_layout = QHBoxLayout()
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.setEnabled(False)
        player_controls_layout.addWidget(self.play_button)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setEnabled(False)
        player_controls_layout.addWidget(self.position_slider, 1)
        self.year_label_for_slider = QLabel("Year: -")
        player_controls_layout.addWidget(self.year_label_for_slider)
        video_display_layout.addLayout(player_controls_layout)

        timelapse_main_layout.addWidget(controls_panel)
        timelapse_main_layout.addWidget(video_display_widget, 1)
        timelapse_gb.setLayout(timelapse_main_layout)

        # Status Log Group
        status_gb = QGroupBox("Status Log")
        status_layout = QVBoxLayout()
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMinimumHeight(80)
        self.status_log.setMaximumHeight(150)
        status_layout.addWidget(self.status_log)
        status_gb.setLayout(status_layout)

        # Main Vertical Splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(timelapse_gb)
        main_splitter.addWidget(status_gb)
        main_splitter.setSizes([350, 420, 80]) # Adjust initial proportions
        main_layout.addWidget(main_splitter)

    def _connect_signals(self):
        """Connects UI element signals to their respective slots."""
        self.analyze_button.clicked.connect(self.run_single_analysis)
        self.generate_tl_button.clicked.connect(self.run_timelapse_generation)
        self.play_button.clicked.connect(self.play_video)
        self.position_slider.sliderMoved.connect(self.set_video_position_from_slider)
        self.media_player.playbackStateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.video_position_changed)
        self.media_player.durationChanged.connect(self.video_duration_changed)
        self.media_player.errorOccurred.connect(self.handle_media_player_error)

    def init_gee_check(self):
        """Initializes GEE and performs a basic test query."""
        self.log_status("Attempting GEE initialization...")
        try:
            ee.Initialize(project=self.project_id)
            self.log_status(f"GEE initialized (Project: {self.project_id}).")
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1).size().getInfo()
            self.log_status("GEE test query successful.")
        except Exception as e:
            self.handle_gee_init_error(e)

    def handle_gee_init_error(self, e: Exception):
        """Handles GEE initialization errors by informing the user."""
        msg = (f"GEE Initialization Error: {e}\n\n"
               "Please ensure:\n"
               "1. 'earthengine authenticate' completed successfully.\n"
               "2. Active internet connection.\n"
               f"3. GEE Project ID ('{self.project_id}') is correct & API enabled.\n"
               "4. Necessary permissions for the GEE project.\n\n"
               "Restart application after verifying.")
        self.log_status(msg.replace("\n\n", "\n").replace("\n", "\nStatus: "))
        QMessageBox.critical(self, "GEE Initialization Error", msg)
        self.analyze_button.setEnabled(False)
        self.generate_tl_button.setEnabled(False)

    def log_status(self, message: str):
        """Appends a message to the status log and processes GUI events."""
        self.status_log.append(message)
        QApplication.processEvents()

    def setup_progress_dialog(self, title: str = "Processing...") -> QProgressDialog:
        """Creates and configures a progress dialog."""
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.close()
        self.progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        return self.progress_dialog

    def _parse_coordinates(self) -> tuple[list[float], list[float]]:
        """Parses and validates coordinates from the input field."""
        coords_text = self.coord_input.text().strip()
        coords_list_str = coords_text.split(',')
        if len(coords_list_str) != 4:
            raise ValueError("Need 4 coords (Lat1,Lon1,Lat2,Lon2).")
        try:
            raw_coords = [float(c.strip()) for c in coords_list_str]
        except ValueError:
            raise ValueError("Invalid number format in coordinates.")
        lat1, lon1, lat2, lon2 = raw_coords
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
            raise ValueError("Latitudes must be -90 to 90.")
        if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            raise ValueError("Longitudes must be -180 to 180.")
        aoi_ee_rect = [min(lon1, lon2), min(lat1, lat2),
                       max(lon1, lon2), max(lat1, lat2)]
        return raw_coords, aoi_ee_rect

    def _set_buttons_enabled(self, enabled: bool):
        """Enables or disables main action buttons."""
        self.analyze_button.setEnabled(enabled)
        self.generate_tl_button.setEnabled(enabled)

    def run_single_analysis(self):
        """Initiates a single period analysis for the interactive map."""
        self.log_status("Starting single period analysis...")
        aoi_name = self.aoi_name_input.text().strip()
        if aoi_name: self.log_status(f"AOI Name: {aoi_name}")

        self._set_buttons_enabled(False)
        self.map_view.setHtml("<body><h2>Processing analysis...</h2></body>")
        try:
            _, aoi_ee_rect = self._parse_coordinates()
            p1_start = self.date1_start.date().toString("yyyy-MM-dd")
            p1_end = self.date1_end.date().toString("yyyy-MM-dd")
            p2_start = self.date2_start.date().toString("yyyy-MM-dd")
            p2_end = self.date2_end.date().toString("yyyy-MM-dd")
            if not (p1_start < p1_end and p2_start < p2_end and p1_end < p2_start):
                raise ValueError("Date ranges invalid or overlapping.")
        except ValueError as ve:
            self.log_status(f"Input Error: {ve}")
            QMessageBox.warning(self, "Input Error", str(ve))
            self.map_view.setHtml(f"<body>Error: {ve}</body>")
            self._set_buttons_enabled(True)
            return

        pd = self.setup_progress_dialog("Single Analysis in Progress...")
        pd.setRange(0, 0) # Indeterminate
        pd.canceled.connect(self.cancel_single_analysis)
        pd.show()

        self.gee_worker = GEEWorker(aoi_ee_rect, p1_start, p1_end, p2_start, p2_end,
                                   self.threshold_input.value(), self.project_id)
        self.worker_thread = threading.Thread(target=self.gee_worker.run, daemon=True)
        self.gee_worker.signals.finished.connect(self.on_single_analysis_complete_map)
        self.gee_worker.signals.error.connect(self.on_single_analysis_error_map)
        self.gee_worker.signals.progress.connect(self.update_progress_label_only)
        self.worker_thread.start()

    def update_progress_label_only(self, message: str):
        """Updates the progress dialog's label text."""
        self.log_status(message)
        if self.progress_dialog and self.progress_dialog.isVisible():
            QMetaObject.invokeMethod(self.progress_dialog, "setLabelText",
                                     Qt.ConnectionType.QueuedConnection,
                                     Q_ARG(str, message))

    def on_single_analysis_complete_map(self, analysis_data: dict):
        """Handles completion of single analysis, displaying the map."""
        self.log_status("Single analysis (map data) received.")
        if self.progress_dialog: self.progress_dialog.close()

        map_id_dict = analysis_data.get('map_id_dict')
        aoi_bounds = analysis_data.get('aoi_bounds')

        if not map_id_dict or not aoi_bounds:
            self.log_status("Error: MapID or AOI bounds missing.")
            self.map_view.setHtml("<body>Error: No map data from GEE.</body>")
            self._set_buttons_enabled(True)
            return

        try:
            lons = [p[0] for p in aoi_bounds]
            lats = [p[1] for p in aoi_bounds]
            map_center = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]

            f_map = folium.Map(location=map_center, zoom_start=12, tiles=None)
            folium.TileLayer(
                'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', # Hybrid
                attr='Google', name='Google Hybrid', overlay=False, control=True, show=True
            ).add_to(f_map)
            folium.TileLayer(
                'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', # Roadmap
                attr='Google', name='Google Roadmap', overlay=False, control=True
            ).add_to(f_map)
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='GEE Analysis', name='GEE Analysis Layer',
                overlay=True, control=True, show=True
            ).add_to(f_map)
            folium.LayerControl().add_to(f_map)

            if self.map_html_temp_dir:
                try: self.map_html_temp_dir.cleanup()
                except Exception as e_clean:
                    self.log_status(f"Note: Error cleaning temp map dir: {e_clean}")
            self.map_html_temp_dir = tempfile.TemporaryDirectory(prefix="galamsey_map_")
            map_file_path = os.path.join(self.map_html_temp_dir.name, "interactive_map.html")
            f_map.save(map_file_path)
            self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(map_file_path)))
            self.log_status("Interactive map loaded.")
        except Exception as e:
            self.log_status(f"Folium map error: {e}\n{traceback.format_exc()}")
            self.map_view.setHtml(f"<body>Map Display Error: {e}</body>")
        self._set_buttons_enabled(True)

    def on_single_analysis_error_map(self, error_message: str):
        """Handles errors from the single analysis worker."""
        self.log_status(f"Single Analysis Error: {error_message}")
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.critical(self, "Single Analysis Error", error_message)
        self.map_view.setHtml(f"<body>Analysis Failed: {error_message}</body>")
        self._set_buttons_enabled(True)

    def cancel_single_analysis(self):
        """Cancels the ongoing single analysis GEE task."""
        self.log_status("Cancelling single analysis...")
        if self.gee_worker: self.gee_worker.is_cancelled = True
        if self.progress_dialog: self.progress_dialog.close()
        self.map_view.setHtml("<body>Analysis Cancelled.</body>")
        self._set_buttons_enabled(True)

    def run_timelapse_generation(self):
        """Initiates the time-lapse video generation process."""
        self.log_status("Starting 720p time-lapse video generation...")
        aoi_name = self.aoi_name_input.text().strip()
        if not aoi_name:
            QMessageBox.warning(self, "Input Error", "AOI name is mandatory for video.")
            return

        sane_aoi_name = "".join(c if c.isalnum() or c == '_' else '_' for c in aoi_name)
        sane_aoi_name = "_".join(s for s in sane_aoi_name.split('_') if s).strip('_')
        if not sane_aoi_name:
            QMessageBox.warning(self, "Input Error", f"AOI name '{aoi_name}' invalid for filename.")
            return
        self.log_status(f"Using AOI Name for Timelapse: {sane_aoi_name}")

        self._set_buttons_enabled(False)
        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self._update_year_label_for_slider(0)

        try:
            raw_coords, aoi_ee_rect = self._parse_coordinates()
            b_start = self.date1_start.date().toString("yyyy-MM-dd")
            b_end = self.date1_end.date().toString("yyyy-MM-dd")
            tl_start = self.tl_start_year_input.value()
            tl_end = self.tl_end_year_input.value()

            if not (b_start < b_end): raise ValueError("Baseline date range invalid.")
            if not (tl_start <= tl_end): raise ValueError("Timelapse year range invalid.")
            if QDate(tl_start, 1, 1) <= self.date1_end.date():
                raise ValueError("Timelapse start year must be after baseline period end.")

            coord_fname_part = '_'.join([str(c).replace('.','p').replace('-','m') for c in raw_coords])
            video_filename = f"{sane_aoi_name}_{coord_fname_part}_{tl_start}-{tl_end}_720p.mp4"
            videos_path = os.path.join(os.getcwd(), "videos")
            os.makedirs(videos_path, exist_ok=True)
            self.final_video_path = os.path.join(videos_path, video_filename)
            self.log_status(f"Video will be saved to: {self.final_video_path}")
        except ValueError as ve:
            self.log_status(f"Input Error (Timelapse): {ve}")
            QMessageBox.warning(self, "Input Error", str(ve))
            self._set_buttons_enabled(True)
            return

        pd = self.setup_progress_dialog("Generating 720p Time-Lapse Video...")
        num_frames = tl_end - tl_start + 1
        pd.setRange(0, num_frames if num_frames > 0 else 1)
        pd.canceled.connect(self.cancel_timelapse_generation)
        pd.show()

        self.timelapse_worker = TimeLapseWorker(
            aoi_rectangle_coords=aoi_ee_rect, baseline_start_date=b_start,
            baseline_end_date=b_end, timelapse_start_year=tl_start,
            timelapse_end_year=tl_end, threshold=self.threshold_input.value(),
            output_dimensions=(1280, 720), project_id=self.project_id,
            output_video_path=self.final_video_path, fps=self.tl_fps_input.value(),
            aoi_name=sane_aoi_name, raw_input_coords=raw_coords
        )
        self.timelapse_worker_thread = threading.Thread(target=self.timelapse_worker.run, daemon=True)
        self.timelapse_worker.signals.finished.connect(self.on_timelapse_complete)
        self.timelapse_worker.signals.error.connect(self.on_timelapse_error)
        self.timelapse_worker.signals.progress.connect(self.update_progress_label_only)
        self.timelapse_worker.signals.frame_processed.connect(self.update_timelapse_progress_value)
        self.timelapse_worker_thread.start()

    def update_timelapse_progress_value(self, current_frame: int, total_frames: int):
        """Updates the progress dialog's value for timelapse generation."""
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.setRange(0, total_frames)
            self.progress_dialog.setValue(current_frame)

    def on_timelapse_complete(self, video_path: str):
        """Handles successful completion of time-lapse video generation."""
        self.log_status(f"720p Time-Lapse video generated: {video_path}")
        if self.progress_dialog: self.progress_dialog.close()
        self.active_timelapse_start_year = self.tl_start_year_input.value()
        self.active_timelapse_end_year = self.tl_end_year_input.value()
        self.active_timelapse_fps = self.tl_fps_input.value()
        self.media_player.setSource(QUrl.fromLocalFile(video_path))
        self.play_button.setEnabled(True)
        self.position_slider.setEnabled(True)
        self._update_year_label_for_slider(0)
        QMessageBox.information(self, "Time-Lapse Ready",
                                f"720p Time-Lapse video loaded.\nSaved: {video_path}")
        self._set_buttons_enabled(True)

    def on_timelapse_error(self, error_message: str):
        """Handles errors from the time-lapse generation worker."""
        self.log_status(f"Time-Lapse Error: {error_message}")
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.critical(self, "Time-Lapse Error", error_message)
        self._set_buttons_enabled(True)
        self.active_timelapse_start_year = None # Reset on error
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self._update_year_label_for_slider(0)

    def cancel_timelapse_generation(self):
        """Cancels the ongoing time-lapse generation task."""
        self.log_status("Cancelling time-lapse generation...")
        if self.timelapse_worker: self.timelapse_worker.is_cancelled = True
        if self.progress_dialog: self.progress_dialog.close()
        self._set_buttons_enabled(True)
        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self._update_year_label_for_slider(0)

    def play_video(self):
        """Plays or pauses the currently loaded video."""
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
        """Updates UI based on media player state changes."""
        is_playing = (state == QMediaPlayer.PlaybackState.PlayingState)
        icon = QStyle.StandardPixmap.SP_MediaPause if is_playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_button.setIcon(self.style().standardIcon(icon))

        # If stopped at the end, keep the last year displayed on slider
        is_stopped_at_end = (
            state == QMediaPlayer.PlaybackState.StoppedState and
            self.active_timelapse_start_year is not None and
            self.media_player.position() == self.media_player.duration() and
            self.media_player.duration() > 0
        )
        if is_stopped_at_end:
            self._update_year_label_for_slider(self.media_player.duration())

    def _update_year_label_for_slider(self, position_ms: int):
        """Updates the year label based on video position and FPS."""
        if not all([self.active_timelapse_start_year,
                    self.active_timelapse_end_year,
                    self.active_timelapse_fps]) or self.active_timelapse_fps <= 0:
            self.year_label_for_slider.setText("Year: -")
            return

        ms_per_frame = 1000.0 / self.active_timelapse_fps
        if ms_per_frame <= 0: # Should be caught by fps > 0 check, but defensive
            self.year_label_for_slider.setText("Year: - (FPS Error)")
            return

        frame_float = position_ms / ms_per_frame
        total_video_frames = (self.active_timelapse_end_year -
                              self.active_timelapse_start_year + 1)
        frame_idx = max(0, min(int(frame_float), total_video_frames - 1))
        current_year_display = self.active_timelapse_start_year + frame_idx
        self.year_label_for_slider.setText(f"Year: {current_year_display}")

    def video_position_changed(self, position_ms: int):
        """Updates slider and year label when video position changes."""
        self.position_slider.setValue(position_ms)
        self._update_year_label_for_slider(position_ms)

    def video_duration_changed(self, duration_ms: int):
        """Updates slider range and UI state when video duration changes."""
        self.position_slider.setRange(0, duration_ms)
        if duration_ms == 0: # Video unloaded or invalid
            self.active_timelapse_start_year = None
            self.active_timelapse_end_year = None
            self.active_timelapse_fps = None
            self._update_year_label_for_slider(0)
            self.play_button.setEnabled(False)
            self.position_slider.setEnabled(False)

    def set_video_position_from_slider(self, position_ms: int):
        """Sets video position when user moves the slider."""
        self.media_player.setPosition(position_ms)
        self._update_year_label_for_slider(position_ms)

    def handle_media_player_error(self):
        """Handles errors from the media player."""
        self.play_button.setEnabled(False)
        self.position_slider.setEnabled(False)
        err_str = self.media_player.errorString()
        self.log_status(f"Media Player Error: {err_str}")
        QMessageBox.critical(self, "Media Player Error",
                             f"Media player error: {err_str}")
        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self._update_year_label_for_slider(0)

    def closeEvent(self, event):
        """Handles application close event, ensuring cleanup."""
        self.log_status("Closing application...")
        if self.worker_thread and self.worker_thread.is_alive():
            self.log_status("Cancelling active GEE worker...")
            self.cancel_single_analysis() # Sets worker.is_cancelled
            self.worker_thread.join(timeout=1.5) # Wait briefly
        if self.timelapse_worker_thread and self.timelapse_worker_thread.is_alive():
            self.log_status("Cancelling active time-lapse worker...")
            self.cancel_timelapse_generation() # Sets worker.is_cancelled
            self.timelapse_worker_thread.join(timeout=1.5)

        self.media_player.stop()
        if self.map_html_temp_dir:
            try:
                self.map_html_temp_dir.cleanup()
                self.log_status("Cleaned temporary map HTML directory.")
            except Exception as e:
                self.log_status(f"Error cleaning map temp dir on close: {e}")
        self.log_status("Cleanup complete. Exiting.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("GalamseyMonitorApp")
    # For a more modern look on some platforms, consider:
    # app.setStyle("Fusion")
    monitor_app = GalamseyMonitorApp()
    monitor_app.show()
    sys.exit(app.exec())