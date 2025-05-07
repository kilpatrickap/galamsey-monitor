import sys
import io
import threading
import traceback
import os
import tempfile
import glob

import ee
import requests
from PIL import Image, ImageDraw, ImageFont  # Added ImageDraw, ImageFont
import cv2
import numpy as np  # Added numpy
import folium

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QDoubleSpinBox, QTextEdit, QFormLayout,
    QGroupBox, QProgressDialog, QMessageBox, QSpinBox, QFileDialog,
    QSlider, QStyle, QSplitter, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QDate, pyqtSignal, QObject, Qt, QMetaObject, Q_ARG, QUrl
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings


# --- Worker Signals ---
class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    frame_processed = pyqtSignal(int, int)


# --- Cloud Masking Function (SCL) ---
def mask_s2_clouds_scl(image):
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])


# --- GEE Single Frame Processing Logic ---
def process_single_gee_frame(aoi_rectangle_coords, period1_start, period1_end,
                             period2_start, period2_end, threshold_val,
                             thumb_size_val, project_id, progress_emitter=None,
                             output_type='image'):
    aoi = ee.Geometry.Rectangle(aoi_rectangle_coords)

    def calculate_ndvi(image):
        return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    rgb_viz_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3, 'gamma': 1.4}
    loss_viz_params = {'palette': ['red']}

    if progress_emitter: progress_emitter(f"Frame: Processing Baseline ({period1_start}-{period1_end})...")
    collection_p1_base = s2_sr.filterBounds(aoi).filterDate(period1_start, period1_end).map(mask_s2_clouds_scl)
    count1 = collection_p1_base.size().getInfo()
    if count1 == 0:
        if progress_emitter: progress_emitter(f"Frame: No suitable cloud-free images for Baseline period.")
        return None
    median_ndvi_p1 = collection_p1_base.map(calculate_ndvi).select('NDVI').median()

    if progress_emitter: progress_emitter(f"Frame: Processing Period 2 ({period2_start}-{period2_end})...")
    collection_p2_base = s2_sr.filterBounds(aoi).filterDate(period2_start, period2_end).map(mask_s2_clouds_scl)
    count2 = collection_p2_base.size().getInfo()
    if count2 == 0:
        if progress_emitter: progress_emitter(f"Frame: No suitable cloud-free images for Period 2.")
        return None
    median_ndvi_p2 = collection_p2_base.map(calculate_ndvi).select('NDVI').median()
    median_rgb_p2 = collection_p2_base.select(['B4', 'B3', 'B2']).median()

    if progress_emitter: progress_emitter("Frame: Calculating NDVI change...")
    ndvi_change = median_ndvi_p2.subtract(median_ndvi_p1).rename('NDVI_Change')
    loss_mask = ndvi_change.lt(threshold_val).selfMask()

    if progress_emitter: progress_emitter("Frame: Creating visual layers...")
    background_layer = median_rgb_p2.visualize(**rgb_viz_params)
    loss_overlay = loss_mask.visualize(**loss_viz_params)
    final_image_viz = ee.ImageCollection([background_layer, loss_overlay]).mosaic().clip(aoi)

    if output_type == 'image':
        if progress_emitter: progress_emitter("Frame: Generating thumbnail for image output...")
        thumb_url = None
        try:
            thumb_url = final_image_viz.getThumbURL(
                {'region': aoi.getInfo()['coordinates'], 'dimensions': thumb_size_val, 'format': 'png'})
        except ee.EEException as thumb_e:
            if "No valid pixels" in str(thumb_e) or "Image.select: Pattern 'constant' did not match any bands." in str(
                    thumb_e):
                if progress_emitter: progress_emitter(
                    f"Frame: Thumbnail error ({thumb_e}). Trying background only for image.")
                try:
                    thumb_url = background_layer.getThumbURL(
                        {'region': aoi.getInfo()['coordinates'], 'dimensions': thumb_size_val, 'format': 'png'})
                except Exception as bg_thumb_e:
                    if progress_emitter: progress_emitter(
                        f"Frame: Error getting background thumbnail for image: {bg_thumb_e}")
                    return None
            else:
                if progress_emitter: progress_emitter(f"Frame: GEE error generating thumbnail: {thumb_e}")
                return None
        if not thumb_url:
            if progress_emitter: progress_emitter("Frame: Failed to generate thumbnail URL for image.")
            return None
        if progress_emitter: progress_emitter("Frame: Downloading image...")
        response = requests.get(thumb_url, timeout=60);
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    elif output_type == 'map_id':
        if progress_emitter: progress_emitter("Frame: Generating MapID for interactive map...")
        try:
            map_id_dict = ee.data.getMapId({
                'image': final_image_viz,
                'region': aoi.getInfo()['coordinates']
            })
            bounds_coords = aoi.bounds(maxError=1).getInfo()['coordinates'][0]
            return {'map_id_dict': map_id_dict, 'aoi_bounds': bounds_coords}
        except Exception as e:
            if progress_emitter: progress_emitter(f"Error getting MapID: {e}")
            raise
    else:
        raise ValueError(f"Invalid output_type specified for GEE processing: {output_type}")


# --- GEE Single Analysis Worker ---
class GEEWorker(QObject):
    def __init__(self, aoi_rectangle_coords, start1, end1, start2, end2, threshold,
                 project_id='galamsey-monitor'):
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
            pass  # Initialization might have happened already

        try:
            analysis_data = process_single_gee_frame(
                self.aoi_rectangle_coords, self.start1, self.end1, self.start2, self.end2,
                self.threshold, None, self.project_id,
                self.signals.progress.emit, output_type='map_id'
            )
            if self.is_cancelled:
                self.signals.error.emit("Single analysis was cancelled during processing.")
                return
            if analysis_data:
                self.signals.finished.emit(analysis_data)
            else:
                self.signals.error.emit("Failed to generate map data from GEE (e.g., no images found).")

        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(f"Error in GEEWorker (map_id): {e}\nTrace: {tb_str}")


# --- Time-Lapse Generation Worker (with Overlays) ---
class TimeLapseWorker(QObject):
    def __init__(self, aoi_rectangle_coords, baseline_start_date, baseline_end_date, timelapse_start_year,
                 timelapse_end_year, threshold, thumb_size=512, project_id='galamsey-monitor',
                 output_video_path="galamsey_timelapse.mp4", fps=1,
                 aoi_name="Default AOI", raw_input_coords=None):
        super().__init__()
        self.signals = WorkerSignals()
        self.aoi_rectangle_coords = aoi_rectangle_coords
        self.baseline_start_date, self.baseline_end_date = baseline_start_date, baseline_end_date
        self.timelapse_start_year, self.timelapse_end_year = timelapse_start_year, timelapse_end_year
        self.threshold = threshold
        self.thumb_size = thumb_size  # This is the GEE image content size
        self.project_id = project_id
        self.output_video_path = output_video_path
        self.fps = fps
        self.is_cancelled = False
        self.aoi_name = aoi_name
        self.raw_input_coords = raw_input_coords if raw_input_coords else [0.0, 0.0, 0.0,
                                                                           0.0]  # GUI: Lat1,Lon1,Lat2,Lon2

    def run(self):
        try:
            ee.Initialize(project=self.project_id)
        except Exception:
            pass  # Initialization might have happened already

        # --- Border and Text Definitions ---
        top_border_height = 40
        bottom_border_height = 70
        left_border_width = 160 # Reduced from 230
        right_border_width = 20  # Minimal right border, text is mostly on left/bottom
        text_color = (255, 255, 255)  # White
        border_bg_color = (0, 0, 0)  # Black

        try:
            font_path = "arial.ttf"  # Ensure this font is accessible or provide a full path
            # Or try a more generic name if arial.ttf is not standard on all target systems
            # font_path = "LiberationSans-Regular.ttf" # Example for Linux
            font_title = ImageFont.truetype(font_path, 22)
            font_coords_header = ImageFont.truetype(font_path, 15)
            font_coords_value = ImageFont.truetype(font_path, 14)
            font_bottom_info = ImageFont.truetype(font_path, 11)
            font_year = ImageFont.truetype(font_path, 16)
        except IOError:
            self.signals.progress.emit(
                f"Warning: Font '{font_path}' not found. Using default PIL font. Text quality may be lower.")
            font_title = ImageFont.load_default()
            font_coords_header = ImageFont.load_default()
            font_coords_value = ImageFont.load_default()
            font_bottom_info = ImageFont.load_default()
            font_year = ImageFont.load_default()

        content_width = self.thumb_size
        content_height = self.thumb_size

        bordered_frame_width = content_width + left_border_width + right_border_width
        bordered_frame_height = content_height + top_border_height + bottom_border_height
        video_frame_size = (bordered_frame_width, bordered_frame_height)
        self.signals.progress.emit(f"Video frame size will be: {video_frame_size[0]}x{video_frame_size[1]}")


        video_writer_initialized = False
        out_video = None
        total_frames_to_generate = self.timelapse_end_year - self.timelapse_start_year + 1
        if total_frames_to_generate <= 0:
            self.signals.error.emit("No frames requested for timelapse (start year not before end year).")
            return

        try:
            for i, year in enumerate(range(self.timelapse_start_year, self.timelapse_end_year + 1)):
                if self.is_cancelled:
                    self.signals.progress.emit("Time-lapse generation cancelled by user.")
                    break
                self.signals.frame_processed.emit(i + 1, total_frames_to_generate)
                self.signals.progress.emit(f"Processing GEE data for year: {year} ({i + 1}/{total_frames_to_generate})")

                gee_pil_image = process_single_gee_frame(
                    self.aoi_rectangle_coords, self.baseline_start_date, self.baseline_end_date,
                    f"{year}-01-01", f"{year}-12-31", self.threshold,
                    self.thumb_size, self.project_id, self.signals.progress.emit,
                    output_type='image'
                )

                current_frame_pil = Image.new('RGB', video_frame_size, border_bg_color)
                draw = ImageDraw.Draw(current_frame_pil)

                if gee_pil_image:
                    if gee_pil_image.mode == 'RGBA':
                        gee_pil_image = gee_pil_image.convert('RGB')
                    current_frame_pil.paste(gee_pil_image, (left_border_width, top_border_height))
                else:
                    self.signals.progress.emit(f"No GEE image for {year}. Drawing 'NO DATA'.")
                    no_data_text = f"NO DATA FOR {year}"
                    try:
                        text_w, text_h = draw.textbbox((0, 0), no_data_text, font=font_title)[2:4]  # PIL 9.2.0+
                    except AttributeError:
                        text_w, text_h = draw.textsize(no_data_text, font=font_title)  # Older PIL
                    draw.text(
                        (left_border_width + (content_width - text_w) / 2,
                         top_border_height + (content_height - text_h) / 2),
                        no_data_text, font=font_title, fill=text_color
                    )

                # Top Border: AOI Name
                aoi_text_content = self.aoi_name
                try:
                    text_w_title, text_h_title = draw.textbbox((0, 0), aoi_text_content, font=font_title)[2:4]
                except AttributeError:
                    text_w_title, text_h_title = draw.textsize(aoi_text_content, font=font_title)
                draw.text(
                    ((bordered_frame_width - text_w_title) / 2, (top_border_height - text_h_title) / 2),
                    aoi_text_content, font=font_title, fill=text_color
                )

                # Left Border: Coordinates
                coord_text_x_margin = 10
                coord_text_y_start = top_border_height + 15
                current_y = coord_text_y_start
                line_spacing_s = 3  # Small space after value
                line_spacing_l = 10  # Larger space after header or group

                # Helper to get text height robustly
                def get_text_height(text_sample, font_obj):
                    try:
                        return font_obj.getbbox(text_sample)[3] - font_obj.getbbox(text_sample)[1]
                    except AttributeError:
                        return font_obj.getsize(text_sample)[1]

                left_texts_config = [
                    ("Top-left-corner:", font_coords_header, line_spacing_s),
                    (f"  Lat 1: {self.raw_input_coords[0]:.6f}", font_coords_value, line_spacing_s),
                    (f"  Lon 1: {self.raw_input_coords[1]:.6f}", font_coords_value, line_spacing_l),
                    # Larger space after Lon1
                    ("Bottom-right-corner:", font_coords_header, line_spacing_s),
                    (f"  Lat 2: {self.raw_input_coords[2]:.6f}", font_coords_value, line_spacing_s),
                    (f"  Lon 2: {self.raw_input_coords[3]:.6f}", font_coords_value, line_spacing_s),
                ]
                for text, font_type, spacing_after in left_texts_config:
                    draw.text((coord_text_x_margin, current_y), text, font=font_type, fill=text_color)
                    current_y += get_text_height("Ay", font_type) + spacing_after

                # Bottom Border
                bottom_content_y_start = bordered_frame_height - bottom_border_height + 10
                try:
                    _, lh_bottom_info = font_bottom_info.getbbox("A")[3], font_bottom_info.getbbox("A")[1]
                except AttributeError:
                    lh_bottom_info = font_bottom_info.getsize("A")[1]

                loss_text = "Red shows potential Vegetative Loss"
                draw.text((left_border_width + 10, bottom_content_y_start), loss_text, font=font_bottom_info,
                          fill=text_color)

                year_text_content = f"YEAR: {year}"
                try:
                    year_text_w, year_text_h = draw.textbbox((0, 0), year_text_content, font=font_year)[2:4]
                except AttributeError:
                    year_text_w, year_text_h = draw.textsize(year_text_content, font=font_year)
                # Align year text vertically with loss_text
                year_text_y_offset = (lh_bottom_info - year_text_h) / 2
                draw.text(
                    (bordered_frame_width - right_border_width - year_text_w - 10,
                     bottom_content_y_start + year_text_y_offset),
                    year_text_content, font=font_year, fill=text_color
                )

                second_line_y = bottom_content_y_start + lh_bottom_info + 8

                copyright_text = "Copyright KilTech Ent 2025"
                draw.text((left_border_width + 10, second_line_y), copyright_text, font=font_bottom_info,
                          fill=text_color)

                rights_text = "All rights reserved."
                try:
                    rights_text_w, _ = draw.textbbox((0, 0), rights_text, font=font_bottom_info)[2:4]
                except AttributeError:
                    rights_text_w, _ = draw.textsize(rights_text, font=font_bottom_info)
                draw.text(
                    (bordered_frame_width - right_border_width - rights_text_w - 10, second_line_y),
                    rights_text, font=font_bottom_info, fill=text_color
                )

                frame_cv = cv2.cvtColor(np.array(current_frame_pil), cv2.COLOR_RGB2BGR)

                if not video_writer_initialized:
                    self.signals.progress.emit("Initializing video writer...")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_output_dir = os.path.dirname(self.output_video_path)
                    if video_output_dir and not os.path.exists(video_output_dir):
                        os.makedirs(video_output_dir, exist_ok=True)
                    out_video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, video_frame_size)
                    if not out_video.isOpened():
                        self.signals.error.emit(f"FATAL: Could not open video writer for {self.output_video_path}.")
                        return
                    video_writer_initialized = True

                out_video.write(frame_cv)
                self.signals.progress.emit(f"Added frame for {year} to video.")

            if video_writer_initialized and out_video:
                self.signals.progress.emit("Finalizing video...")
                out_video.release()
                self.signals.progress.emit(f"Video compilation complete: {self.output_video_path}")
                self.signals.finished.emit(self.output_video_path)
            elif not video_writer_initialized and total_frames_to_generate > 0:
                self.signals.error.emit("No valid frames were processed to create the video.")

        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(f"Time-lapse overlay/compilation error: {e}\nTrace: {tb_str}")
            if video_writer_initialized and out_video:
                out_video.release()


# --- Main Application Window ---
class GalamseyMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galamsey Monitor")
        self.setGeometry(10, 40, 1400, 750)  # Adjusted for typical screen
        self.worker_thread = None;
        self.worker = None
        self.timelapse_worker_thread = None;
        self.timelapse_worker = None
        self.progress_dialog = None
        self.project_id = 'galamsey-monitor'  # Replace with your GEE Project ID if different
        self.map_html_temp_dir = None
        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None
        self.final_video_path = None
        self.media_player = QMediaPlayer();
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.play_button = QPushButton();
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay));
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)
        self.position_slider = QSlider(Qt.Orientation.Horizontal);
        self.position_slider.setRange(0, 0);
        self.position_slider.sliderMoved.connect(self.set_video_position_from_slider);
        self.position_slider.setEnabled(False)
        self.year_label_for_slider = QLabel("Year: -")
        self.media_player.playbackStateChanged.connect(self.media_state_changed);
        self.media_player.positionChanged.connect(self.video_position_changed)
        self.media_player.durationChanged.connect(self.video_duration_changed);
        self.media_player.errorOccurred.connect(self.handle_media_player_error)
        main_layout = QVBoxLayout(self)
        single_analysis_group = QGroupBox("Single Period Analysis");
        single_analysis_form_layout = QFormLayout()
        self.aoi_name_label = QLabel("AOI:");
        self.aoi_name_input = QLineEdit();
        self.aoi_name_input.setPlaceholderText("e.g. Anyinam");
        self.aoi_name_input.setToolTip("Enter a descriptive name for this Area of Interest (Mandatory for Video).")
        single_analysis_form_layout.addRow(self.aoi_name_label, self.aoi_name_input)
        self.coord_input = QLineEdit("6.401452, -0.594587, 6.355603, -0.496084");
        self.coord_input.setToolTip("Enter coordinates as: Lat1, Lon1, Lat2, Lon2 (e.g., 6.3, -1.8, 6.4, -1.7)")
        self.date1_start = QDateEdit(QDate(2020, 1, 1));
        self.date1_end = QDateEdit(QDate(2020, 12, 31))
        self.date2_start = QDateEdit(QDate(2025, 1, 1));
        self.date2_end = QDateEdit(QDate(QDate.currentDate()))
        self.threshold_input = QDoubleSpinBox();
        self.threshold_input.setRange(-1.0, 0.0);
        self.threshold_input.setSingleStep(0.05);
        self.threshold_input.setValue(-0.20)
        self.analyze_button = QPushButton("Analyze Single Period")
        for dt_edit in [self.date1_start, self.date1_end, self.date2_start, self.date2_end]: dt_edit.setCalendarPopup(
            True); dt_edit.setDisplayFormat("dd-MM-yyyy")
        single_analysis_form_layout.addRow(QLabel("AOI Coords (Lat1, Lon1, Lat2, Lon2):"), self.coord_input)
        single_analysis_form_layout.addRow(QLabel("Period 1 Start (Baseline):"), self.date1_start);
        single_analysis_form_layout.addRow(QLabel("Period 1 End (Baseline):"), self.date1_end)
        single_analysis_form_layout.addRow(QLabel("Period 2 Start (Comparison):"), self.date2_start);
        single_analysis_form_layout.addRow(QLabel("Period 2 End (Comparison):"), self.date2_end)
        single_analysis_form_layout.addRow(QLabel("NDVI Change Threshold:"), self.threshold_input);
        single_analysis_form_layout.addRow(self.analyze_button)
        single_analysis_group.setLayout(single_analysis_form_layout);
        single_analysis_group.setMinimumWidth(450)  # Increased min width
        map_group = QGroupBox("Interactive Map Preview (Red shows potential vegetation loss)");
        map_v_layout = QVBoxLayout()
        self.map_view = QWebEngineView();
        self.map_view.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True);
        self.map_view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True);
        self.map_view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.map_view.setHtml(
            "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;color:grey;'><p>Map will appear here after analysis.</p></body></html>");
        self.map_view.setMinimumSize(400, 300)
        map_v_layout.addWidget(self.map_view);
        map_group.setLayout(map_v_layout)
        top_h_splitter = QSplitter(Qt.Orientation.Horizontal);
        top_h_splitter.addWidget(single_analysis_group);
        top_h_splitter.addWidget(map_group);
        top_h_splitter.setSizes([400, 700])  # Adjusted initial split
        timelapse_video_group = QGroupBox("Time-Lapse Video");
        main_timelapse_h_layout = QHBoxLayout()
        timelapse_controls_panel = QWidget();
        left_v_layout = QVBoxLayout(timelapse_controls_panel);
        left_v_layout.setContentsMargins(0, 5, 5, 5)
        self.timelapse_start_year_input = QSpinBox();
        self.timelapse_start_year_input.setRange(2000, QDate.currentDate().year());
        self.timelapse_start_year_input.setValue(2021)
        self.timelapse_end_year_input = QSpinBox();
        self.timelapse_end_year_input.setRange(2000, QDate.currentDate().year() + 5);
        self.timelapse_end_year_input.setValue(QDate.currentDate().year())
        self.timelapse_fps_input = QSpinBox();
        self.timelapse_fps_input.setRange(1, 30);
        self.timelapse_fps_input.setValue(1)
        self.generate_timelapse_button = QPushButton("Generate & Load Time-Lapse Video")
        timelapse_form_inputs_layout = QFormLayout();
        timelapse_form_inputs_layout.addRow(QLabel("Time-Lapse Start Year:"), self.timelapse_start_year_input);
        timelapse_form_inputs_layout.addRow(QLabel("Time-Lapse End Year:"), self.timelapse_end_year_input);
        timelapse_form_inputs_layout.addRow(QLabel("Video FPS:"), self.timelapse_fps_input)
        left_v_layout.addLayout(timelapse_form_inputs_layout);
        left_v_layout.addWidget(self.generate_timelapse_button);
        left_v_layout.addStretch(1)
        timelapse_controls_panel.setMinimumWidth(220);
        timelapse_controls_panel.setMaximumWidth(300)  # Adjusted for tighter layout
        video_display_panel = QWidget();
        right_v_layout = QVBoxLayout(video_display_panel);
        right_v_layout.setContentsMargins(0, 0, 0, 0)
        video_player_controls_layout = QHBoxLayout();
        video_player_controls_layout.addWidget(self.play_button);
        video_player_controls_layout.addWidget(self.position_slider, 1);
        video_player_controls_layout.addWidget(self.year_label_for_slider)
        right_v_layout.addWidget(self.video_widget, 1);
        right_v_layout.addLayout(video_player_controls_layout);
        self.video_widget.setMinimumHeight(200)
        main_timelapse_h_layout.addWidget(timelapse_controls_panel);
        main_timelapse_h_layout.addWidget(video_display_panel, 1);
        timelapse_video_group.setLayout(main_timelapse_h_layout)
        status_group = QGroupBox("Status Log");
        status_v_layout = QVBoxLayout();
        self.status_log = QTextEdit();
        self.status_log.setReadOnly(True);
        self.status_log.setMinimumHeight(80);
        self.status_log.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        status_v_layout.addWidget(self.status_log);
        status_group.setLayout(status_v_layout);
        main_v_splitter = QSplitter(Qt.Orientation.Vertical);
        main_v_splitter.addWidget(top_h_splitter);
        main_v_splitter.addWidget(timelapse_video_group);
        main_v_splitter.addWidget(status_group);
        main_v_splitter.setSizes([300, 370, 80])  # Give more space to video, less to log and top
        main_layout.addWidget(main_v_splitter)
        self.analyze_button.clicked.connect(self.run_single_analysis);
        self.generate_timelapse_button.clicked.connect(self.run_timelapse_generation)
        self.init_gee_check()

    def init_gee_check(self):
        self.log_status("Attempting GEE initialization...");
        try:
            ee.Initialize(project=self.project_id);
            self.log_status(f"GEE initialized successfully (Project: {self.project_id}).");
            # Perform a simple GEE operation to confirm connectivity and permissions
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(
                1).size().getInfo();  # SR_HARMONIZED is generally available
            self.log_status("GEE test query successful.")
        except Exception as e:
            self.handle_gee_init_error(e)

    def handle_gee_init_error(self, e):
        msg = (f"GEE Initialization Error: {e}\n\n"
               "Please ensure:\n"
               "1. You have run 'earthengine authenticate' in your terminal and authenticated successfully.\n"
               "2. You have an active internet connection.\n"
               f"3. The GEE Project ID ('{self.project_id}') is correct and the Earth Engine API is enabled for this project in Google Cloud Console.\n"
               "4. You have the necessary permissions for the specified GEE project.\n\n"
               "Please restart the application after verifying these points.")
        self.log_status(msg.replace("\n\n", "\n").replace("\n", "\nStatus: "));  # Log concise status
        QMessageBox.critical(self, "GEE Initialization Error", msg)
        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)

    def log_status(self, message):
        self.status_log.append(message);
        QApplication.processEvents()  # Ensure GUI updates

    def setup_progress_dialog(self, title="Processing..."):
        if self.progress_dialog and self.progress_dialog.isVisible(): self.progress_dialog.close()
        self.progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self);
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal);
        self.progress_dialog.setAutoClose(False);  # We control closing
        self.progress_dialog.setAutoReset(False)  # We control reset
        self.progress_dialog.setMinimumDuration(0);  # Show immediately
        self.progress_dialog.setValue(0);
        return self.progress_dialog

    def _parse_coordinates(self):
        coords_str_list = self.coord_input.text().strip().split(',')
        if len(coords_str_list) != 4:
            raise ValueError("Coordinate input requires 4 numbers (Lat1, Lon1, Lat2, Lon2) separated by commas.")
        try:
            # These are the coordinates as entered by the user: [Lat1_gui, Lon1_gui, Lat2_gui, Lon2_gui]
            raw_coords_float = [float(c.strip()) for c in coords_str_list]
        except ValueError:
            raise ValueError("Invalid number format in coordinates.")

        lat1, lon1, lat2, lon2 = raw_coords_float
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
            raise ValueError("Latitudes must be between -90 and 90.")
        if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            raise ValueError("Longitudes must be between -180 and 180.")

        # For ee.Geometry.Rectangle: [xmin, ymin, xmax, ymax]
        aoi_ee_rect_coords = [min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)]
        return raw_coords_float, aoi_ee_rect_coords

    def run_single_analysis(self):
        self.log_status("Starting single period analysis (interactive map)...");
        aoi_name = self.aoi_name_input.text().strip()
        if aoi_name:
            self.log_status(f"AOI Name: {aoi_name}")

        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)
        self.map_view.setHtml(
            "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;'><h2>Processing analysis...</h2><p>Map will load shortly.</p></body></html>")
        try:
            _, aoi_ee_rect_coords = self._parse_coordinates()  # raw_coords_float is ignored here
            start1 = self.date1_start.date().toString("yyyy-MM-dd");
            end1 = self.date1_end.date().toString("yyyy-MM-dd")
            start2 = self.date2_start.date().toString("yyyy-MM-dd");
            end2 = self.date2_end.date().toString("yyyy-MM-dd")
            if self.date1_start.date() >= self.date1_end.date() or \
                    self.date2_start.date() >= self.date2_end.date() or \
                    self.date1_end.date() >= self.date2_start.date():  # Check for valid, non-overlapping ranges
                raise ValueError(
                    "Date ranges are invalid or overlapping. Ensure P1_End < P2_Start, and P1_Start < P1_End, P2_Start < P2_End.")
        except ValueError as ve:
            self.log_status(f"Input Error: {ve}");
            QMessageBox.warning(self, "Input Error", str(ve))
            self.map_view.setHtml(f"<html><body><h2>Input Error:</h2><pre>{ve}</pre></body></html>")
            self.analyze_button.setEnabled(True);
            self.generate_timelapse_button.setEnabled(True);
            return

        pd = self.setup_progress_dialog("Single Analysis in Progress...");
        pd.setRange(0, 0);  # Indeterminate progress for single analysis map
        pd.canceled.connect(self.cancel_single_analysis);
        pd.show()

        self.worker = GEEWorker(aoi_ee_rect_coords, start1, end1, start2, end2, self.threshold_input.value(),
                                project_id=self.project_id)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker.signals.finished.connect(self.on_single_analysis_complete_map)
        self.worker.signals.error.connect(self.on_single_analysis_error_map)
        self.worker.signals.progress.connect(self.update_progress_label_only);
        self.worker_thread.start()

    def update_progress_label_only(self, message):
        self.log_status(message)
        if self.progress_dialog and self.progress_dialog.isVisible():
            # Ensure this is called on the main thread for GUI updates
            QMetaObject.invokeMethod(self.progress_dialog, "setLabelText", Qt.ConnectionType.QueuedConnection,
                                     Q_ARG(str, message))

    def on_single_analysis_complete_map(self, analysis_data):
        self.log_status("Single analysis (map data) received.")
        if self.progress_dialog: self.progress_dialog.close()

        map_id_dict = analysis_data.get('map_id_dict')
        aoi_bounds_gee = analysis_data.get('aoi_bounds')  # These are [lon, lat] pairs for the GEE geometry bounds

        if not map_id_dict or not aoi_bounds_gee:
            self.log_status("Error: MapID or AOI bounds missing from GEE result.")
            self.map_view.setHtml(
                "<html><body><h2>Error:</h2><p>Could not retrieve map data from GEE.</p></body></html>")
            self.analyze_button.setEnabled(True);
            self.generate_timelapse_button.setEnabled(True)
            return

        try:
            # Calculate center for Folium map from GEE bounds
            lons = [p[0] for p in aoi_bounds_gee]
            lats = [p[1] for p in aoi_bounds_gee]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2

            folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                                    tiles=None)  # Start with no base tiles
            # Add Google Hybrid as a default visible layer
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                # y=satellite, s=hybrid, m=roadmap, t=terrain
                attr='Google', name='Google Hybrid', overlay=False, control=True, show=True  # Show this by default
            ).add_to(folium_map)
            # Add Google Roadmap as an option
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                attr='Google', name='Google Roadmap', overlay=False, control=True
            ).add_to(folium_map)
            # Add GEE layer
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine Analysis',
                name='GEE Analysis Layer',
                overlay=True,  # This is an overlay
                control=True,
                show=True,  # Show GEE layer by default
                max_native_zoom=18  # Adjust if needed based on GEE layer resolution
            ).add_to(folium_map)
            folium.LayerControl().add_to(folium_map)  # Add layer control to switch layers

            if self.map_html_temp_dir:  # Cleanup previous temp dir if it exists
                try:
                    self.map_html_temp_dir.cleanup()
                except Exception as e_clean:
                    self.log_status(f"Note: Error cleaning up previous map temp dir: {e_clean}")

            self.map_html_temp_dir = tempfile.TemporaryDirectory(prefix="galamsey_map_html_")
            map_html_path = os.path.join(self.map_html_temp_dir.name, "interactive_map.html")

            folium_map.save(map_html_path)
            self.log_status(f"Interactive map saved to temporary file: {map_html_path}")
            self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(map_html_path)))  # Load local file
            self.log_status("Interactive map loaded into view.")

        except Exception as e:
            self.log_status(f"Error creating/displaying Folium map: {e}\n{traceback.format_exc()}")
            self.map_view.setHtml(f"<html><body><h2>Map Display Error:</h2><pre>{e}</pre></body></html>")

        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def on_single_analysis_error_map(self, error_message):
        self.log_status(f"Single Analysis (map) Error: {error_message}")
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.critical(self, "Single Analysis Error", error_message)
        self.map_view.setHtml(f"<html><body><h2>Analysis Failed:</h2><pre>{error_message}</pre></body></html>")
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def cancel_single_analysis(self):
        self.log_status("Attempting to cancel single analysis...")
        if self.worker: self.worker.is_cancelled = True
        if self.progress_dialog: self.progress_dialog.close()  # Close it on cancel
        self.map_view.setHtml(
            "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;color:grey;'><p>Analysis Cancelled. Map will appear here.</p></body></html>")
        self.analyze_button.setEnabled(True);  # Re-enable
        self.generate_timelapse_button.setEnabled(True)

    def run_timelapse_generation(self):
        self.log_status("Starting time-lapse video generation...");

        aoi_name_from_input = self.aoi_name_input.text().strip()
        if not aoi_name_from_input:
            self.log_status("Input Error TL: AOI name is mandatory for video generation.");
            QMessageBox.warning(self, "Input Error",
                                "AOI name is mandatory for video generation.\nPlease enter a name in the 'AOI:' field.")
            return  # Do not proceed, keep buttons disabled until valid input

        temp_sanitized_aoi_name = "".join(c if c.isalnum() or c == '_' else '_' for c in aoi_name_from_input)
        final_sanitized_aoi_name = "_".join(s for s in temp_sanitized_aoi_name.split('_') if s).strip(
            '_')  # Remove leading/trailing

        if not final_sanitized_aoi_name:
            self.log_status(
                f"Input Error TL: AOI name '{aoi_name_from_input}' sanitized to an empty/invalid string. Please use alphanumeric characters.");
            QMessageBox.warning(self, "Input Error",
                                f"AOI name '{aoi_name_from_input}' is invalid for a filename (e.g., contains only special characters or spaces). Please use a name with alphanumeric characters and underscores.")
            return

        self.log_status(f"Using AOI Name for Timelapse: {final_sanitized_aoi_name}")

        self.analyze_button.setEnabled(False);  # Disable buttons during processing
        self.generate_timelapse_button.setEnabled(False)
        self.active_timelapse_start_year = None;  # Reset active timelapse info
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None;
        self._update_year_label_for_slider(0)  # Reset year label

        try:
            # raw_coords_input are the [Lat1, Lon1, Lat2, Lon2] from GUI
            raw_coords_input, aoi_ee_rect_coords = self._parse_coordinates()
            baseline_start = self.date1_start.date().toString("yyyy-MM-dd");
            baseline_end = self.date1_end.date().toString("yyyy-MM-dd")
            tl_start_year = self.timelapse_start_year_input.value();
            tl_end_year = self.timelapse_end_year_input.value()

            if self.date1_start.date() >= self.date1_end.date(): raise ValueError(
                "Baseline date range is invalid (start must be before end).")
            if tl_start_year > tl_end_year: raise ValueError(
                "Timelapse year range is invalid (start year must be before or same as end year).")
            if QDate(tl_start_year, 1, 1) <= self.date1_end.date():  # Ensure timelapse starts after baseline
                raise ValueError("Timelapse start year must be after the baseline period's end date.")

            coord_fn_parts_str = '_'.join([str(c).replace('.', 'p').replace('-', 'm') for c in raw_coords_input])
            video_filename_base = f"{final_sanitized_aoi_name}_{coord_fn_parts_str}_{tl_start_year}-{tl_end_year}_timelapse.mp4"

            videos_dir = os.path.join(os.getcwd(), "videos");
            os.makedirs(videos_dir, exist_ok=True)
            self.final_video_path = os.path.join(videos_dir, video_filename_base)
            self.log_status(f"Video will be saved to: {self.final_video_path}")

        except ValueError as ve:
            self.log_status(f"Input Error (Timelapse Setup): {ve}");
            QMessageBox.warning(self, "Input Error", str(ve))
            self.analyze_button.setEnabled(True);  # Re-enable buttons on input error
            self.generate_timelapse_button.setEnabled(True);
            return

        pd = self.setup_progress_dialog("Generating Time-Lapse Video...");
        # Total frames = (end_year - start_year + 1)
        total_frames_to_process = tl_end_year - tl_start_year + 1
        pd.setRange(0, total_frames_to_process if total_frames_to_process > 0 else 1)  # Ensure range is at least 0-1
        pd.canceled.connect(self.cancel_timelapse_generation);
        pd.show()

        self.timelapse_worker = TimeLapseWorker(
            aoi_rectangle_coords=aoi_ee_rect_coords,
            baseline_start_date=baseline_start,
            baseline_end_date=baseline_end,
            timelapse_start_year=tl_start_year,
            timelapse_end_year=tl_end_year,
            threshold=self.threshold_input.value(),
            thumb_size=768,  # Increased from 512
            project_id=self.project_id,
            output_video_path=self.final_video_path,
            fps=self.timelapse_fps_input.value(),
            aoi_name=final_sanitized_aoi_name,
            raw_input_coords=raw_coords_input
        )
        self.timelapse_worker_thread = threading.Thread(target=self.timelapse_worker.run, daemon=True)
        self.timelapse_worker.signals.finished.connect(self.on_timelapse_complete);
        self.timelapse_worker.signals.error.connect(self.on_timelapse_error)
        self.timelapse_worker.signals.progress.connect(self.update_progress_label_only);
        self.timelapse_worker.signals.frame_processed.connect(self.update_timelapse_progress_value)
        self.timelapse_worker_thread.start()

    def update_timelapse_progress_value(self, current_frame_number, total_frames):
        # This is called by frame_processed signal
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.setRange(0, total_frames)  # Update range in case it changed
            self.progress_dialog.setValue(current_frame_number)

    def on_timelapse_complete(self, video_path):
        self.log_status(f"Time-Lapse video generated successfully: {video_path}");
        if self.progress_dialog: self.progress_dialog.close()
        # Store active timelapse info for media player
        self.active_timelapse_start_year = self.timelapse_start_year_input.value();
        self.active_timelapse_end_year = self.timelapse_end_year_input.value();
        self.active_timelapse_fps = self.timelapse_fps_input.value()

        self.media_player.setSource(QUrl.fromLocalFile(video_path));
        self.play_button.setEnabled(True);
        self.position_slider.setEnabled(True)
        self._update_year_label_for_slider(0);  # Reset to start
        QMessageBox.information(self, "Time-Lapse Ready",
                                f"Time-Lapse video has been generated and loaded.\nSaved to: {video_path}")
        self.analyze_button.setEnabled(True);  # Re-enable buttons
        self.generate_timelapse_button.setEnabled(True)

    def on_timelapse_error(self, error_message):
        self.log_status(f"Time-Lapse Generation Error: {error_message}");
        if self.progress_dialog: self.progress_dialog.close();
        QMessageBox.critical(self, "Time-Lapse Error", error_message)
        self.analyze_button.setEnabled(True);  # Re-enable buttons
        self.generate_timelapse_button.setEnabled(True)
        # Reset active timelapse info as it failed
        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None;
        self._update_year_label_for_slider(0)

    def cancel_timelapse_generation(self):
        self.log_status("Cancelling time-lapse generation...");
        if self.timelapse_worker: self.timelapse_worker.is_cancelled = True
        if self.progress_dialog: self.progress_dialog.close();
        self.analyze_button.setEnabled(True);  # Re-enable
        self.generate_timelapse_button.setEnabled(True)
        # Reset active timelapse info
        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None;
        self._update_year_label_for_slider(0)

    def play_video(self):
        if self.media_player.source().isEmpty():
            self.log_status("No video loaded to play.");
            return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause();
            self.log_status("Video paused.")
        else:
            self.media_player.play();
            self.log_status("Video playing.")

    def media_state_changed(self, state: QMediaPlayer.PlaybackState):
        self.play_button.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaPause if state == QMediaPlayer.PlaybackState.PlayingState else QStyle.StandardPixmap.SP_MediaPlay
            )
        )
        # If stopped at the end, keep the last year displayed
        if state == QMediaPlayer.PlaybackState.StoppedState and \
                self.active_timelapse_start_year is not None and \
                self.media_player.position() == self.media_player.duration() and \
                self.media_player.duration() > 0:
            self._update_year_label_for_slider(self.media_player.duration())  # Show last year

    def _update_year_label_for_slider(self, position_ms):
        if self.active_timelapse_start_year is not None and \
                self.active_timelapse_end_year is not None and \
                self.active_timelapse_fps is not None and self.active_timelapse_fps > 0:

            ms_per_frame = 1000.0 / self.active_timelapse_fps
            if ms_per_frame <= 0:  # Avoid division by zero or negative
                self.year_label_for_slider.setText("Year: - (FPS Error)");
                return

            current_frame_index_float = position_ms / ms_per_frame
            # Ensure index is within bounds [0, num_total_frames - 1]
            num_total_frames = (self.active_timelapse_end_year - self.active_timelapse_start_year + 1)

            # Round to nearest frame for display, or floor, depending on desired behavior.
            # Using floor to match typical video player behavior (shows frame until next one starts)
            current_frame_index = int(current_frame_index_float)
            current_frame_index = max(0, min(current_frame_index, num_total_frames - 1))  # Clamp

            current_year = self.active_timelapse_start_year + current_frame_index
            self.year_label_for_slider.setText(f"Year: {current_year}")
        else:
            self.year_label_for_slider.setText("Year: -")

    def video_position_changed(self, position_ms):  # position is in milliseconds
        self.position_slider.setValue(position_ms);
        self._update_year_label_for_slider(position_ms)

    def video_duration_changed(self, duration_ms):  # duration is in milliseconds
        self.position_slider.setRange(0, duration_ms)
        if duration_ms == 0:  # Video unloaded or invalid
            self.active_timelapse_start_year = None;
            self.active_timelapse_end_year = None;
            self.active_timelapse_fps = None;
            self._update_year_label_for_slider(0);
            self.play_button.setEnabled(False);
            self.position_slider.setEnabled(False)

    def set_video_position_from_slider(self, position_ms):
        self.media_player.setPosition(position_ms);
        self._update_year_label_for_slider(position_ms)  # Update label immediately

    def handle_media_player_error(self):
        self.play_button.setEnabled(False);
        self.position_slider.setEnabled(False);
        error_string = self.media_player.errorString()
        self.log_status(f"Media Player Error: {error_string}");
        QMessageBox.critical(self, "Media Player Error", f"An error occurred with the media player: {error_string}")
        # Reset active timelapse info
        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None;
        self._update_year_label_for_slider(0)

    def closeEvent(self, event):
        self.log_status("Closing application...");
        # Attempt to cancel any running workers
        if self.worker_thread and self.worker_thread.is_alive():
            self.log_status("Cancelling active single analysis worker...")
            self.cancel_single_analysis()  # This sets worker.is_cancelled
            self.worker_thread.join(timeout=2)  # Wait briefly for thread to finish
        if self.timelapse_worker_thread and self.timelapse_worker_thread.is_alive():
            self.log_status("Cancelling active time-lapse worker...")
            self.cancel_timelapse_generation()  # This sets worker.is_cancelled
            self.timelapse_worker_thread.join(timeout=2)  # Wait briefly

        self.media_player.stop()  # Stop media player
        if self.map_html_temp_dir:
            try:
                self.map_html_temp_dir.cleanup()
                self.log_status("Cleaned up temporary map HTML directory.")
            except Exception as e:
                self.log_status(f"Error cleaning up map HTML temp dir on close: {e}")
        self.log_status("Cleanup complete. Exiting.");
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("GalamseyMonitorApp")
    # For better cross-platform look, consider Fusion style
    # app.setStyle("Fusion")
    monitor_app = GalamseyMonitorApp()
    monitor_app.show()
    sys.exit(app.exec())