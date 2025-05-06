import sys
import io
import threading
import traceback
import os
import tempfile
import glob

import ee
import requests
from PIL import Image
import cv2  # OpenCV for video creation

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QDoubleSpinBox, QTextEdit, QFormLayout,
    QGroupBox, QProgressDialog, QMessageBox, QSpinBox, QFileDialog,
    QSlider, QStyle  # Added QSlider, QStyle
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QDate, pyqtSignal, QObject, Qt, QMetaObject, Q_ARG, QUrl  # Added QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QMediaFormat  # Added QMediaPlayer, QMediaFormat
from PyQt6.QtMultimediaWidgets import QVideoWidget  # Added QVideoWidget


# --- Worker Signals ---
class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    frame_processed = pyqtSignal(int, int)


# --- Cloud Masking Function (SCL) ---
def mask_s2_clouds_scl(image):
    scl = image.select('SCL')
    unwanted_classes = [3, 8, 9, 10]
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])


# --- GEE Single Frame Processing Logic ---
def process_single_gee_frame(aoi_coords, period1_start, period1_end,
                             period2_start, period2_end, threshold_val,
                             thumb_size_val, project_id, progress_emitter=None):
    if progress_emitter:
        progress_emitter(f"Initializing GEE for frame ({period2_start}-{period2_end})...")
    try:
        ee.Initialize(project=project_id)
    except Exception as init_e:
        if progress_emitter: progress_emitter(f"GEE Init failed for frame: {init_e}")
        raise init_e
    aoi = ee.Geometry.Rectangle(aoi_coords)

    def calculate_ndvi(image):
        return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    rgb_viz_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3, 'gamma': 1.4}
    loss_viz_params = {'palette': ['red']}

    if progress_emitter: progress_emitter(f"Frame: Processing Baseline ({period1_start}-{period1_end})...")
    collection_p1_base = s2_sr.filterBounds(aoi).filterDate(period1_start, period1_end).map(mask_s2_clouds_scl)
    count1 = collection_p1_base.size().getInfo()
    if count1 == 0:
        if progress_emitter: progress_emitter(f"Frame: No cloud-free images for Baseline.")
        return None
    median_ndvi_p1 = collection_p1_base.map(calculate_ndvi).select('NDVI').median()

    if progress_emitter: progress_emitter(f"Frame: Processing Period 2 ({period2_start}-{period2_end})...")
    collection_p2_base = s2_sr.filterBounds(aoi).filterDate(period2_start, period2_end).map(mask_s2_clouds_scl)
    count2 = collection_p2_base.size().getInfo()
    if count2 == 0:
        if progress_emitter: progress_emitter(f"Frame: No cloud-free images for Period 2.")
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

    if progress_emitter: progress_emitter("Frame: Generating thumbnail...")
    thumb_url = None
    try:
        thumb_url = final_image_viz.getThumbURL(
            {'region': aoi.getInfo()['coordinates'], 'dimensions': thumb_size_val, 'format': 'png'})
    except ee.EEException as thumb_e:
        if "No valid pixels" in str(thumb_e) or "Image has no bands" in str(thumb_e):
            if progress_emitter: progress_emitter("Frame: No significant loss. Generating background only.")
            try:
                thumb_url = background_layer.getThumbURL(
                    {'region': aoi.getInfo()['coordinates'], 'dimensions': thumb_size_val, 'format': 'png'})
            except Exception as bg_thumb_e:
                if progress_emitter: progress_emitter(f"Frame: Error getting background thumbnail: {bg_thumb_e}")
                raise bg_thumb_e
        else:
            raise thumb_e
    if not thumb_url:
        if progress_emitter: progress_emitter("Frame: Failed to generate thumbnail URL.")
        return None
    if progress_emitter: progress_emitter("Frame: Downloading image...")
    response = requests.get(thumb_url);
    response.raise_for_status()
    img_data = response.content
    pil_image = Image.open(io.BytesIO(img_data))
    return pil_image


# --- GEE Single Analysis Worker ---
class GEEWorker(QObject):
    def __init__(self, aoi_coords, start1, end1, start2, end2, threshold, thumb_size=512,
                 project_id='galamsey-monitor'):
        super().__init__()
        self.signals = WorkerSignals();
        self.aoi_coords = aoi_coords;
        self.start1 = start1;
        self.end1 = end1
        self.start2 = start2;
        self.end2 = end2;
        self.threshold = threshold;
        self.thumb_size = thumb_size
        self.project_id = project_id;
        self.is_cancelled = False

    def run(self):
        try:
            pil_image = process_single_gee_frame(self.aoi_coords, self.start1, self.end1, self.start2, self.end2,
                                                 self.threshold, self.thumb_size, self.project_id,
                                                 self.signals.progress.emit)
            self.signals.finished.emit(pil_image)
        except Exception as e:
            tb_str = traceback.format_exc();
            self.signals.error.emit(f"Error in GEEWorker: {e}\nTrace: {tb_str}")


# --- Time-Lapse Generation Worker ---
class TimeLapseWorker(QObject):
    def __init__(self, aoi_coords, baseline_start_date, baseline_end_date, timelapse_start_year,
                 timelapse_end_year, threshold, thumb_size=512, project_id='galamsey-monitor',
                 output_video_path="galamsey_timelapse.mp4", fps=1):
        super().__init__()
        self.signals = WorkerSignals();
        self.aoi_coords = aoi_coords;
        self.baseline_start_date = baseline_start_date
        self.baseline_end_date = baseline_end_date;
        self.timelapse_start_year = timelapse_start_year
        self.timelapse_end_year = timelapse_end_year;
        self.threshold = threshold;
        self.thumb_size = thumb_size
        self.project_id = project_id;
        self.output_video_path = output_video_path;
        self.fps = fps;
        self.is_cancelled = False

    def run(self):
        temp_dir = tempfile.mkdtemp(prefix="galamsey_frames_")
        self.signals.progress.emit(f"Temp frames in: {temp_dir}")
        frame_paths = [];
        total_frames = self.timelapse_end_year - self.timelapse_start_year + 1
        try:
            for i, year in enumerate(range(self.timelapse_start_year, self.timelapse_end_year + 1)):
                if self.is_cancelled: self.signals.progress.emit("Time-lapse cancelled."); break
                self.signals.frame_processed.emit(i + 1, total_frames)
                self.signals.progress.emit(f"Processing frame for year: {year} ({i + 1}/{total_frames})")
                try:
                    pil_image = process_single_gee_frame(self.aoi_coords, self.baseline_start_date,
                                                         self.baseline_end_date,
                                                         f"{year}-01-01", f"{year}-12-31", self.threshold,
                                                         self.thumb_size, self.project_id, self.signals.progress.emit)
                    if pil_image:
                        if pil_image.mode == 'RGBA': pil_image = pil_image.convert('RGB')
                        frame_filename = os.path.join(temp_dir, f"frame_{i:04d}.png")
                        pil_image.save(frame_filename);
                        frame_paths.append(frame_filename)
                        self.signals.progress.emit(f"Saved frame for {year}: {frame_filename}")
                    else:
                        self.signals.progress.emit(f"Skipping frame for {year} (no image).")
                except Exception as frame_e:
                    self.signals.progress.emit(f"Error for year {year}: {frame_e}. Skipping.")
            if self.is_cancelled: self.signals.error.emit("Cancelled before video compilation."); return
            if not frame_paths: self.signals.error.emit("No frames generated."); return
            self.signals.progress.emit("Compiling video...");
            first_frame_img = cv2.imread(frame_paths[0])
            if first_frame_img is None: self.signals.error.emit(f"Could not read first frame: {frame_paths[0]}"); return
            height, width, _ = first_frame_img.shape;
            size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v');
            out_video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, size)
            for frame_path in sorted(frame_paths):
                img = cv2.imread(frame_path)
                if img is not None:
                    out_video.write(img)
                else:
                    self.signals.progress.emit(f"Warning: Could not read {frame_path} for video.")
            out_video.release();
            self.signals.progress.emit(f"Video complete: {self.output_video_path}")
            self.signals.finished.emit(self.output_video_path)
        except Exception as e:
            tb_str = traceback.format_exc(); self.signals.error.emit(f"Time-lapse error: {e}\nTrace: {tb_str}")
        finally:
            self.signals.progress.emit("Cleaning up temp frames...")
            for fp in frame_paths:
                try:
                    os.remove(fp)
                except OSError:
                    pass
            try:
                os.rmdir(temp_dir); self.signals.progress.emit("Temp directory removed.")
            except OSError:
                self.signals.progress.emit(f"Could not remove temp dir: {temp_dir}. Remove manually.")


# --- Main Application Window ---
class GalamseyMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galamsey Monitor with Time-Lapse & Video Player")
        self.setGeometry(100, 100, 900, 850)  # Increased size for video player

        self.worker_thread = None;
        self.worker = None
        self.timelapse_worker_thread = None;
        self.timelapse_worker = None
        self.progress_dialog = None;
        self.project_id = 'galamsey-monitor'

        # Media Player components
        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)  # Link player to widget

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_video_position)
        self.position_slider.setEnabled(False)

        self.media_player.playbackStateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.video_position_changed)
        self.media_player.durationChanged.connect(self.video_duration_changed)
        self.media_player.errorOccurred.connect(self.handle_media_player_error)

        # --- Layouts ---
        main_layout = QVBoxLayout(self)
        top_h_layout = QHBoxLayout()  # For single analysis inputs and map preview

        # Single Analysis Section
        single_analysis_group = QGroupBox("Single Period Analysis")
        single_analysis_form_layout = QFormLayout()  # Use QFormLayout directly

        self.coord_input = QLineEdit("-1.795049, 6.335836, -1.745373, 6.365126")
        self.date1_start = QDateEdit(QDate(2020, 1, 1))
        self.date1_end = QDateEdit(QDate(2020, 12, 31))
        self.date2_start = QDateEdit(QDate(2024, 1, 1))
        self.date2_end = QDateEdit(QDate(2024, 12, 31))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(-1.0, 0.0);
        self.threshold_input.setSingleStep(0.05);
        self.threshold_input.setValue(-0.20)
        self.analyze_button = QPushButton("Analyze Single Period")
        for dt_edit in [self.date1_start, self.date1_end, self.date2_start, self.date2_end]:
            dt_edit.setCalendarPopup(True);
            dt_edit.setDisplayFormat("yyyy-MM-dd")

        single_analysis_form_layout.addRow(QLabel("AOI Coords:"), self.coord_input)
        single_analysis_form_layout.addRow(QLabel("Period 1 Start (Baseline):"), self.date1_start)
        single_analysis_form_layout.addRow(QLabel("Period 1 End (Baseline):"), self.date1_end)
        single_analysis_form_layout.addRow(QLabel("Period 2 Start (Comparison):"), self.date2_start)
        single_analysis_form_layout.addRow(QLabel("Period 2 End (Comparison):"), self.date2_end)
        single_analysis_form_layout.addRow(QLabel("NDVI Change Threshold:"), self.threshold_input)
        single_analysis_form_layout.addRow(self.analyze_button)
        single_analysis_group.setLayout(single_analysis_form_layout)
        top_h_layout.addWidget(single_analysis_group, 1)  # Add with stretch factor

        # Map Display (next to single analysis inputs)
        map_group = QGroupBox("Single Analysis Map Preview")
        map_v_layout = QVBoxLayout()
        self.map_label = QLabel("Map will appear here.")
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_label.setMinimumSize(400, 300)  # Adjust size
        self.map_label.setStyleSheet("QLabel { border: 1px solid gray; background-color: #f0f0f0; }")
        map_v_layout.addWidget(self.map_label)
        map_group.setLayout(map_v_layout)
        top_h_layout.addWidget(map_group, 1)  # Add with stretch factor
        main_layout.addLayout(top_h_layout)

        # Time-Lapse and Video Player Section
        timelapse_video_group = QGroupBox("Time-Lapse Video")
        timelapse_video_layout = QVBoxLayout()

        timelapse_controls_layout = QFormLayout()
        self.timelapse_start_year_input = QSpinBox();
        self.timelapse_start_year_input.setRange(2000, QDate.currentDate().year());
        self.timelapse_start_year_input.setValue(2021)
        self.timelapse_end_year_input = QSpinBox();
        self.timelapse_end_year_input.setRange(2000, QDate.currentDate().year() + 5);
        self.timelapse_end_year_input.setValue(2024)
        self.timelapse_fps_input = QSpinBox();
        self.timelapse_fps_input.setRange(1, 30);
        self.timelapse_fps_input.setValue(1)
        self.generate_timelapse_button = QPushButton("Generate & Load Time-Lapse Video")
        timelapse_controls_layout.addRow(QLabel("Time-Lapse Start Year:"), self.timelapse_start_year_input)
        timelapse_controls_layout.addRow(QLabel("Time-Lapse End Year:"), self.timelapse_end_year_input)
        timelapse_controls_layout.addRow(QLabel("Video FPS:"), self.timelapse_fps_input)
        timelapse_controls_layout.addRow(self.generate_timelapse_button)
        timelapse_video_layout.addLayout(timelapse_controls_layout)

        # Video Player
        video_player_layout = QHBoxLayout()  # For play button and slider
        video_player_layout.addWidget(self.play_button)
        video_player_layout.addWidget(self.position_slider)
        timelapse_video_layout.addWidget(self.video_widget)  # The video display
        self.video_widget.setMinimumHeight(300)  # Give video widget some space
        timelapse_video_layout.addLayout(video_player_layout)  # Add controls below video
        timelapse_video_group.setLayout(timelapse_video_layout)
        main_layout.addWidget(timelapse_video_group)

        # Status Log
        status_group = QGroupBox("Status Log")
        status_v_layout = QVBoxLayout();
        self.status_log = QTextEdit();
        self.status_log.setReadOnly(True);
        self.status_log.setFixedHeight(100)
        status_v_layout.addWidget(self.status_log);
        status_group.setLayout(status_v_layout);
        main_layout.addWidget(status_group)

        self.analyze_button.clicked.connect(self.run_single_analysis)
        self.generate_timelapse_button.clicked.connect(self.run_timelapse_generation)
        self.init_gee_check()

    # ... (init_gee_check, handle_gee_init_error, log_status, setup_progress_dialog as before) ...
    def init_gee_check(self):
        self.log_status("Attempting to initialize Google Earth Engine...")
        try:
            ee.Initialize(project=self.project_id); self.log_status("GEE initialized.")
        except Exception as e:
            self.handle_gee_init_error(e)

    def handle_gee_init_error(self, e):
        msg = (f"ERROR: GEE Init: {e}\n\nEnsure:\n1. 'earthengine authenticate' run.\n2. Internet access.\n"
               f"3. Project ID ('{self.project_id}') correct.\n4. Earth Engine API enabled in Google Cloud.")
        self.log_status(msg.replace("\n\n", "\n").replace("\n", "\nStatus: "));
        QMessageBox.critical(self, "GEE Init Error", msg)
        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)

    def log_status(self, message):
        self.status_log.append(message); QApplication.processEvents()

    def setup_progress_dialog(self, title="Processing..."):
        if self.progress_dialog: self.progress_dialog.close()
        self.progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal);
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False);
        self.progress_dialog.setMinimumDuration(0);
        self.progress_dialog.setValue(0)
        return self.progress_dialog

    # --- Single Analysis Methods ---
    def run_single_analysis(self):
        self.log_status("Starting single analysis...");
        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)
        self.map_label.setText("Processing...");
        self.map_label.setPixmap(QPixmap())
        try:
            coords_str = self.coord_input.text().strip().split(',');
            if len(coords_str) != 4: raise ValueError("Coords: 4 numbers.")
            aoi_coords = [float(c.strip()) for c in coords_str]
            if not (aoi_coords[0] < aoi_coords[2] and aoi_coords[1] < aoi_coords[3]): raise ValueError(
                "Coords order issue.")
            start1 = self.date1_start.date().toString("yyyy-MM-dd");
            end1 = self.date1_end.date().toString("yyyy-MM-dd")
            start2 = self.date2_start.date().toString("yyyy-MM-dd");
            end2 = self.date2_end.date().toString("yyyy-MM-dd")
            threshold = self.threshold_input.value()
            if self.date1_start.date() >= self.date1_end.date() or \
                    self.date2_start.date() >= self.date2_end.date() or \
                    self.date1_end.date() >= self.date2_start.date(): raise ValueError(
                "Date ranges invalid/overlapping.")
        except ValueError as ve:
            self.log_status(f"Input Error: {ve}");
            QMessageBox.warning(self, "Input Error", f"{ve}")
            self.analyze_button.setEnabled(True);
            self.generate_timelapse_button.setEnabled(True);
            return
        progress_dialog = self.setup_progress_dialog("Single Analysis...");
        progress_dialog.canceled.connect(self.cancel_single_analysis);
        progress_dialog.show()
        self.worker = GEEWorker(aoi_coords, start1, end1, start2, end2, threshold, project_id=self.project_id)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker.signals.finished.connect(self.on_single_analysis_complete);
        self.worker.signals.error.connect(self.on_single_analysis_error)
        self.worker.signals.progress.connect(self.update_progress_label_only);
        self.worker_thread.start()

    def update_progress_label_only(self, message):
        self.log_status(message)
        if self.progress_dialog and self.progress_dialog.isVisible():
            QMetaObject.invokeMethod(self.progress_dialog, "setLabelText", Qt.ConnectionType.QueuedConnection,
                                     Q_ARG(str, message))

    def on_single_analysis_complete(self, pil_image):
        self.log_status("Single analysis complete.")
        if self.progress_dialog: self.progress_dialog.close()
        if pil_image:
            try:
                img_byte_array = io.BytesIO();
                pil_image.save(img_byte_array, format='PNG');
                img_byte_array.seek(0)
                qimage = QImage.fromData(img_byte_array.read());
                if qimage.isNull(): raise ValueError("QImage is null")
                pixmap = QPixmap.fromImage(qimage)
                self.map_label.setPixmap(pixmap.scaled(self.map_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))
                self.log_status("Map preview displayed.")
            except Exception as e:
                self.log_status(f"Error display map: {e}"); self.map_label.setText("Error displaying map.")
        else:
            self.map_label.setText("No significant change or error.")
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def on_single_analysis_error(self, error_message):
        self.log_status(f"Single Analysis Error: {error_message}")
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.critical(self, "Single Analysis Error", error_message)
        self.map_label.setText("Analysis failed.");
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def cancel_single_analysis(self):
        self.log_status("Single analysis cancelled.")
        if self.worker: self.worker.is_cancelled = True
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)
        self.map_label.setText("Analysis Cancelled.");
        if self.progress_dialog: self.progress_dialog.close()

    # --- Time-Lapse Methods ---
    def run_timelapse_generation(self):
        self.log_status("Starting time-lapse...");
        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)
        try:
            coords_str = self.coord_input.text().strip().split(',');
            if len(coords_str) != 4: raise ValueError("Coords: 4 numbers.")
            aoi_coords = [float(c.strip()) for c in coords_str]
            if not (aoi_coords[0] < aoi_coords[2] and aoi_coords[1] < aoi_coords[3]): raise ValueError(
                "Coords order issue.")
            baseline_start = self.date1_start.date().toString("yyyy-MM-dd");
            baseline_end = self.date1_end.date().toString("yyyy-MM-dd")
            tl_start_year = self.timelapse_start_year_input.value();
            tl_end_year = self.timelapse_end_year_input.value()
            fps = self.timelapse_fps_input.value();
            threshold = self.threshold_input.value()
            if self.date1_start.date() >= self.date1_end.date(): raise ValueError("Baseline date error.")
            if tl_start_year > tl_end_year: raise ValueError("Timelapse year order error.")
            if QDate(tl_start_year, 1, 1) <= self.date1_end.date(): raise ValueError(
                "Timelapse start must be after baseline.")
            # Use a temporary file for the video
            temp_video_dir = tempfile.mkdtemp(prefix="galamsey_video_")
            self.temp_video_path = os.path.join(temp_video_dir, f"timelapse_{tl_start_year}-{tl_end_year}.mp4")
            self.log_status(f"Temporary video will be saved to: {self.temp_video_path}")

        except ValueError as ve:
            self.log_status(f"Input Error: {ve}");
            QMessageBox.warning(self, "Input Error", f"{ve}")
            self.analyze_button.setEnabled(True);
            self.generate_timelapse_button.setEnabled(True);
            return

        progress_dialog = self.setup_progress_dialog("Generating Time-Lapse...");
        progress_dialog.setRange(0, tl_end_year - tl_start_year + 1)
        progress_dialog.canceled.connect(self.cancel_timelapse_generation);
        progress_dialog.show()
        self.timelapse_worker = TimeLapseWorker(aoi_coords, baseline_start, baseline_end, tl_start_year, tl_end_year,
                                                threshold, project_id=self.project_id,
                                                output_video_path=self.temp_video_path, fps=fps)
        self.timelapse_worker_thread = threading.Thread(target=self.timelapse_worker.run, daemon=True)
        self.timelapse_worker.signals.finished.connect(self.on_timelapse_complete)
        self.timelapse_worker.signals.error.connect(self.on_timelapse_error)
        self.timelapse_worker.signals.progress.connect(self.update_progress_label_only)
        self.timelapse_worker.signals.frame_processed.connect(self.update_timelapse_progress_value)
        self.timelapse_worker_thread.start()

    def update_timelapse_progress_value(self, current_frame, total_frames):
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.setRange(0, total_frames);
            self.progress_dialog.setValue(current_frame)

    def on_timelapse_complete(self, video_path):
        self.log_status(f"Time-Lapse video generated: {video_path}")
        if self.progress_dialog: self.progress_dialog.close()
        self.media_player.setSource(QUrl.fromLocalFile(video_path))  # Load video
        self.play_button.setEnabled(True)
        self.position_slider.setEnabled(True)
        QMessageBox.information(self, "Time-Lapse Ready", f"Time-Lapse video is ready to play.\nSource: {video_path}")
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def on_timelapse_error(self, error_message):
        self.log_status(f"Time-Lapse Error: {error_message}")
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.critical(self, "Time-Lapse Error", error_message)
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def cancel_timelapse_generation(self):
        self.log_status("Time-Lapse cancelled.")
        if self.timelapse_worker: self.timelapse_worker.is_cancelled = True
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)
        if self.progress_dialog: self.progress_dialog.close()

    # --- Media Player Methods ---
    def play_video(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def media_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def video_position_changed(self, position):
        self.position_slider.setValue(position)

    def video_duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def set_video_position(self, position):
        self.media_player.setPosition(position)

    def handle_media_player_error(self):
        self.play_button.setEnabled(False)
        error_string = self.media_player.errorString()
        self.log_status(f"Media Player Error: {self.media_player.error()} - {error_string}")
        QMessageBox.critical(self, "Media Player Error", f"Error playing video: {error_string}")

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.is_alive(): self.cancel_single_analysis()
        if self.timelapse_worker_thread and self.timelapse_worker_thread.is_alive(): self.cancel_timelapse_generation()
        # Clean up temporary video if it exists and app is closing
        if hasattr(self, 'temp_video_path') and self.temp_video_path:
            temp_video_dir = os.path.dirname(self.temp_video_path)
            try:
                if os.path.exists(self.temp_video_path): os.remove(self.temp_video_path)
                if os.path.exists(temp_video_dir): os.rmdir(temp_video_dir)  # Only if empty
                self.log_status(f"Cleaned up temporary video: {self.temp_video_path}")
            except Exception as e:
                self.log_status(f"Error cleaning up temp video: {e}")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    monitor_app = GalamseyMonitorApp()
    monitor_app.show()
    sys.exit(app.exec())
