import sys
import io
import threading
import traceback
import os
import tempfile  # Keep for temp frames, not for final video
import glob  # Keep for temp frames

import ee
import requests
from PIL import Image
import cv2  # OpenCV for video creation

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QDoubleSpinBox, QTextEdit, QFormLayout,
    QGroupBox, QProgressDialog, QMessageBox, QSpinBox, QFileDialog,
    QSlider, QStyle
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QDate, pyqtSignal, QObject, Qt, QMetaObject, Q_ARG, QUrl
from PyQt6.QtMultimedia import QMediaPlayer  # QMediaFormat not used
from PyQt6.QtMultimediaWidgets import QVideoWidget


# --- Worker Signals ---
class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    frame_processed = pyqtSignal(int, int)  # current_frame, total_frames


# --- Cloud Masking Function (SCL) ---
def mask_s2_clouds_scl(image):
    scl = image.select('SCL')
    # Mask to keep clear and vegetated pixels, remove clouds, shadows, cirrus, snow.
    # Classes to remove: 3 (Cloud Shadow), 8 (Cloud Medium Prob), 9 (Cloud High Prob), 10 (Thin Cirrus)
    # SCL classes: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])


# --- GEE Single Frame Processing Logic ---
def process_single_gee_frame(aoi_rectangle_coords, period1_start, period1_end,
                             period2_start, period2_end, threshold_val,
                             thumb_size_val, project_id, progress_emitter=None):
    # GEE should be initialized by the worker's run method or globally.
    # No ee.Initialize() here to avoid conflicts if already initialized.

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
        if progress_emitter: progress_emitter(f"Frame: No suitable cloud-free images found for Baseline period.")
        return None
    median_ndvi_p1 = collection_p1_base.map(calculate_ndvi).select('NDVI').median()

    if progress_emitter: progress_emitter(f"Frame: Processing Period 2 ({period2_start}-{period2_end})...")
    collection_p2_base = s2_sr.filterBounds(aoi).filterDate(period2_start, period2_end).map(mask_s2_clouds_scl)
    count2 = collection_p2_base.size().getInfo()
    if count2 == 0:
        if progress_emitter: progress_emitter(f"Frame: No suitable cloud-free images found for Period 2.")
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
        # Handle cases where loss_mask might be empty, resulting in no valid pixels for combined image
        if "No valid pixels" in str(thumb_e) or "Image.select: Pattern 'constant' did not match any bands." in str(
                thumb_e):
            if progress_emitter: progress_emitter(
                f"Frame: Thumbnail error ({thumb_e}). Likely no significant loss. Generating background only.")
            try:
                thumb_url = background_layer.getThumbURL(
                    {'region': aoi.getInfo()['coordinates'], 'dimensions': thumb_size_val, 'format': 'png'})
            except Exception as bg_thumb_e:
                if progress_emitter: progress_emitter(f"Frame: Error getting background thumbnail: {bg_thumb_e}")
                raise bg_thumb_e  # Re-raise if background also fails
        else:
            raise thumb_e  # Re-raise other GEE errors

    if not thumb_url:
        if progress_emitter: progress_emitter(
            "Frame: Failed to generate thumbnail URL (possibly no significant loss and background also failed or was empty).")
        return None

    if progress_emitter: progress_emitter("Frame: Downloading image...")
    response = requests.get(thumb_url, timeout=60)
    response.raise_for_status()
    img_data = response.content
    pil_image = Image.open(io.BytesIO(img_data))
    return pil_image


# --- GEE Single Analysis Worker ---
class GEEWorker(QObject):
    def __init__(self, aoi_rectangle_coords, start1, end1, start2, end2, threshold, thumb_size=512,
                 project_id='galamsey-monitor'):
        super().__init__()
        self.signals = WorkerSignals()
        self.aoi_rectangle_coords = aoi_rectangle_coords
        self.start1, self.end1 = start1, end1
        self.start2, self.end2 = start2, end2
        self.threshold, self.thumb_size = threshold, thumb_size
        self.project_id = project_id
        self.is_cancelled = False

    def run(self):
        try:
            # Ensure GEE is initialized for this thread's context if needed.
            ee.Initialize(project=self.project_id)
        except Exception:  # Handles if already initialized or other non-critical issues
            pass  # GEE often manages context well after first init.

        try:
            pil_image = process_single_gee_frame(self.aoi_rectangle_coords, self.start1, self.end1, self.start2,
                                                 self.end2,
                                                 self.threshold, self.thumb_size, self.project_id,
                                                 self.signals.progress.emit)
            if self.is_cancelled:  # Check before emitting finished
                self.signals.error.emit("Single analysis was cancelled during processing.")  # Or a different signal
                return
            self.signals.finished.emit(pil_image)
        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(f"Error in GEEWorker: {e}\nTrace: {tb_str}")


# --- Time-Lapse Generation Worker ---
class TimeLapseWorker(QObject):
    def __init__(self, aoi_rectangle_coords, baseline_start_date, baseline_end_date, timelapse_start_year,
                 timelapse_end_year, threshold, thumb_size=512, project_id='galamsey-monitor',
                 output_video_path="galamsey_timelapse.mp4", fps=1):
        super().__init__()
        self.signals = WorkerSignals()
        self.aoi_rectangle_coords = aoi_rectangle_coords
        self.baseline_start_date, self.baseline_end_date = baseline_start_date, baseline_end_date
        self.timelapse_start_year, self.timelapse_end_year = timelapse_start_year, timelapse_end_year
        self.threshold, self.thumb_size = threshold, thumb_size
        self.project_id = project_id
        self.output_video_path = output_video_path  # Final video path
        self.fps = fps
        self.is_cancelled = False

    def run(self):
        try:
            ee.Initialize(project=self.project_id)
        except Exception:
            pass

        temp_frames_dir = tempfile.mkdtemp(prefix="galamsey_frames_")
        self.signals.progress.emit(f"Temporary frames will be stored in: {temp_frames_dir}")
        frame_paths = []
        total_frames_to_generate = self.timelapse_end_year - self.timelapse_start_year + 1

        try:
            for i, year in enumerate(range(self.timelapse_start_year, self.timelapse_end_year + 1)):
                if self.is_cancelled:
                    self.signals.progress.emit("Time-lapse generation cancelled by user.")
                    break
                self.signals.frame_processed.emit(i + 1, total_frames_to_generate)
                self.signals.progress.emit(f"Processing frame for year: {year} ({i + 1}/{total_frames_to_generate})")

                try:
                    period2_start_str = f"{year}-01-01"
                    period2_end_str = f"{year}-12-31"

                    pil_image = process_single_gee_frame(
                        self.aoi_rectangle_coords,
                        self.baseline_start_date, self.baseline_end_date,
                        period2_start_str, period2_end_str,
                        self.threshold, self.thumb_size, self.project_id,
                        self.signals.progress.emit
                    )

                    if pil_image:
                        if pil_image.mode == 'RGBA': pil_image = pil_image.convert('RGB')
                        frame_filename = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
                        pil_image.save(frame_filename)
                        frame_paths.append(frame_filename)
                        self.signals.progress.emit(f"Saved temporary frame for {year}: {frame_filename}")
                    else:
                        self.signals.progress.emit(f"Skipping frame for year {year} (no image data returned).")

                except Exception as frame_e:
                    self.signals.progress.emit(
                        f"Error processing frame for year {year}: {frame_e}. Skipping this year.")

            if self.is_cancelled:
                self.signals.error.emit("Video compilation cancelled before completion.")
                return

            if not frame_paths:
                self.signals.error.emit("No frames were generated. Cannot create video.")
                return

            self.signals.progress.emit("Compiling video from generated frames...")

            video_output_dir = os.path.dirname(self.output_video_path)
            if video_output_dir:  # Ensure directory exists if it's not the CWD
                os.makedirs(video_output_dir, exist_ok=True)

            first_frame_img = cv2.imread(frame_paths[0])
            if first_frame_img is None:
                self.signals.error.emit(f"Could not read the first frame: {frame_paths[0]}. Video compilation failed.")
                return

            height, width, _ = first_frame_img.shape
            video_size = (width, height)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, video_size)

            for frame_path in sorted(frame_paths):
                img = cv2.imread(frame_path)
                if img is not None:
                    out_video.write(img)
                else:
                    self.signals.progress.emit(f"Warning: Could not read frame {frame_path}. Skipping.")

            out_video.release()
            self.signals.progress.emit(f"Video compilation complete. Saved to: {self.output_video_path}")
            self.signals.finished.emit(self.output_video_path)

        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(f"An error occurred during time-lapse generation: {e}\nTrace: {tb_str}")
        finally:
            self.signals.progress.emit("Cleaning up temporary frame files...")
            for fp in frame_paths:
                try:
                    os.remove(fp)
                except OSError:
                    pass
            try:
                if os.path.exists(temp_frames_dir):
                    os.rmdir(temp_frames_dir)
                    self.signals.progress.emit("Temporary frames directory removed.")
            except OSError:
                self.signals.progress.emit(
                    f"Could not remove temporary frames directory: {temp_frames_dir}. Please remove it manually.")


# --- Main Application Window ---
class GalamseyMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galamsey Monitor with Time-Lapse & Video Player")
        self.setGeometry(100, 100, 900, 900)

        self.worker_thread = None;
        self.worker = None
        self.timelapse_worker_thread = None;
        self.timelapse_worker = None
        self.progress_dialog = None
        self.project_id = 'galamsey-monitor'  # !!! REPLACE WITH YOUR GEE PROJECT ID !!!

        self.active_timelapse_start_year = None
        self.active_timelapse_end_year = None
        self.active_timelapse_fps = None
        self.final_video_path = None  # Stores path of last generated video

        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_video_position_from_slider)
        self.position_slider.setEnabled(False)

        self.year_label_for_slider = QLabel("Year: -")

        self.media_player.playbackStateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.video_position_changed)
        self.media_player.durationChanged.connect(self.video_duration_changed)
        self.media_player.errorOccurred.connect(self.handle_media_player_error)

        main_layout = QVBoxLayout(self)
        top_h_layout = QHBoxLayout()

        single_analysis_group = QGroupBox("Single Period Analysis")
        single_analysis_form_layout = QFormLayout()
        self.coord_input = QLineEdit("-1.795049, 6.335836, -1.745373, 6.365126")
        self.coord_input.setToolTip("Enter coordinates as: Lon1, Lat1, Lon2, Lat2 (e.g., -1.8, 6.3, -1.7, 6.4)")
        self.date1_start = QDateEdit(QDate(2020, 1, 1));
        self.date1_end = QDateEdit(QDate(2020, 12, 31))
        self.date2_start = QDateEdit(QDate(2024, 1, 1));
        self.date2_end = QDateEdit(QDate(QDate.currentDate()))  # Default end date to current
        self.threshold_input = QDoubleSpinBox();
        self.threshold_input.setRange(-1.0, 0.0);
        self.threshold_input.setSingleStep(0.05);
        self.threshold_input.setValue(-0.20)
        self.analyze_button = QPushButton("Analyze Single Period")
        for dt_edit in [self.date1_start, self.date1_end, self.date2_start, self.date2_end]:
            dt_edit.setCalendarPopup(True);
            dt_edit.setDisplayFormat("dd-MM-yyyy")
        single_analysis_form_layout.addRow(QLabel("AOI Coords (Lon1, Lat1, Lon2, Lat2):"), self.coord_input)
        single_analysis_form_layout.addRow(QLabel("Period 1 Start (Baseline):"), self.date1_start)
        single_analysis_form_layout.addRow(QLabel("Period 1 End (Baseline):"), self.date1_end)
        single_analysis_form_layout.addRow(QLabel("Period 2 Start (Comparison):"), self.date2_start)
        single_analysis_form_layout.addRow(QLabel("Period 2 End (Comparison):"), self.date2_end)
        single_analysis_form_layout.addRow(QLabel("NDVI Change Threshold:"), self.threshold_input)
        single_analysis_form_layout.addRow(self.analyze_button)
        single_analysis_group.setLayout(single_analysis_form_layout)
        top_h_layout.addWidget(single_analysis_group, 1)

        map_group = QGroupBox("Single Analysis Map Preview")
        map_v_layout = QVBoxLayout()
        self.map_label = QLabel("Map will appear here.");
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_label.setMinimumSize(400, 300);
        self.map_label.setStyleSheet("QLabel { border: 1px solid gray; background-color: #f0f0f0; }")
        map_v_layout.addWidget(self.map_label);
        map_group.setLayout(map_v_layout)
        top_h_layout.addWidget(map_group, 1);
        main_layout.addLayout(top_h_layout)

        timelapse_video_group = QGroupBox("Time-Lapse Video")
        timelapse_video_layout = QVBoxLayout()
        timelapse_controls_layout = QFormLayout()
        self.timelapse_start_year_input = QSpinBox();
        self.timelapse_start_year_input.setRange(2000, QDate.currentDate().year());
        self.timelapse_start_year_input.setValue(2021)
        self.timelapse_end_year_input = QSpinBox();
        self.timelapse_end_year_input.setRange(2000, QDate.currentDate().year() + 5);
        self.timelapse_end_year_input.setValue(QDate.currentDate().year())
        self.timelapse_fps_input = QSpinBox();
        self.timelapse_fps_input.setRange(1, 30);
        self.timelapse_fps_input.setValue(2)
        self.generate_timelapse_button = QPushButton("Generate & Load Time-Lapse Video")
        timelapse_controls_layout.addRow(QLabel("Time-Lapse Start Year:"), self.timelapse_start_year_input)
        timelapse_controls_layout.addRow(QLabel("Time-Lapse End Year:"), self.timelapse_end_year_input)
        timelapse_controls_layout.addRow(QLabel("Video FPS:"), self.timelapse_fps_input)
        timelapse_controls_layout.addRow(self.generate_timelapse_button)
        timelapse_video_layout.addLayout(timelapse_controls_layout)
        video_player_controls_layout = QHBoxLayout();
        video_player_controls_layout.addWidget(self.play_button)
        video_player_controls_layout.addWidget(self.position_slider, 1);
        video_player_controls_layout.addWidget(self.year_label_for_slider)
        timelapse_video_layout.addWidget(self.video_widget);
        self.video_widget.setMinimumHeight(300)
        timelapse_video_layout.addLayout(video_player_controls_layout)
        timelapse_video_group.setLayout(timelapse_video_layout);
        main_layout.addWidget(timelapse_video_group)

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

    def init_gee_check(self):
        self.log_status("Attempting to initialize Google Earth Engine...")
        try:
            ee.Initialize(project=self.project_id)
            self.log_status(f"GEE initialized with project: {self.project_id}.")
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1).size().getInfo()  # Test command
            self.log_status("GEE test command successful.")
        except Exception as e:
            self.handle_gee_init_error(e)

    def handle_gee_init_error(self, e):
        msg = (f"ERROR: GEE Initialization or Test Failed: {e}\n\n"
               "Ensure:\n1. 'earthengine authenticate' has been run.\n2. Internet connection is stable.\n"
               f"3. GEE Project ID ('{self.project_id}') is correct and active.\n4. Earth Engine API is enabled in Google Cloud Project.\n"
               "5. Restart the application after checking the above.")
        self.log_status(msg.replace("\n\n", "\n").replace("\n", "\nStatus: "));
        QMessageBox.critical(self, "GEE Init Error", msg)
        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)

    def log_status(self, message):
        self.status_log.append(message); QApplication.processEvents()

    def setup_progress_dialog(self, title="Processing..."):
        if self.progress_dialog and self.progress_dialog.isVisible(): self.progress_dialog.close()
        self.progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal);
        self.progress_dialog.setAutoClose(False);
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumDuration(0);
        self.progress_dialog.setValue(0)
        return self.progress_dialog

    def _parse_coordinates(self):
        coords_str_list = self.coord_input.text().strip().split(',')
        if len(coords_str_list) != 4: raise ValueError("Coords: 4 numbers (Lon1,Lat1,Lon2,Lat2) separated by commas.")
        try:
            raw_coords_float = [float(c.strip()) for c in coords_str_list]
        except ValueError:
            raise ValueError("Invalid coordinate number format.")
        lon1, lat1, lon2, lat2 = raw_coords_float
        if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180): raise ValueError(
            "Longitudes must be between -180 and 180.")
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90): raise ValueError("Latitudes must be between -90 and 90.")
        aoi_ee_rect_coords = [min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)]
        return raw_coords_float, aoi_ee_rect_coords

    def run_single_analysis(self):
        self.log_status("Starting single period analysis...");
        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)
        self.map_label.setText("Processing...");
        self.map_label.setPixmap(QPixmap())
        try:
            _, aoi_ee_rect_coords = self._parse_coordinates()  # raw_coords not needed here
            start1 = self.date1_start.date().toString("yyyy-MM-dd");
            end1 = self.date1_end.date().toString("yyyy-MM-dd")
            start2 = self.date2_start.date().toString("yyyy-MM-dd");
            end2 = self.date2_end.date().toString("yyyy-MM-dd")
            if self.date1_start.date() >= self.date1_end.date() or self.date2_start.date() >= self.date2_end.date() or self.date1_end.date() >= self.date2_start.date():
                raise ValueError("Date ranges invalid/overlapping.")
        except ValueError as ve:
            self.log_status(f"Input Error: {ve}");
            QMessageBox.warning(self, "Input Error", str(ve))
            self.analyze_button.setEnabled(True);
            self.generate_timelapse_button.setEnabled(True);
            return

        pd = self.setup_progress_dialog("Single Analysis in Progress...");
        pd.setRange(0, 0);  # Indeterminate
        pd.canceled.connect(self.cancel_single_analysis);
        pd.show()
        self.worker = GEEWorker(aoi_ee_rect_coords, start1, end1, start2, end2, self.threshold_input.value(),
                                project_id=self.project_id)
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
        self.log_status("Single analysis processing completed.")
        if self.progress_dialog: self.progress_dialog.close()
        if pil_image:
            try:
                qimage = QImage(pil_image.tobytes("raw", "RGB"), pil_image.width, pil_image.height,
                                QImage.Format.Format_RGB888)
                if qimage.isNull(): raise ValueError("Failed to create QImage.")
                self.map_label.setPixmap(
                    QPixmap.fromImage(qimage).scaled(self.map_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))
                self.log_status("Map preview updated.")
            except Exception as e:
                self.log_status(f"Error display map: {e}"); self.map_label.setText("Error display map.")
        else:
            self.map_label.setText("No significant change or no data."); self.log_status("Single analysis: No image.")
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def on_single_analysis_error(self, error_message):
        self.log_status(f"Single Analysis Error: {error_message}")
        if self.progress_dialog: self.progress_dialog.close(); QMessageBox.critical(self, "Single Analysis Error",
                                                                                    error_message)
        self.map_label.setText("Analysis failed.");
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def cancel_single_analysis(self):
        self.log_status("Attempting to cancel single analysis...")
        if self.worker: self.worker.is_cancelled = True
        if self.progress_dialog: self.progress_dialog.close()
        self.map_label.setText("Analysis Cancelled.");
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def run_timelapse_generation(self):
        self.log_status("Starting time-lapse...");
        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)
        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None;
        self._update_year_label_for_slider(0)
        try:
            raw_coords_input, aoi_ee_rect_coords = self._parse_coordinates()
            baseline_start = self.date1_start.date().toString("yyyy-MM-dd");
            baseline_end = self.date1_end.date().toString("yyyy-MM-dd")
            tl_start_year = self.timelapse_start_year_input.value();
            tl_end_year = self.timelapse_end_year_input.value()
            if self.date1_start.date() >= self.date1_end.date(): raise ValueError("Baseline date range error.")
            if tl_start_year > tl_end_year: raise ValueError("Timelapse year order error.")
            if QDate(tl_start_year, 1, 1) <= self.date1_end.date(): raise ValueError(
                "Timelapse start must be after baseline end.")

            coord_fn_parts = [str(c).replace('.', 'p').replace('-', 'm') for c in raw_coords_input]
            video_filename = f"{'_'.join(coord_fn_parts)}_{tl_start_year}-{tl_end_year}_timelapse.mp4"
            videos_dir = os.path.join(os.getcwd(), "videos");
            os.makedirs(videos_dir, exist_ok=True)
            self.final_video_path = os.path.join(videos_dir, video_filename)
            self.log_status(f"Video will be saved to: {self.final_video_path}")
        except ValueError as ve:
            self.log_status(f"Input Error TL: {ve}");
            QMessageBox.warning(self, "Input Error", str(ve))
            self.analyze_button.setEnabled(True);
            self.generate_timelapse_button.setEnabled(True);
            return

        pd = self.setup_progress_dialog("Generating Time-Lapse Video...");
        pd.setRange(0, tl_end_year - tl_start_year + 1)
        pd.canceled.connect(self.cancel_timelapse_generation);
        pd.show()
        self.timelapse_worker = TimeLapseWorker(aoi_ee_rect_coords, baseline_start, baseline_end, tl_start_year,
                                                tl_end_year,
                                                self.threshold_input.value(), project_id=self.project_id,
                                                output_video_path=self.final_video_path,
                                                fps=self.timelapse_fps_input.value())
        self.timelapse_worker_thread = threading.Thread(target=self.timelapse_worker.run, daemon=True)
        self.timelapse_worker.signals.finished.connect(self.on_timelapse_complete);
        self.timelapse_worker.signals.error.connect(self.on_timelapse_error)
        self.timelapse_worker.signals.progress.connect(self.update_progress_label_only);
        self.timelapse_worker.signals.frame_processed.connect(self.update_timelapse_progress_value)
        self.timelapse_worker_thread.start()

    def update_timelapse_progress_value(self, current_frame, total_frames):
        if self.progress_dialog and self.progress_dialog.isVisible(): self.progress_dialog.setValue(current_frame)

    def on_timelapse_complete(self, video_path):
        self.log_status(f"Time-Lapse video generated: {video_path}")
        if self.progress_dialog: self.progress_dialog.close()
        self.active_timelapse_start_year = self.timelapse_start_year_input.value()
        self.active_timelapse_end_year = self.timelapse_end_year_input.value()
        self.active_timelapse_fps = self.timelapse_fps_input.value()
        self.media_player.setSource(QUrl.fromLocalFile(video_path));
        self.play_button.setEnabled(True);
        self.position_slider.setEnabled(True)
        self._update_year_label_for_slider(0)
        QMessageBox.information(self, "Time-Lapse Ready", f"Time-Lapse video ready.\nSaved at: {video_path}")
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def on_timelapse_error(self, error_message):
        self.log_status(f"Time-Lapse Error: {error_message}")
        if self.progress_dialog: self.progress_dialog.close(); QMessageBox.critical(self, "Time-Lapse Error",
                                                                                    error_message)
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)
        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None;
        self._update_year_label_for_slider(0)

    def cancel_timelapse_generation(self):
        self.log_status("Cancelling time-lapse...");
        if self.timelapse_worker: self.timelapse_worker.is_cancelled = True
        if self.progress_dialog: self.progress_dialog.close();
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)
        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None;
        self._update_year_label_for_slider(0)

    def play_video(self):
        if self.media_player.source().isEmpty(): self.log_status("No video loaded."); return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause(); self.log_status("Video paused.")
        else:
            self.media_player.play(); self.log_status("Video playing.")

    def media_state_changed(self, state: QMediaPlayer.PlaybackState):
        self.play_button.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPause if state == QMediaPlayer.PlaybackState.PlayingState else QStyle.StandardPixmap.SP_MediaPlay))
        if state == QMediaPlayer.PlaybackState.StoppedState and self.active_timelapse_start_year is not None and self.media_player.position() == self.media_player.duration() and self.media_player.duration() > 0:
            self._update_year_label_for_slider(self.media_player.duration())

    def _update_year_label_for_slider(self, position_ms):
        if self.active_timelapse_start_year is not None and self.active_timelapse_end_year is not None and self.active_timelapse_fps is not None and self.active_timelapse_fps > 0:
            ms_per_frame = 1000.0 / self.active_timelapse_fps
            if ms_per_frame <= 0: self.year_label_for_slider.setText("Year: - (FPS Error)"); return
            current_frame_index = int(position_ms / ms_per_frame)
            num_total_frames = (self.active_timelapse_end_year - self.active_timelapse_start_year + 1)
            current_frame_index = max(0, min(current_frame_index, num_total_frames - 1))
            self.year_label_for_slider.setText(f"Year: {self.active_timelapse_start_year + current_frame_index}")
        else:
            self.year_label_for_slider.setText("Year: -")

    def video_position_changed(self, position_ms):
        self.position_slider.setValue(position_ms); self._update_year_label_for_slider(position_ms)

    def video_duration_changed(self, duration_ms):
        self.position_slider.setRange(0, duration_ms)
        if duration_ms == 0:
            self.active_timelapse_start_year = None;
            self.active_timelapse_end_year = None;
            self.active_timelapse_fps = None
            self._update_year_label_for_slider(0);
            self.play_button.setEnabled(False);
            self.position_slider.setEnabled(False)

    def set_video_position_from_slider(self, position_ms):
        self.media_player.setPosition(position_ms); self._update_year_label_for_slider(position_ms)

    def handle_media_player_error(self):
        self.play_button.setEnabled(False);
        self.position_slider.setEnabled(False)
        err_str = self.media_player.errorString();
        self.log_status(f"Media Player Error: {err_str}")
        QMessageBox.critical(self, "Media Player Error", f"Error playing video: {err_str}")
        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None;
        self._update_year_label_for_slider(0)

    def closeEvent(self, event):
        self.log_status("Closing app...");
        if self.worker_thread and self.worker_thread.is_alive(): self.cancel_single_analysis()
        if self.timelapse_worker_thread and self.timelapse_worker_thread.is_alive(): self.cancel_timelapse_generation()
        self.media_player.stop();
        self.log_status("Cleanup complete. Exiting.");
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("GalamseyMonitorApp")
    monitor_app = GalamseyMonitorApp()
    monitor_app.show()
    sys.exit(app.exec())