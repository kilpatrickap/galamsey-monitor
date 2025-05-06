import sys
import io
import threading
import traceback
import os
import tempfile # For creating temporary directories
import glob # For finding image files

import ee
import requests
from PIL import Image
import cv2 # OpenCV for video creation

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QDoubleSpinBox, QTextEdit, QFormLayout,
    QGroupBox, QProgressDialog, QMessageBox, QSpinBox, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QDate, pyqtSignal, QObject, Qt, QMetaObject, Q_ARG

# --- Worker Signals (reused for both workers) ---
class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    # New signal for time-lapse, passing current frame and total frames
    frame_processed = pyqtSignal(int, int)


# --- Cloud Masking Function (SCL) ---
def mask_s2_clouds_scl(image):
    scl = image.select('SCL')
    unwanted_classes = [3, 8, 9, 10]
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])


# --- GEE Single Frame Processing Logic (Extracted for reusability) ---
def process_single_gee_frame(aoi_coords, period1_start, period1_end,
                             period2_start, period2_end, threshold_val,
                             thumb_size_val, project_id, progress_emitter=None):
    """
    Processes a single frame for GEE analysis.
    Returns a PIL.Image or None if an error occurs or no significant change.
    `progress_emitter` is a function like `worker.signals.progress.emit`.
    """
    if progress_emitter:
        progress_emitter(f"Initializing GEE for frame ({period2_start}-{period2_end})...")

    try:
        ee.Initialize(project=project_id)
    except Exception as init_e:
        if progress_emitter:
            progress_emitter(f"GEE Init failed for frame: {init_e}")
        raise init_e # Re-raise to be caught by caller

    aoi = ee.Geometry.Rectangle(aoi_coords)

    def calculate_ndvi(image):
        return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    rgb_viz_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3, 'gamma': 1.4}
    loss_viz_params = {'palette': ['red']}

    # Period 1 (Baseline)
    if progress_emitter: progress_emitter(f"Frame: Processing Baseline ({period1_start}-{period1_end})...")
    collection_p1_base = s2_sr.filterBounds(aoi).filterDate(period1_start, period1_end).map(mask_s2_clouds_scl)
    count1 = collection_p1_base.size().getInfo()
    if count1 == 0:
        if progress_emitter: progress_emitter(f"Frame: No cloud-free images for Baseline.")
        return None # Or raise an error specific to this
    median_ndvi_p1 = collection_p1_base.map(calculate_ndvi).select('NDVI').median()

    # Period 2 (Current period for this frame)
    if progress_emitter: progress_emitter(f"Frame: Processing Period 2 ({period2_start}-{period2_end})...")
    collection_p2_base = s2_sr.filterBounds(aoi).filterDate(period2_start, period2_end).map(mask_s2_clouds_scl)
    count2 = collection_p2_base.size().getInfo()
    if count2 == 0:
        if progress_emitter: progress_emitter(f"Frame: No cloud-free images for Period 2.")
        return None # Or raise an error
    median_ndvi_p2 = collection_p2_base.map(calculate_ndvi).select('NDVI').median()
    median_rgb_p2 = collection_p2_base.select(['B4', 'B3', 'B2']).median()

    # Change & Loss Mask
    if progress_emitter: progress_emitter("Frame: Calculating NDVI change...")
    ndvi_change = median_ndvi_p2.subtract(median_ndvi_p1).rename('NDVI_Change')
    loss_mask = ndvi_change.lt(threshold_val).selfMask()

    # Visual Layers
    if progress_emitter: progress_emitter("Frame: Creating visual layers...")
    background_layer = median_rgb_p2.visualize(**rgb_viz_params)
    loss_overlay = loss_mask.visualize(**loss_viz_params)
    final_image_viz = ee.ImageCollection([background_layer, loss_overlay]).mosaic().clip(aoi)

    # Thumbnail
    if progress_emitter: progress_emitter("Frame: Generating thumbnail...")
    thumb_url = None
    try:
        thumb_url = final_image_viz.getThumbURL({
            'region': aoi.getInfo()['coordinates'],
            'dimensions': thumb_size_val, 'format': 'png'
        })
    except ee.EEException as thumb_e:
        if "No valid pixels" in str(thumb_e) or "Image has no bands" in str(thumb_e):
            if progress_emitter: progress_emitter("Frame: No significant loss or vis issue. Generating background only.")
            try:
                thumb_url = background_layer.getThumbURL({
                    'region': aoi.getInfo()['coordinates'],
                    'dimensions': thumb_size_val, 'format': 'png'
                })
            except Exception as bg_thumb_e:
                if progress_emitter: progress_emitter(f"Frame: Error getting background thumbnail: {bg_thumb_e}")
                raise bg_thumb_e # Re-raise
        else:
            raise thumb_e # Re-raise other GEE exceptions

    if not thumb_url:
        if progress_emitter: progress_emitter("Frame: Failed to generate thumbnail URL.")
        return None # Indicate failure to get a URL

    # Fetch and process image
    if progress_emitter: progress_emitter("Frame: Downloading image...")
    response = requests.get(thumb_url)
    response.raise_for_status()
    img_data = response.content
    pil_image = Image.open(io.BytesIO(img_data))
    return pil_image


# --- GEE Single Analysis Worker (largely unchanged, but now uses the shared function) ---
class GEEWorker(QObject):
    def __init__(self, aoi_coords, start1, end1, start2, end2, threshold, thumb_size=512, project_id='galamsey-monitor'):
        super().__init__()
        self.signals = WorkerSignals()
        self.aoi_coords = aoi_coords
        self.start1 = start1
        self.end1 = end1
        self.start2 = start2
        self.end2 = end2
        self.threshold = threshold
        self.thumb_size = thumb_size
        self.project_id = project_id # Pass project ID
        self.is_cancelled = False # Cancellation not fully implemented in this simple GEE call

    def run(self):
        try:
            pil_image = process_single_gee_frame(
                self.aoi_coords, self.start1, self.end1, self.start2, self.end2,
                self.threshold, self.thumb_size, self.project_id,
                self.signals.progress.emit # Pass the progress emitter
            )
            self.signals.finished.emit(pil_image)
        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(f"Error in GEEWorker: {e}\nTrace: {tb_str}")


# --- Time-Lapse Generation Worker ---
class TimeLapseWorker(QObject):
    def __init__(self, aoi_coords, baseline_start_date, baseline_end_date,
                 timelapse_start_year, timelapse_end_year, threshold,
                 thumb_size=512, project_id='galamsey-monitor', output_video_path="galamsey_timelapse.mp4", fps=1):
        super().__init__()
        self.signals = WorkerSignals()
        self.aoi_coords = aoi_coords
        self.baseline_start_date = baseline_start_date
        self.baseline_end_date = baseline_end_date
        self.timelapse_start_year = timelapse_start_year
        self.timelapse_end_year = timelapse_end_year
        self.threshold = threshold
        self.thumb_size = thumb_size
        self.project_id = project_id
        self.output_video_path = output_video_path
        self.fps = fps
        self.is_cancelled = False

    def run(self):
        temp_dir = tempfile.mkdtemp(prefix="galamsey_frames_")
        self.signals.progress.emit(f"Temporary image frames will be saved in: {temp_dir}")
        frame_paths = []
        total_frames = self.timelapse_end_year - self.timelapse_start_year + 1

        try:
            for i, year in enumerate(range(self.timelapse_start_year, self.timelapse_end_year + 1)):
                if self.is_cancelled:
                    self.signals.progress.emit("Time-lapse generation cancelled.")
                    break

                self.signals.frame_processed.emit(i + 1, total_frames) # Update progress
                self.signals.progress.emit(f"Processing frame for year: {year} ({i+1}/{total_frames})")

                period2_start = f"{year}-01-01"
                period2_end = f"{year}-12-31"

                try:
                    pil_image = process_single_gee_frame(
                        self.aoi_coords, self.baseline_start_date, self.baseline_end_date,
                        period2_start, period2_end, self.threshold, self.thumb_size,
                        self.project_id, self.signals.progress.emit
                    )

                    if pil_image:
                        # Convert PIL RGBA (from PNG) to RGB for OpenCV if necessary
                        if pil_image.mode == 'RGBA':
                            pil_image = pil_image.convert('RGB')
                        frame_filename = os.path.join(temp_dir, f"frame_{i:04d}.png")
                        pil_image.save(frame_filename)
                        frame_paths.append(frame_filename)
                        self.signals.progress.emit(f"Saved frame for {year}: {frame_filename}")
                    else:
                        self.signals.progress.emit(f"Skipping frame for {year} (no image generated).")

                except Exception as frame_e:
                    self.signals.progress.emit(f"Error processing frame for year {year}: {frame_e}. Skipping.")
                    # Decide if you want to stop on error or continue
                    # continue

            if self.is_cancelled:
                self.signals.error.emit("Time-lapse cancelled before video compilation.")
                return

            if not frame_paths:
                self.signals.error.emit("No frames were generated. Cannot create video.")
                return

            self.signals.progress.emit("All frames processed. Compiling video...")
            # Determine video size from the first frame
            first_frame_img = cv2.imread(frame_paths[0])
            if first_frame_img is None:
                self.signals.error.emit(f"Could not read first frame: {frame_paths[0]}")
                return
            height, width, layers = first_frame_img.shape
            size = (width, height)

            # Define the codec and create VideoWriter object
            # Try common codecs if 'mp4v' fails on some systems (e.g., 'XVID')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For .mp4
            out_video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, size)

            for frame_path in sorted(frame_paths): # Ensure correct order
                img = cv2.imread(frame_path)
                if img is not None:
                    out_video.write(img)
                else:
                    self.signals.progress.emit(f"Warning: Could not read frame {frame_path} for video.")

            out_video.release()
            self.signals.progress.emit(f"Video compilation complete: {self.output_video_path}")
            self.signals.finished.emit(self.output_video_path)

        except Exception as e:
            tb_str = traceback.format_exc()
            self.signals.error.emit(f"Error during time-lapse generation: {e}\nTrace: {tb_str}")
        finally:
            # Clean up temporary frame images
            self.signals.progress.emit("Cleaning up temporary frames...")
            for fp in frame_paths:
                try:
                    os.remove(fp)
                except OSError:
                    pass # Ignore if already removed or other issue
            try:
                os.rmdir(temp_dir)
                self.signals.progress.emit("Temporary directory removed.")
            except OSError:
                self.signals.progress.emit(f"Could not remove temporary directory: {temp_dir}. Please remove manually.")


# --- Main Application Window ---
class GalamseyMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galamsey Monitor with Time-Lapse")
        self.setGeometry(100, 100, 850, 750) # Adjusted size

        self.worker_thread = None
        self.worker = None
        self.timelapse_worker_thread = None
        self.timelapse_worker = None
        self.progress_dialog = None
        self.project_id = 'galamsey-monitor' # Set your project ID here

        # --- Layouts ---
        main_layout = QVBoxLayout(self)

        # Single Analysis Section
        single_analysis_group = QGroupBox("Single Period Analysis")
        single_analysis_layout = QVBoxLayout()
        input_form_layout = QFormLayout()

        self.coord_input = QLineEdit("-1.933843, 6.242898, -1.794923, 6.307095") # Corrected example
        self.date1_start = QDateEdit(QDate(2017, 1, 1)) # Baseline year
        self.date1_end = QDateEdit(QDate(2017, 12, 31))
        self.date2_start = QDateEdit(QDate(2019, 1, 1)) # Example comparison year
        self.date2_end = QDateEdit(QDate(2019, 12, 31))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(-1.0, 0.0); self.threshold_input.setSingleStep(0.05); self.threshold_input.setValue(-0.20)
        self.analyze_button = QPushButton("Analyze Single Period")

        for dt_edit in [self.date1_start, self.date1_end, self.date2_start, self.date2_end]:
            dt_edit.setCalendarPopup(True)
            dt_edit.setDisplayFormat("yyyy-MM-dd")

        input_form_layout.addRow(QLabel("AOI Coordinates (lon_min, lat_min, lon_max, lat_max):"), self.coord_input)
        input_form_layout.addRow(QLabel("Period 1 Start (Baseline):"), self.date1_start)
        input_form_layout.addRow(QLabel("Period 1 End (Baseline):"), self.date1_end)
        input_form_layout.addRow(QLabel("Period 2 Start (Comparison):"), self.date2_start)
        input_form_layout.addRow(QLabel("Period 2 End (Comparison):"), self.date2_end)
        input_form_layout.addRow(QLabel("NDVI Change Threshold (Loss):"), self.threshold_input)
        input_form_layout.addRow(self.analyze_button)
        single_analysis_layout.addLayout(input_form_layout)

        # Map Display
        map_group = QGroupBox("Map Preview (Red shows potential vegetation loss)")
        map_v_layout = QVBoxLayout()
        self.map_label = QLabel("Map will appear here after single analysis.")
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_label.setMinimumSize(500, 300)
        self.map_label.setStyleSheet("QLabel { border: 1px solid gray; background-color: #f0f0f0; }")
        map_v_layout.addWidget(self.map_label)
        map_group.setLayout(map_v_layout)
        single_analysis_layout.addWidget(map_group)
        single_analysis_group.setLayout(single_analysis_layout)
        main_layout.addWidget(single_analysis_group)


        # Time-Lapse Section
        timelapse_group = QGroupBox("Time-Lapse Generation")
        timelapse_layout = QFormLayout()
        self.timelapse_start_year_input = QSpinBox()
        self.timelapse_start_year_input.setRange(2000, QDate.currentDate().year())
        self.timelapse_start_year_input.setValue(2018) # Example start for timelapse
        self.timelapse_end_year_input = QSpinBox()
        self.timelapse_end_year_input.setRange(2000, QDate.currentDate().year() + 5) # Allow a bit into future
        self.timelapse_end_year_input.setValue(2022) # Example end for timelapse
        self.timelapse_fps_input = QSpinBox()
        self.timelapse_fps_input.setRange(1, 30); self.timelapse_fps_input.setValue(1) # 1 frame (year) per second
        self.generate_timelapse_button = QPushButton("Generate Time-Lapse Video")

        timelapse_layout.addRow(QLabel("Time-Lapse Start Year:"), self.timelapse_start_year_input)
        timelapse_layout.addRow(QLabel("Time-Lapse End Year:"), self.timelapse_end_year_input)
        timelapse_layout.addRow(QLabel("Video FPS (Frames/Years per Second):"), self.timelapse_fps_input)
        timelapse_layout.addRow(self.generate_timelapse_button)
        timelapse_group.setLayout(timelapse_layout)
        main_layout.addWidget(timelapse_group)

        # Status Log
        status_group = QGroupBox("Status Log")
        status_v_layout = QVBoxLayout()
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True); self.status_log.setFixedHeight(150)
        status_v_layout.addWidget(self.status_log)
        status_group.setLayout(status_v_layout)
        main_layout.addWidget(status_group)

        # Connections
        self.analyze_button.clicked.connect(self.run_single_analysis)
        self.generate_timelapse_button.clicked.connect(self.run_timelapse_generation)

        self.init_gee_check()

    def init_gee_check(self):
        self.log_status("Attempting to initialize Google Earth Engine...")
        try:
            ee.Initialize(project=self.project_id)
            self.log_status("Google Earth Engine initialized successfully.")
        except Exception as e:
            self.handle_gee_init_error(e)

    def handle_gee_init_error(self, e):
        msg = (f"ERROR: Failed to initialize GEE: {e}\n\n"
               "Please ensure:\n"
               "1. You have run 'earthengine authenticate'.\n"
               "2. You have internet access.\n"
               f"3. The project ID ('{self.project_id}') is correct.\n"
               "4. The Earth Engine API is enabled for this project in Google Cloud Console.")
        self.log_status(msg.replace("\n\n", "\n").replace("\n", "\nStatus: ")) # Log more concisely
        QMessageBox.critical(self, "GEE Initialization Error", msg)
        self.analyze_button.setEnabled(False)
        self.generate_timelapse_button.setEnabled(False)


    def log_status(self, message):
        self.status_log.append(message)
        QApplication.processEvents()

    def setup_progress_dialog(self, title="Processing..."):
        if self.progress_dialog: # Close existing if any
            self.progress_dialog.close()
        self.progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self) # 0-100 for %
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0) # Start at 0%
        return self.progress_dialog


    def run_single_analysis(self):
        self.log_status("Starting single period analysis...")
        self.analyze_button.setEnabled(False)
        self.generate_timelapse_button.setEnabled(False) # Disable other button too
        self.map_label.setText("Processing... Please wait."); self.map_label.setPixmap(QPixmap())

        try:
            # Input validation (similar to your existing one)
            coords_str = self.coord_input.text().strip().split(',')
            if len(coords_str) != 4: raise ValueError("Coords: 4 numbers.")
            aoi_coords = [float(c.strip()) for c in coords_str]
            if not (aoi_coords[0] < aoi_coords[2] and aoi_coords[1] < aoi_coords[3]):
                 raise ValueError("Coords order: min_lon, min_lat, max_lon, max_lat.")

            start1 = self.date1_start.date().toString("yyyy-MM-dd")
            end1 = self.date1_end.date().toString("yyyy-MM-dd")
            start2 = self.date2_start.date().toString("yyyy-MM-dd")
            end2 = self.date2_end.date().toString("yyyy-MM-dd")
            threshold = self.threshold_input.value()
            # Date range validation
            if self.date1_start.date() >= self.date1_end.date() or \
               self.date2_start.date() >= self.date2_end.date() or \
               self.date1_end.date() >= self.date2_start.date():
                raise ValueError("Date ranges invalid/overlapping.")

        except ValueError as ve:
            self.log_status(f"Input Error: {ve}"); QMessageBox.warning(self, "Input Error", f"{ve}")
            self.analyze_button.setEnabled(True); self.generate_timelapse_button.setEnabled(True)
            return

        progress_dialog = self.setup_progress_dialog("Running Single Analysis...")
        progress_dialog.canceled.connect(self.cancel_single_analysis)
        progress_dialog.show()

        self.worker = GEEWorker(aoi_coords, start1, end1, start2, end2, threshold, project_id=self.project_id)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker.signals.finished.connect(self.on_single_analysis_complete)
        self.worker.signals.error.connect(self.on_single_analysis_error)
        self.worker.signals.progress.connect(self.update_progress_label_only) # No percentage for single
        self.worker_thread.start()

    def update_progress_label_only(self, message):
        """Updates only the label of the progress dialog."""
        self.log_status(message)
        if self.progress_dialog and self.progress_dialog.isVisible():
            QMetaObject.invokeMethod(self.progress_dialog, "setLabelText", Qt.ConnectionType.QueuedConnection, Q_ARG(str, message))


    def on_single_analysis_complete(self, pil_image):
        self.log_status("Single analysis thread finished.")
        if self.progress_dialog: self.progress_dialog.close()
        if pil_image:
            try:
                img_byte_array = io.BytesIO()
                pil_image.save(img_byte_array, format='PNG')
                img_byte_array.seek(0)
                qimage = QImage.fromData(img_byte_array.read())
                if qimage.isNull(): raise ValueError("QImage is null")
                pixmap = QPixmap.fromImage(qimage)
                scaled_pixmap = pixmap.scaled(self.map_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.map_label.setPixmap(scaled_pixmap)
                self.log_status("Single analysis map preview displayed.")
            except Exception as e:
                self.log_status(f"Error displaying single analysis map: {e}"); self.map_label.setText("Error displaying map.")
        else:
            self.map_label.setText("No significant change or error generating map.")
        self.analyze_button.setEnabled(True); self.generate_timelapse_button.setEnabled(True)

    def on_single_analysis_error(self, error_message):
        self.log_status(f"Single Analysis Error: {error_message}")
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.critical(self, "Single Analysis Error", error_message)
        self.map_label.setText("Single analysis failed. See Status Log.")
        self.analyze_button.setEnabled(True); self.generate_timelapse_button.setEnabled(True)

    def cancel_single_analysis(self):
        self.log_status("Single analysis cancelled by user.")
        if self.worker: self.worker.is_cancelled = True # Basic cancellation flag
        self.analyze_button.setEnabled(True); self.generate_timelapse_button.setEnabled(True)
        self.map_label.setText("Single analysis Cancelled.")
        if self.progress_dialog: self.progress_dialog.close()


    def run_timelapse_generation(self):
        self.log_status("Starting time-lapse generation...")
        self.analyze_button.setEnabled(False) # Disable other button too
        self.generate_timelapse_button.setEnabled(False)

        try: # Input validation
            coords_str = self.coord_input.text().strip().split(',')
            if len(coords_str) != 4: raise ValueError("Coords: 4 numbers.")
            aoi_coords = [float(c.strip()) for c in coords_str]
            if not (aoi_coords[0] < aoi_coords[2] and aoi_coords[1] < aoi_coords[3]):
                 raise ValueError("Coords order: min_lon, min_lat, max_lon, max_lat.")

            baseline_start = self.date1_start.date().toString("yyyy-MM-dd")
            baseline_end = self.date1_end.date().toString("yyyy-MM-dd")
            timelapse_start_year = self.timelapse_start_year_input.value()
            timelapse_end_year = self.timelapse_end_year_input.value()
            fps = self.timelapse_fps_input.value()
            threshold = self.threshold_input.value() # Use same threshold

            if self.date1_start.date() >= self.date1_end.date():
                raise ValueError("Baseline start date must be before baseline end date.")
            if timelapse_start_year > timelapse_end_year:
                raise ValueError("Time-lapse start year must be before or same as end year.")
            if QDate(timelapse_start_year,1,1) <= self.date1_end.date():
                raise ValueError("Time-lapse start year must be after baseline period ends.")

            # Ask user where to save the video
            default_video_name = f"galamsey_timelapse_{timelapse_start_year}-{timelapse_end_year}.mp4"
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Time-Lapse Video", default_video_name, "MP4 Video Files (*.mp4)")
            if not save_path:
                self.log_status("Time-lapse save cancelled by user.")
                self.analyze_button.setEnabled(True); self.generate_timelapse_button.setEnabled(True)
                return

        except ValueError as ve:
            self.log_status(f"Input Error: {ve}"); QMessageBox.warning(self, "Input Error", f"{ve}")
            self.analyze_button.setEnabled(True); self.generate_timelapse_button.setEnabled(True)
            return

        progress_dialog = self.setup_progress_dialog("Generating Time-Lapse Video...")
        progress_dialog.setRange(0, timelapse_end_year - timelapse_start_year + 1) # Set range for frames
        progress_dialog.canceled.connect(self.cancel_timelapse_generation)
        progress_dialog.show()

        self.timelapse_worker = TimeLapseWorker(
            aoi_coords, baseline_start, baseline_end,
            timelapse_start_year, timelapse_end_year, threshold,
            project_id=self.project_id, output_video_path=save_path, fps=fps
        )
        self.timelapse_worker_thread = threading.Thread(target=self.timelapse_worker.run, daemon=True)
        self.timelapse_worker.signals.finished.connect(self.on_timelapse_complete)
        self.timelapse_worker.signals.error.connect(self.on_timelapse_error)
        self.timelapse_worker.signals.progress.connect(self.update_progress_label_only) # Updates label
        self.timelapse_worker.signals.frame_processed.connect(self.update_timelapse_progress_value) # Updates value
        self.timelapse_worker_thread.start()

    def update_timelapse_progress_value(self, current_frame, total_frames):
        """Updates the value of the progress dialog for time-lapse."""
        if self.progress_dialog and self.progress_dialog.isVisible():
            # Progress dialog range is set to total_frames. Value is current_frame.
            self.progress_dialog.setRange(0, total_frames)
            self.progress_dialog.setValue(current_frame)


    def on_timelapse_complete(self, video_path):
        self.log_status(f"Time-Lapse generation complete! Video saved to: {video_path}")
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.information(self, "Time-Lapse Complete", f"Video saved to:\n{video_path}")
        self.analyze_button.setEnabled(True); self.generate_timelapse_button.setEnabled(True)

    def on_timelapse_error(self, error_message):
        self.log_status(f"Time-Lapse Error: {error_message}")
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.critical(self, "Time-Lapse Error", error_message)
        self.analyze_button.setEnabled(True); self.generate_timelapse_button.setEnabled(True)

    def cancel_timelapse_generation(self):
        self.log_status("Time-Lapse generation cancelled by user.")
        if self.timelapse_worker: self.timelapse_worker.is_cancelled = True
        self.analyze_button.setEnabled(True); self.generate_timelapse_button.setEnabled(True)
        if self.progress_dialog: self.progress_dialog.close()

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.is_alive():
             self.log_status("Window closing, attempting to cancel single analysis...")
             self.cancel_single_analysis()
        if self.timelapse_worker_thread and self.timelapse_worker_thread.is_alive():
             self.log_status("Window closing, attempting to cancel time-lapse...")
             self.cancel_timelapse_generation()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    monitor_app = GalamseyMonitorApp()
    monitor_app.show()
    sys.exit(app.exec())