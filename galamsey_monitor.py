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
import cv2
import folium

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QDoubleSpinBox, QTextEdit, QFormLayout,
    QGroupBox, QProgressDialog, QMessageBox, QSpinBox, QFileDialog,
    QSlider, QStyle, QSplitter, QSizePolicy  # Added QSplitter, QSizePolicy
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
            pass

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
        self.output_video_path = output_video_path
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
                    pil_image = process_single_gee_frame(
                        self.aoi_rectangle_coords, self.baseline_start_date, self.baseline_end_date,
                        f"{year}-01-01", f"{year}-12-31", self.threshold,
                        self.thumb_size, self.project_id, self.signals.progress.emit,
                        output_type='image'
                    )
                    if pil_image:
                        if pil_image.mode == 'RGBA': pil_image = pil_image.convert('RGB')
                        frame_filename = os.path.join(temp_frames_dir, f"frame_{i:04d}.png")
                        pil_image.save(frame_filename);
                        frame_paths.append(frame_filename)
                        self.signals.progress.emit(f"Saved temporary frame for {year}: {frame_filename}")
                    else:
                        self.signals.progress.emit(f"Skipping frame for year {year} (no image data).")
                except Exception as frame_e:
                    self.signals.progress.emit(f"Error for year {year}: {frame_e}. Skipping.")

            if self.is_cancelled: self.signals.error.emit("Video compilation cancelled."); return
            if not frame_paths: self.signals.error.emit("No frames generated for video."); return
            self.signals.progress.emit("Compiling video...");
            video_output_dir = os.path.dirname(self.output_video_path)
            if video_output_dir: os.makedirs(video_output_dir, exist_ok=True)
            first_frame_img = cv2.imread(frame_paths[0])
            if first_frame_img is None: self.signals.error.emit(f"Could not read first frame: {frame_paths[0]}"); return
            height, width, _ = first_frame_img.shape;
            video_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v');
            out_video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, video_size)
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
            tb_str = traceback.format_exc();
            self.signals.error.emit(f"Time-lapse error: {e}\nTrace: {tb_str}")
        finally:
            self.signals.progress.emit("Cleaning up temp frames...")
            for fp in frame_paths:
                try:
                    os.remove(fp)
                except OSError:
                    pass
            try:
                if os.path.exists(temp_frames_dir): os.rmdir(temp_frames_dir); self.signals.progress.emit(
                    "Temp frames dir removed.")
            except OSError:
                self.signals.progress.emit(f"Could not remove temp frames dir: {temp_frames_dir}.")


# --- Main Application Window ---
class GalamseyMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galamsey Monitor with Interactive Map & Time-Lapse")

        # Start with a reasonably large default size, user can resize
        self.setGeometry(10, 40, 1300, 900)  # Increased initial size slightly

        self.worker_thread = None;
        self.worker = None
        self.timelapse_worker_thread = None;
        self.timelapse_worker = None
        self.progress_dialog = None
        self.project_id = 'galamsey-monitor'
        self.map_html_temp_dir = None

        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None
        self.final_video_path = None

        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)

        # Set an expanding size policy for the video widget, crucial for layout management
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

        # Main layout will hold the main vertical splitter
        main_layout = QVBoxLayout(self)

        # --- Top Section: Analysis Controls and Map (Horizontal Splitter) ---
        single_analysis_group = QGroupBox("Single Period Analysis")
        single_analysis_form_layout = QFormLayout()
        self.coord_input = QLineEdit("6.335836, -1.795049, 6.365126, -1.745373")
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
        # single_analysis_group.setMinimumWidth(350) # Optional: ensure controls don't get too squished

        map_group = QGroupBox("Interactive Map Preview (Red shows potential vegetation loss)")
        map_v_layout = QVBoxLayout()
        self.map_view = QWebEngineView()
        self.map_view.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        self.map_view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.map_view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.map_view.setHtml(
            "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;color:grey;'><p>Map will appear here after analysis.</p></body></html>")
        self.map_view.setMinimumSize(400, 300)  # Ensure map has a minimum usable size
        map_v_layout.addWidget(self.map_view)
        map_group.setLayout(map_v_layout)

        top_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_h_splitter.addWidget(single_analysis_group)
        top_h_splitter.addWidget(map_group)
        top_h_splitter.setSizes([380, 800])  # Initial proportional sizes for controls and map

        # --- Time-Lapse and Video Player Section (MODIFIED LAYOUT) ---
        timelapse_video_group = QGroupBox("Time-Lapse Video")
        # Main layout for this groupbox is now Horizontal
        main_timelapse_h_layout = QHBoxLayout()

        # -- Left Panel: Timelapse Controls --
        timelapse_controls_panel = QWidget()
        left_v_layout = QVBoxLayout(timelapse_controls_panel)
        left_v_layout.setContentsMargins(0, 5, 5, 5)  # Adjust margins as needed

        self.timelapse_start_year_input = QSpinBox()
        self.timelapse_start_year_input.setRange(2000, QDate.currentDate().year())
        self.timelapse_start_year_input.setValue(2021)
        self.timelapse_end_year_input = QSpinBox()
        self.timelapse_end_year_input.setRange(2000, QDate.currentDate().year() + 5)
        self.timelapse_end_year_input.setValue(QDate.currentDate().year())
        self.timelapse_fps_input = QSpinBox()
        self.timelapse_fps_input.setRange(1, 30)
        self.timelapse_fps_input.setValue(1)

        # Button is initialized here but added to left_v_layout later
        self.generate_timelapse_button = QPushButton("Generate & Load Time-Lapse Video")

        timelapse_form_inputs_layout = QFormLayout()  # Form for inputs only
        timelapse_form_inputs_layout.addRow(QLabel("Time-Lapse Start Year:"), self.timelapse_start_year_input)
        timelapse_form_inputs_layout.addRow(QLabel("Time-Lapse End Year:"), self.timelapse_end_year_input)
        timelapse_form_inputs_layout.addRow(QLabel("Video FPS:"), self.timelapse_fps_input)

        left_v_layout.addLayout(timelapse_form_inputs_layout)  # Add form to left panel
        left_v_layout.addWidget(self.generate_timelapse_button)  # Add button below form
        left_v_layout.addStretch(1)  # Push controls to the top

        timelapse_controls_panel.setMinimumWidth(220)  # Give controls panel a minimum width
        timelapse_controls_panel.setMaximumWidth(300)  # And a maximum width

        # -- Right Panel: Video Player --
        video_display_panel = QWidget()
        right_v_layout = QVBoxLayout(video_display_panel)
        right_v_layout.setContentsMargins(0, 0, 0, 0)

        # Video player controls (play, slider)
        video_player_controls_layout = QHBoxLayout()
        video_player_controls_layout.addWidget(self.play_button)
        video_player_controls_layout.addWidget(self.position_slider, 1)  # Slider takes available space
        video_player_controls_layout.addWidget(self.year_label_for_slider)

        right_v_layout.addWidget(self.video_widget, 1)  # Video widget takes most space, stretch factor 1
        right_v_layout.addLayout(video_player_controls_layout)  # Player controls below video

        self.video_widget.setMinimumHeight(200)  # Minimum height for video player itself

        # Add left and right panels to the main horizontal layout of the groupbox
        main_timelapse_h_layout.addWidget(timelapse_controls_panel)  # Default stretch factor 0
        main_timelapse_h_layout.addWidget(video_display_panel, 1)  # Stretch factor 1 (takes more space)

        timelapse_video_group.setLayout(main_timelapse_h_layout)

        # --- Status Log Section ---
        status_group = QGroupBox("Status Log")
        status_v_layout = QVBoxLayout();
        self.status_log = QTextEdit();
        self.status_log.setReadOnly(True);
        self.status_log.setMinimumHeight(80)  # Minimum height for the log
        # Allow log to expand vertically if space is available in its splitter pane
        self.status_log.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        status_v_layout.addWidget(self.status_log);
        status_group.setLayout(status_v_layout);

        # --- Main Vertical Splitter ---
        main_v_splitter = QSplitter(Qt.Orientation.Vertical)
        main_v_splitter.addWidget(top_h_splitter)
        main_v_splitter.addWidget(timelapse_video_group)
        main_v_splitter.addWidget(status_group)

        # Adjust initial sizes if needed, e.g., timelapse section might be shorter now
        main_v_splitter.setSizes([450, 280, 170])  # Example: more for top, bit less for video, more for log

        main_layout.addWidget(main_v_splitter)  # Add the main splitter to the window's layout

        self.analyze_button.clicked.connect(self.run_single_analysis)
        self.generate_timelapse_button.clicked.connect(self.run_timelapse_generation)
        self.init_gee_check()

    def init_gee_check(self):
        self.log_status("Attempting GEE init...");
        try:
            ee.Initialize(project=self.project_id);
            self.log_status(
                f"GEE init OK (Project: {self.project_id}).");
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(
                1).size().getInfo();
            self.log_status("GEE test OK.")
        except Exception as e:
            self.handle_gee_init_error(e)

    def handle_gee_init_error(self, e):
        msg = (f"ERROR GEE Init: {e}\n\nEnsure:\n1. 'earthengine authenticate' run.\n2. Internet.\n"
               f"3. Project ID ('{self.project_id}') correct & API enabled.\n4. Restart app after checks.")
        self.log_status(msg.replace("\n\n", "\n").replace("\n", "\nStatus: "));
        QMessageBox.critical(self, "GEE Init Error", msg)
        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)

    def log_status(self, message):
        self.status_log.append(message);
        QApplication.processEvents()

    def setup_progress_dialog(self, title="Processing..."):
        if self.progress_dialog and self.progress_dialog.isVisible(): self.progress_dialog.close()
        self.progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self);
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal);
        self.progress_dialog.setAutoClose(False);
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumDuration(0);
        self.progress_dialog.setValue(0);
        return self.progress_dialog

    def _parse_coordinates(self):
        coords_str_list = self.coord_input.text().strip().split(',')
        if len(coords_str_list) != 4: raise ValueError("Coords: 4 numbers (Lat1,Lon1,Lat2,Lon2) comma-separated.")
        try:
            raw_coords_float = [float(c.strip()) for c in coords_str_list]
        except ValueError:
            raise ValueError("Invalid coordinate number format.")
        lat1, lon1, lat2, lon2 = raw_coords_float
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90): raise ValueError("Latitudes must be -90 to 90.")
        if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180): raise ValueError("Longitudes must be -180 to 180.")
        aoi_ee_rect_coords = [min(lon1, lon2), min(lat1, lat2), max(lon1, lon2),
                              max(lat1, lat2)]
        return raw_coords_float, aoi_ee_rect_coords

    def run_single_analysis(self):
        self.log_status("Starting single period analysis (interactive map)...");
        self.analyze_button.setEnabled(False);
        self.generate_timelapse_button.setEnabled(False)
        self.map_view.setHtml(
            "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;'><h2>Processing analysis...</h2><p>Map will load shortly.</p></body></html>")
        try:
            _, aoi_ee_rect_coords = self._parse_coordinates()
            start1 = self.date1_start.date().toString("yyyy-MM-dd");
            end1 = self.date1_end.date().toString("yyyy-MM-dd")
            start2 = self.date2_start.date().toString("yyyy-MM-dd");
            end2 = self.date2_end.date().toString("yyyy-MM-dd")
            if self.date1_start.date() >= self.date1_end.date() or self.date2_start.date() >= self.date2_end.date() or self.date1_end.date() >= self.date2_start.date():
                raise ValueError("Date ranges invalid/overlapping.")
        except ValueError as ve:
            self.log_status(f"Input Error: {ve}");
            QMessageBox.warning(self, "Input Error", str(ve))
            self.map_view.setHtml(f"<html><body><h2>Input Error:</h2><pre>{ve}</pre></body></html>")
            self.analyze_button.setEnabled(True);
            self.generate_timelapse_button.setEnabled(True);
            return

        pd = self.setup_progress_dialog("Single Analysis in Progress...");
        pd.setRange(0, 0);
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
            QMetaObject.invokeMethod(self.progress_dialog, "setLabelText", Qt.ConnectionType.QueuedConnection,
                                     Q_ARG(str, message))

    def on_single_analysis_complete_map(self, analysis_data):
        self.log_status("Single analysis (map data) received.")
        if self.progress_dialog: self.progress_dialog.close()

        map_id_dict = analysis_data.get('map_id_dict')
        aoi_bounds_gee = analysis_data.get('aoi_bounds')

        if not map_id_dict or not aoi_bounds_gee:
            self.log_status("Error: MapID or AOI bounds missing from GEE result.")
            self.map_view.setHtml(
                "<html><body><h2>Error:</h2><p>Could not retrieve map data from GEE.</p></body></html>")
            self.analyze_button.setEnabled(True);
            self.generate_timelapse_button.setEnabled(True)
            return

        try:
            lons = [p[0] for p in aoi_bounds_gee]
            lats = [p[1] for p in aoi_bounds_gee]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2

            folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                attr='Google', name='Google Hybrid', overlay=False, control=True, show=True
            ).add_to(folium_map)
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                attr='Google', name='Google Roadmap', overlay=False, control=True
            ).add_to(folium_map)
            folium.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine Analysis',
                name='GEE Analysis Layer',
                overlay=True, control=True, show=True, max_native_zoom=18  # Adjusted max_native_zoom
            ).add_to(folium_map)
            folium.LayerControl().add_to(folium_map)

            if self.map_html_temp_dir:
                try:
                    self.map_html_temp_dir.cleanup()
                except Exception as e_clean:
                    self.log_status(f"Note: Error cleaning up previous map temp dir: {e_clean}")

            self.map_html_temp_dir = tempfile.TemporaryDirectory(prefix="galamsey_map_html_")
            map_html_path = os.path.join(self.map_html_temp_dir.name, "interactive_map.html")

            folium_map.save(map_html_path)
            self.log_status(f"Interactive map saved to: {map_html_path}")
            self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(map_html_path)))
            self.log_status("Interactive map loaded.")

        except Exception as e:
            self.log_status(f"Error creating/displaying Folium map: {e}")
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
        if self.progress_dialog: self.progress_dialog.close()
        self.map_view.setHtml(
            "<html><body style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;color:grey;'><p>Analysis Cancelled. Map will appear here.</p></body></html>")
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
            if self.date1_start.date() >= self.date1_end.date(): raise ValueError("Baseline date error.")
            if tl_start_year > tl_end_year: raise ValueError("Timelapse year order error.")
            if QDate(tl_start_year, 1, 1) <= self.date1_end.date(): raise ValueError("Timelapse start after baseline.")
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

    def update_timelapse_progress_value(self, cf, tf):
        if self.progress_dialog and self.progress_dialog.isVisible(): self.progress_dialog.setRange(0,
                                                                                                    tf); self.progress_dialog.setValue(
            cf)

    def on_timelapse_complete(self, video_path):
        self.log_status(f"Time-Lapse video generated: {video_path}");
        if self.progress_dialog: self.progress_dialog.close()
        self.active_timelapse_start_year = self.timelapse_start_year_input.value();
        self.active_timelapse_end_year = self.timelapse_end_year_input.value();
        self.active_timelapse_fps = self.timelapse_fps_input.value()
        self.media_player.setSource(QUrl.fromLocalFile(video_path));
        self.play_button.setEnabled(True);
        self.position_slider.setEnabled(True)
        self._update_year_label_for_slider(0);
        QMessageBox.information(self, "Time-Lapse Ready", f"Time-Lapse video ready.\nSaved: {video_path}")
        self.analyze_button.setEnabled(True);
        self.generate_timelapse_button.setEnabled(True)

    def on_timelapse_error(self, error_message):
        self.log_status(f"Time-Lapse Error: {error_message}");
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
            self.media_player.pause();
            self.log_status("Video paused.")
        else:
            self.media_player.play();
            self.log_status("Video playing.")

    def media_state_changed(self, state: QMediaPlayer.PlaybackState):
        self.play_button.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPause if state == QMediaPlayer.PlaybackState.PlayingState else QStyle.StandardPixmap.SP_MediaPlay))
        if state == QMediaPlayer.PlaybackState.StoppedState and self.active_timelapse_start_year is not None and self.media_player.position() == self.media_player.duration() and self.media_player.duration() > 0:
            self._update_year_label_for_slider(self.media_player.duration())

    def _update_year_label_for_slider(self, position_ms):
        if self.active_timelapse_start_year is not None and self.active_timelapse_end_year is not None and self.active_timelapse_fps is not None and self.active_timelapse_fps > 0:
            ms_per_frame = 1000.0 / self.active_timelapse_fps
            if ms_per_frame <= 0: self.year_label_for_slider.setText("Year: - (FPS Err)"); return
            cfi = int(position_ms / ms_per_frame);
            ntf = (self.active_timelapse_end_year - self.active_timelapse_start_year + 1)
            cfi = max(0, min(cfi, ntf - 1));
            self.year_label_for_slider.setText(f"Year: {self.active_timelapse_start_year + cfi}")
        else:
            self.year_label_for_slider.setText("Year: -")

    def video_position_changed(self, p):
        self.position_slider.setValue(p);
        self._update_year_label_for_slider(p)

    def video_duration_changed(self, d):
        self.position_slider.setRange(0, d)
        if d == 0: self.active_timelapse_start_year = None;self.active_timelapse_end_year = None;self.active_timelapse_fps = None;self._update_year_label_for_slider(
            0);self.play_button.setEnabled(False);self.position_slider.setEnabled(False)

    def set_video_position_from_slider(self, p):
        self.media_player.setPosition(p);
        self._update_year_label_for_slider(p)

    def handle_media_player_error(self):
        self.play_button.setEnabled(False);
        self.position_slider.setEnabled(False);
        err = self.media_player.errorString()
        self.log_status(f"Media Player Error: {err}");
        QMessageBox.critical(self, "Media Player Error", f"Error: {err}")
        self.active_timelapse_start_year = None;
        self.active_timelapse_end_year = None;
        self.active_timelapse_fps = None;
        self._update_year_label_for_slider(0)

    def closeEvent(self, event):
        self.log_status("Closing app...");
        if self.worker_thread and self.worker_thread.is_alive(): self.cancel_single_analysis()
        if self.timelapse_worker_thread and self.timelapse_worker_thread.is_alive(): self.cancel_timelapse_generation()
        self.media_player.stop()
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
    # It's good practice to apply a style for consistency, if desired
    # app.setStyle("Fusion") # Example: try "Fusion", "Windows", "macOS" (system dependent)
    monitor_app = GalamseyMonitorApp()
    monitor_app.show()
    sys.exit(app.exec())