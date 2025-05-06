import sys
import io
import threading # To run GEE tasks in the background
import traceback # For detailed error logging

import ee
import requests # To fetch the image from the URL
from PIL import Image # To process the image data

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QDoubleSpinBox, QTextEdit, QFormLayout,
    QGroupBox, QProgressDialog, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QDate, pyqtSignal, QObject, Qt, QMetaObject, Q_ARG


# --- Worker Object Signals ---
# Necessary to emit signals from a non-GUI thread back to the GUI thread
class WorkerSignals(QObject):
    finished = pyqtSignal(object) # Pass the result (PIL image or None)
    error = pyqtSignal(str)       # Pass error message
    progress = pyqtSignal(str)    # Pass status updates

# --- Cloud Masking Function (using SCL - NEEDS TO BE DEFINED GLOBALLY) ---
# Define this function *before* the GEEWorker class
def mask_s2_clouds_scl(image):
    """Masks clouds in a Sentinel-2 SR image using the SCL band."""
    scl = image.select('SCL')
    # Define SCL values to mask (treat as unusable/cloudy)
    # 3: Cloud Shadow, 8: Cloud Medium Prob, 9: Cloud High Prob, 10: Cirrus
    unwanted_classes = [3, 8, 9, 10]
    # Create a mask where non-cloudy pixels are 1, others are 0.
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    # Apply mask and scale reflectance values (common for SR products)
    return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])
# -------------------------------------------------------------------------


# --- GEE Processing Worker ---
class GEEWorker(QObject):
    """Handles Google Earth Engine tasks in a separate thread."""
    def __init__(self, aoi_coords, start1, end1, start2, end2, threshold, thumb_size=512):
        super().__init__()
        self.signals = WorkerSignals()
        self.aoi_coords = aoi_coords
        self.start1 = start1
        self.end1 = end1
        self.start2 = start2
        self.end2 = end2
        self.threshold = threshold
        self.thumb_size = thumb_size
        self.is_cancelled = False

    def run(self):
        """Executes the GEE analysis."""
        try:
            self.signals.progress.emit("Initializing Earth Engine...")
            try:
                # Initialize EE within the thread. Replace with your Project ID.
                ee.Initialize(project='galamsey-monitor')
            except Exception as init_e:
                self.signals.error.emit(f"GEE Initialization failed: {init_e}")
                return

            self.signals.progress.emit("Defining AOI...")
            if self.is_cancelled: return
            aoi = ee.Geometry.Rectangle(self.aoi_coords)

            # --- Define GEE Calculation Functions ---
            def calculate_ndvi(image):
                """Calculates NDVI for a Sentinel-2 image."""
                return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

            # Reference the globally defined SCL cloud mask function
            # mask_s2_clouds_scl is defined outside this class

            # Use the recommended Harmonized Sentinel-2 Surface Reflectance collection
            s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

            # --- Define Visualization Parameters ---
            # For the background satellite image (True Color)
            rgb_viz_params = {
                'bands': ['B4', 'B3', 'B2'], # Red, Green, Blue bands
                'min': 0.0,                 # Minimum reflectance
                'max': 0.3,                 # Max reflectance (adjust for brightness)
                'gamma': 1.4                # Adjust contrast if needed
            }
            # For the loss overlay (Solid Red for clarity)
            loss_viz_params = {
                'palette': ['red'] # Solid red for pixels meeting loss threshold
                # min/max not needed here as we'll visualize the mask directly
            }

            # --- Process Period 1 (Need only NDVI median) ---
            self.signals.progress.emit(f"Processing Period 1 ({self.start1} to {self.end1})...")
            if self.is_cancelled: return
            collection_p1_base = s2_sr.filterBounds(aoi) \
                                      .filterDate(self.start1, self.end1) \
                                      .map(mask_s2_clouds_scl) # Apply SCL mask

            count1 = collection_p1_base.size().getInfo()
            if count1 == 0:
                self.signals.error.emit(f"No cloud-free images found for Period 1 in AOI.")
                return
            median_ndvi_p1 = collection_p1_base.map(calculate_ndvi).select('NDVI').median()


            # --- Process Period 2 (Need NDVI median and RGB median) ---
            self.signals.progress.emit(f"Processing Period 2 ({self.start2} to {self.end2})...")
            if self.is_cancelled: return
            collection_p2_base = s2_sr.filterBounds(aoi) \
                                      .filterDate(self.start2, self.end2) \
                                      .map(mask_s2_clouds_scl) # Apply SCL mask

            count2 = collection_p2_base.size().getInfo()
            if count2 == 0:
                self.signals.error.emit(f"No cloud-free images found for Period 2 in AOI.")
                return
            median_ndvi_p2 = collection_p2_base.map(calculate_ndvi).select('NDVI').median()
            # Create a median RGB image for Period 2 background
            median_rgb_p2 = collection_p2_base.select(['B4', 'B3', 'B2']).median()


            # --- Calculate Change & Identify Loss Mask---
            self.signals.progress.emit("Calculating NDVI change...")
            if self.is_cancelled: return
            ndvi_change = median_ndvi_p2.subtract(median_ndvi_p1).rename('NDVI_Change')
            # Create a mask image: 1 where loss occurred, masked (0) otherwise
            loss_mask = ndvi_change.lt(self.threshold).selfMask()


            # --- Create Visual Layers ---
            self.signals.progress.emit("Creating visual layers...")
            if self.is_cancelled: return
            # 1. Background Layer: Visualize the median RGB image from Period 2
            background_layer = median_rgb_p2.visualize(**rgb_viz_params)

            # 2. Loss Overlay Layer: Visualize the loss mask directly in red
            loss_overlay = loss_mask.visualize(**loss_viz_params)


            # --- Combine Layers ---
            # Mosaic the layers. Loss overlay is drawn on top of the background.
            # Where loss_overlay is masked (no significant loss), the background shows through.
            final_image = ee.ImageCollection([
                background_layer,
                loss_overlay
            ]).mosaic().clip(aoi) # Clip final mosaic to AOI bounds


            # --- Get Thumbnail URL of the Combined Image ---
            self.signals.progress.emit("Generating composite map preview...")
            if self.is_cancelled: return
            thumb_url = None # Initialize thumb_url
            try:
                # Get ThumbURL for the already visualized, mosaicked image
                thumb_url = final_image.getThumbURL({
                    'region': aoi.getInfo()['coordinates'], # Pass coordinates directly
                    'dimensions': self.thumb_size,
                    'format': 'png' # Request PNG format
                })
            except ee.EEException as thumb_e:
                 # Check common errors indicating no loss pixels or issues visualizing
                 if "No valid pixels" in str(thumb_e) or \
                    "Image has no bands" in str(thumb_e) or \
                    "Request payload size exceeds the limit" in str(thumb_e): # Add check for large AOI
                     self.signals.progress.emit("Calculation complete. No significant loss detected or issue visualizing loss layer.")
                     self.signals.progress.emit("Generating background map only...")
                     # If visualizing loss fails (e.g., no loss pixels), try getting just the background
                     try:
                         background_thumb_url = background_layer.getThumbURL({
                             'region': aoi.getInfo()['coordinates'],
                             'dimensions': self.thumb_size,
                             'format': 'png'
                         })
                         thumb_url = background_thumb_url # Use background URL instead
                     except Exception as bg_thumb_e:
                          self.signals.error.emit(f"GEE Error getting background map preview: {bg_thumb_e}")
                          return
                 else:
                    # Re-raise other GEE exceptions
                    raise thumb_e
            except Exception as e:
                # Catch other potential errors (e.g., during getInfo)
                self.signals.error.emit(f"Error during thumbnail URL generation: {e}")
                return

            # Check if we actually got a URL
            if not thumb_url:
                 self.signals.error.emit("Failed to generate a thumbnail URL for the map.")
                 return

            # --- Fetch Image Data ---
            self.signals.progress.emit("Downloading map image...")
            if self.is_cancelled: return
            response = requests.get(thumb_url)
            response.raise_for_status() # Raise an exception for bad HTTP status codes

            # --- Process Image with PIL ---
            self.signals.progress.emit("Processing image...")
            if self.is_cancelled: return
            img_data = response.content
            pil_image = Image.open(io.BytesIO(img_data))

            # Emit the result (the PIL image)
            self.signals.finished.emit(pil_image)

        except requests.exceptions.RequestException as req_e:
            self.signals.error.emit(f"Network Error downloading map: {req_e}")
        except ee.EEException as gee_e:
             self.signals.error.emit(f"Google Earth Engine Error: {gee_e}")
        except Exception as e:
            # Log detailed traceback for unexpected errors
            tb_str = traceback.format_exc()
            self.signals.error.emit(f"An unexpected error occurred in worker thread: {e}\nTrace: {tb_str}")


# --- Main Application Window ---
class GalamseyMonitorApp(QWidget):
    """Main GUI Application Window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Galamsey Monitor")
        self.setGeometry(100, 100, 800, 650) # Increased height slightly

        self.worker_thread = None
        self.worker = None
        self.progress_dialog = None

        # --- Layouts ---
        main_layout = QVBoxLayout(self)
        input_layout = QFormLayout()
        map_layout = QVBoxLayout()
        status_layout = QVBoxLayout()

        # --- Input Widgets ---
        input_group = QGroupBox("Analysis Parameters")
        # Example coordinates (adjust as needed)
        self.coord_input = QLineEdit("-1.795049, 6.335836, -1.745373, 6.365126")
        self.date1_start = QDateEdit(QDate(2020, 1, 1))
        self.date1_end = QDateEdit(QDate(2020, 12, 31))
        self.date2_start = QDateEdit(QDate(2025, 1, 1)) # Using a past date for Period 2
        self.date2_end = QDateEdit(QDate(2025, 5, 5)) # Using a past date for Period 2
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(-1.0, 0.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setValue(-0.20) # Default threshold
        self.analyze_button = QPushButton("Analyze Area")

        self.date1_start.setCalendarPopup(True)
        self.date1_end.setCalendarPopup(True)
        self.date2_start.setCalendarPopup(True)
        self.date2_end.setCalendarPopup(True)
        self.date1_start.setDisplayFormat("yyyy-MM-dd")
        self.date1_end.setDisplayFormat("yyyy-MM-dd")
        self.date2_start.setDisplayFormat("yyyy-MM-dd")
        self.date2_end.setDisplayFormat("yyyy-MM-dd")


        input_layout.addRow(QLabel("AOI Coordinates (lon_min, lat_min, lon_max, lat_max):"), self.coord_input)
        input_layout.addRow(QLabel("Period 1 Start Date:"), self.date1_start)
        input_layout.addRow(QLabel("Period 1 End Date:"), self.date1_end)
        input_layout.addRow(QLabel("Period 2 Start Date:"), self.date2_start)
        input_layout.addRow(QLabel("Period 2 End Date:"), self.date2_end)
        input_layout.addRow(QLabel("NDVI Change Threshold (Loss):"), self.threshold_input)
        input_layout.addRow(self.analyze_button)
        input_group.setLayout(input_layout)

        # --- Map Display Widget ---
        map_group = QGroupBox("Map Preview (Red shows potential vegetation loss)")
        self.map_label = QLabel("Map will appear here after analysis.")
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_label.setMinimumSize(500, 400) # Adjust size as needed
        self.map_label.setStyleSheet("QLabel { border: 1px solid gray; background-color: #f0f0f0; }")
        map_layout.addWidget(self.map_label)
        map_group.setLayout(map_layout)

        # --- Status Widget ---
        status_group = QGroupBox("Status Log")
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setFixedHeight(100)
        status_layout.addWidget(self.status_log)
        status_group.setLayout(status_layout)

        # --- Assemble Main Layout ---
        main_layout.addWidget(input_group)
        main_layout.addWidget(map_group)
        main_layout.addWidget(status_group)

        # --- Connections ---
        self.analyze_button.clicked.connect(self.run_analysis)

        # --- Initial GEE Check ---
        self.log_status("Attempting to initialize Google Earth Engine...")
        try:
            # Initialize GEE on startup. Replace with your Project ID.
            ee.Initialize(project='galamsey-monitor')
            self.log_status("Google Earth Engine initialized successfully.")
        except Exception as e:
            self.log_status(f"ERROR: Failed to initialize GEE on startup: {e}")
            self.log_status("Please ensure GEE is authenticated ('earthengine authenticate'),")
            self.log_status("the project ID is correct, and the Earth Engine API is enabled in Google Cloud.")
            QMessageBox.critical(self, "GEE Initialization Error",
                                 f"Failed to initialize Google Earth Engine:\n{e}\n\n"
                                 "Please ensure:\n"
                                 "1. You have run 'earthengine authenticate' in your terminal.\n"
                                 "2. You have internet access.\n"
                                 "3. The project ID ('galamsey-monitor') is correct.\n"
                                 "4. The Earth Engine API is enabled for this project in Google Cloud Console.")
            self.analyze_button.setEnabled(False)

    def log_status(self, message):
        """Appends a message to the status log."""
        self.status_log.append(message)
        QApplication.processEvents() # Allow UI updates during logging

    def run_analysis(self):
        """Starts the GEE analysis process in a background thread."""
        self.log_status("Starting analysis...")
        self.analyze_button.setEnabled(False)
        self.map_label.setText("Processing... Please wait.")
        self.map_label.setPixmap(QPixmap()) # Clear previous map

        # --- Get Inputs and Validate ---
        try:
            coords_str = self.coord_input.text().strip().split(',')
            if len(coords_str) != 4:
                raise ValueError("Coordinates must be 4 comma-separated numbers.")
            aoi_coords = [float(c.strip()) for c in coords_str]
            if not (aoi_coords[0] < aoi_coords[2] and aoi_coords[1] < aoi_coords[3]):
                 raise ValueError("Invalid coordinates order (must be min_lon, min_lat, max_lon, max_lat).")

            start1 = self.date1_start.date().toString("yyyy-MM-dd")
            end1 = self.date1_end.date().toString("yyyy-MM-dd")
            start2 = self.date2_start.date().toString("yyyy-MM-dd")
            end2 = self.date2_end.date().toString("yyyy-MM-dd")
            threshold = self.threshold_input.value()

            # Validate date ranges
            date_start1 = self.date1_start.date()
            date_end1 = self.date1_end.date()
            date_start2 = self.date2_start.date()
            date_end2 = self.date2_end.date()

            if date_start1 >= date_end1 or date_start2 >= date_end2:
                raise ValueError("Start date must be before end date for each period.")
            if date_end1 >= date_start2:
                 raise ValueError("Period 1 must end before Period 2 begins.")
            # Check if Period 2 is in the future (optional, but good practice)
            if date_start2 > QDate.currentDate():
                 QMessageBox.warning(self, "Future Date Warning",
                                     "Period 2 starts in the future. Analysis will use data up to the present.")

        except ValueError as ve:
            self.log_status(f"ERROR: Invalid input - {ve}")
            QMessageBox.warning(self, "Input Error", f"Invalid input:\n{ve}")
            self.analyze_button.setEnabled(True)
            return
        except Exception as e:
            self.log_status(f"ERROR: Unexpected input error - {e}")
            QMessageBox.warning(self, "Input Error", f"Unexpected input error:\n{e}")
            self.analyze_button.setEnabled(True)
            return

        # --- Setup Progress Dialog ---
        self.progress_dialog = QProgressDialog("Running GEE Analysis...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(False) # Keep open until explicitly closed
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumDuration(0) # Show immediately
        self.progress_dialog.canceled.connect(self.cancel_analysis)
        self.progress_dialog.setValue(0) # Use as indeterminate progress bar
        self.progress_dialog.show()

        # --- Setup Worker Thread ---
        self.worker = GEEWorker(aoi_coords, start1, end1, start2, end2, threshold)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True) # Use daemon thread

        # Connect worker signals to GUI slots
        self.worker.signals.finished.connect(self.on_analysis_complete)
        self.worker.signals.error.connect(self.on_analysis_error)
        self.worker.signals.progress.connect(self.update_progress)

        # Start the thread
        self.worker_thread.start()

    def update_progress(self, message):
        """Updates the status log and progress dialog message."""
        self.log_status(message)
        if self.progress_dialog and self.progress_dialog.isVisible():
            # Use invokeMethod for thread safety when updating GUI from signal
             QMetaObject.invokeMethod(self.progress_dialog, "setLabelText", Qt.ConnectionType.QueuedConnection, Q_ARG(str, message))

    def on_analysis_complete(self, pil_image):
        """Handles successful completion of the GEE analysis."""
        self.log_status("Analysis thread finished.")
        if self.progress_dialog:
            self.progress_dialog.close()

        if pil_image:
            try:
                # Convert PIL Image (likely RGBA from PNG) to QPixmap
                img_byte_array = io.BytesIO()
                # Ensure saving as PNG to preserve potential transparency
                pil_image.save(img_byte_array, format='PNG')
                img_byte_array.seek(0)

                qimage = QImage.fromData(img_byte_array.read())
                if qimage.isNull():
                    self.log_status("ERROR: Failed to load received image data into QImage.")
                    self.map_label.setText("Error: Could not display map image.")
                else:
                    pixmap = QPixmap.fromImage(qimage)
                    # Scale pixmap to fit the label while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(self.map_label.size(),
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
                    self.map_label.setPixmap(scaled_pixmap)
                    self.log_status("Map preview displayed.")

            except Exception as e:
                self.log_status(f"ERROR displaying map image: {e}")
                tb_str = traceback.format_exc()
                self.log_status(f"Traceback: {tb_str}")
                self.map_label.setText("Error displaying map.")
        else:
             # Handle case where worker signaled finished but sent None (e.g., no loss detected, background shown)
             # Status messages should have been logged by the worker.
             # Check if map label still shows processing text
             if "Processing..." in self.map_label.text():
                  self.map_label.setText("Analysis complete. No significant loss detected\nor map could not be generated.")


        self.analyze_button.setEnabled(True)
        self.worker_thread = None # Clear thread reference
        self.worker = None

    def on_analysis_error(self, error_message):
        """Handles errors reported by the GEE worker thread."""
        self.log_status(f"ERROR during analysis: {error_message}")
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n{error_message}")
        self.map_label.setText("Analysis failed. See Status Log.")
        self.analyze_button.setEnabled(True)
        self.worker_thread = None
        self.worker = None

    def cancel_analysis(self):
        """Handles user cancellation via the progress dialog."""
        self.log_status("Analysis cancelled by user.")
        if self.worker:
            self.worker.is_cancelled = True # Signal the worker thread to stop early if possible
        # Note: Thread termination isn't guaranteed immediately.
        # GEE tasks might continue until they finish or error out.
        self.analyze_button.setEnabled(True)
        self.map_label.setText("Analysis Cancelled.")
        # Progress dialog closes automatically here if autoClose was true,
        # otherwise, we closed it manually in error/complete slots.
        if self.progress_dialog:
             self.progress_dialog.close()
        self.worker_thread = None # Allow garbage collection
        self.worker = None

    def closeEvent(self, event):
        """Handles the main window being closed."""
        if self.worker_thread and self.worker_thread.is_alive():
             self.log_status("Window closing, attempting to signal cancel to ongoing analysis...")
             self.cancel_analysis() # Signal cancellation
             # Give thread a very short time, but don't block exit
             self.worker_thread.join(timeout=0.2)
             if self.worker_thread.is_alive():
                  self.log_status("Warning: Analysis thread may still be running in background.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    monitor_app = GalamseyMonitorApp()
    monitor_app.show()
    sys.exit(app.exec())