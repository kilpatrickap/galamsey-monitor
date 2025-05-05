import sys
import io
import threading # To run GEE tasks in the background

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


# --- Worker Object for Threading ---
# Necessary to emit signals from a non-GUI thread back to the GUI thread
class WorkerSignals(QObject):
    finished = pyqtSignal(object) # Pass the result (PIL image or None)
    error = pyqtSignal(str)       # Pass error message
    progress = pyqtSignal(str)    # Pass status updates

class GEEWorker(QObject):
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
        try:
            self.signals.progress.emit("Initializing Earth Engine...")
            # Initialize EE within the thread if not already done globally
            # Note: Frequent ee.Initialize() calls are okay.
            try:
                # *** MODIFICATION HERE ***
                # Replace 'YOUR_PROJECT_ID_HERE' with your actual GCP Project ID
                ee.Initialize(project='galamsey-monitor')
            except Exception as init_e:
                self.signals.error.emit(f"GEE Initialization failed: {init_e}")
                return

            self.signals.progress.emit("Defining AOI...")
            if self.is_cancelled: return
            aoi = ee.Geometry.Rectangle(self.aoi_coords)

            # --- Define GEE Functions (as before) ---
            def calculate_ndvi(image):
                return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

            def mask_s2_clouds(image):
                qa = image.select('QA60')
                cloud_bit_mask = 1 << 10
                cirrus_bit_mask = 1 << 11
                mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                       qa.bitwiseAnd(cirrus_bit_mask).eq(0))
                return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

            # Use the recommended Harmonized Sentinel-2 Surface Reflectance collection
            s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

            # --- Process Period 1 ---
            self.signals.progress.emit(f"Processing Period 1 ({self.start1} to {self.end1})...")
            if self.is_cancelled: return
            collection_p1 = s2_sr.filterBounds(aoi) \
                                 .filterDate(self.start1, self.end1) \
                                 .map(mask_s2_clouds)
            # Check if collection is empty
            count1 = collection_p1.size().getInfo()
            if count1 == 0:
                self.signals.error.emit(f"No cloud-free images found for Period 1.")
                return
            median_ndvi_p1 = collection_p1.map(calculate_ndvi).select('NDVI').median()


            # --- Process Period 2 ---
            self.signals.progress.emit(f"Processing Period 2 ({self.start2} to {self.end2})...")
            if self.is_cancelled: return
            collection_p2 = s2_sr.filterBounds(aoi) \
                                 .filterDate(self.start2, self.end2) \
                                 .map(mask_s2_clouds)
            # Check if collection is empty
            count2 = collection_p2.size().getInfo()
            if count2 == 0:
                self.signals.error.emit(f"No cloud-free images found for Period 2.")
                return
            median_ndvi_p2 = collection_p2.map(calculate_ndvi).select('NDVI').median()

            # --- Calculate Change & Identify Loss ---
            self.signals.progress.emit("Calculating NDVI change...")
            if self.is_cancelled: return
            ndvi_change = median_ndvi_p2.subtract(median_ndvi_p1).rename('NDVI_Change')
            loss_mask = ndvi_change.lt(self.threshold)
            significant_loss = ndvi_change.updateMask(loss_mask).clip(aoi) # Clip result to AOI

            # --- Get Thumbnail URL ---
            self.signals.progress.emit("Generating map preview...")
            if self.is_cancelled: return
            loss_viz_params = {
                'min': self.threshold, # Use the dynamic threshold
                'max': 0,
                'palette': ['red', 'orange', 'yellow'], # Red = max loss
                'region': aoi, # Important for getThumbURL
                'dimensions': self.thumb_size # Size of the preview image
            }
            try:
                # Use getThumbURL to get a direct image URL
                thumb_url = significant_loss.getThumbURL(loss_viz_params)
            except ee.EEException as thumb_e:
                 # Check if the error is because no pixels met the threshold
                 if "No valid pixels were found" in str(thumb_e) or "Image.visualize: No valid pixels" in str(thumb_e):
                     self.signals.progress.emit("Calculation complete. No significant vegetation loss detected matching the threshold.")
                     self.signals.finished.emit(None) # Send None to indicate no map
                     return
                 else:
                    self.signals.error.emit(f"GEE Error getting map preview: {thumb_e}")
                    return

            # --- Fetch Image Data ---
            self.signals.progress.emit("Downloading map image...")
            if self.is_cancelled: return
            response = requests.get(thumb_url)
            response.raise_for_status() # Raise an exception for bad status codes

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
            self.signals.error.emit(f"An unexpected error occurred: {e}")

# --- Main Application Window ---
class GalamseyMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Galamsey Monitor")
        self.setGeometry(100, 100, 800, 600) # x, y, width, height

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
        self.coord_input = QLineEdit("-1.795049, 6.335836, -1.745373, 6.365126") # Default Example AOI
        self.date1_start = QDateEdit(QDate(2020, 1, 1))
        self.date1_end = QDateEdit(QDate(2020, 12, 31))
        self.date2_start = QDateEdit(QDate(2025, 1, 1))
        self.date2_end = QDateEdit(QDate(2025, 5, 1))
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(-1.0, 0.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setValue(-0.20) # Default threshold
        self.analyze_button = QPushButton("Analyze Area")

        self.date1_start.setCalendarPopup(True)
        self.date1_end.setCalendarPopup(True)
        self.date2_start.setCalendarPopup(True)
        self.date2_end.setCalendarPopup(True)

        input_layout.addRow(QLabel("AOI Coordinates (lon_min, lat_min, lon_max, lat_max):"), self.coord_input)
        input_layout.addRow(QLabel("Period 1 Start Date:"), self.date1_start)
        input_layout.addRow(QLabel("Period 1 End Date:"), self.date1_end)
        input_layout.addRow(QLabel("Period 2 Start Date:"), self.date2_start)
        input_layout.addRow(QLabel("Period 2 End Date:"), self.date2_end)
        input_layout.addRow(QLabel("NDVI Change Threshold (Loss):"), self.threshold_input)
        input_layout.addRow(self.analyze_button)
        input_group.setLayout(input_layout)

        # --- Map Display Widget ---
        map_group = QGroupBox("Potential Vegetation Loss Map Preview")
        self.map_label = QLabel("Map will appear here after analysis.\nRed/Orange indicates potential loss.")
        self.map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_label.setMinimumSize(400, 300) # Give it some initial size
        self.map_label.setStyleSheet("QLabel { border: 1px solid gray; }")
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
            # *** MODIFICATION HERE ***
            # Replace 'YOUR_PROJECT_ID_HERE' with your actual GCP Project ID
            ee.Initialize(project='galamsey-monitor')
            self.log_status("Google Earth Engine initialized successfully.")
        except Exception as e:
            # Error message now includes the project parameter hint
            self.log_status(f"ERROR: Failed to initialize GEE on startup: {e}")
            self.log_status("Please ensure you have run 'earthengine authenticate', have internet access,")
            self.log_status("and the project ID is correct and has the Earth Engine API enabled.")
            QMessageBox.critical(self, "GEE Error",
                                 f"Failed to initialize Google Earth Engine:\n{e}\n\nPlease ensure you are authenticated, online, and using a valid Project ID with the Earth Engine API enabled.")
            self.analyze_button.setEnabled(False)  # Disable analysis if init fails

    def log_status(self, message):
        self.status_log.append(message)
        QApplication.processEvents() # Keep UI responsive during logging

    def run_analysis(self):
        self.log_status("Starting analysis...")
        self.analyze_button.setEnabled(False)
        self.map_label.setText("Processing... Map will appear here.") # Clear previous map

        # --- Get Inputs and Validate ---
        try:
            coords_str = self.coord_input.text().split(',')
            if len(coords_str) != 4:
                raise ValueError("Coordinates must be 4 comma-separated numbers.")
            aoi_coords = [float(c.strip()) for c in coords_str]
            # Basic validation (lon_min < lon_max, lat_min < lat_max)
            if not (aoi_coords[0] < aoi_coords[2] and aoi_coords[1] < aoi_coords[3]):
                 raise ValueError("Invalid coordinates order (must be min_lon, min_lat, max_lon, max_lat).")

            start1 = self.date1_start.date().toString("yyyy-MM-dd")
            end1 = self.date1_end.date().toString("yyyy-MM-dd")
            start2 = self.date2_start.date().toString("yyyy-MM-dd")
            end2 = self.date2_end.date().toString("yyyy-MM-dd")
            threshold = self.threshold_input.value()

            if self.date1_start.date() >= self.date1_end.date() or \
               self.date2_start.date() >= self.date2_end.date() or \
               self.date1_end.date() >= self.date2_start.date():
                raise ValueError("Date ranges are invalid or overlapping incorrectly.")

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
        self.progress_dialog.setAutoClose(True) # Close when finished or canceled
        self.progress_dialog.setAutoReset(True) # Reset on next show()
        self.progress_dialog.canceled.connect(self.cancel_analysis) # Connect cancel signal
        self.progress_dialog.show()

        # --- Setup Worker Thread ---
        self.worker = GEEWorker(aoi_coords, start1, end1, start2, end2, threshold)
        self.worker_thread = threading.Thread(target=self.worker.run)

        # Connect worker signals to GUI slots
        self.worker.signals.finished.connect(self.on_analysis_complete)
        self.worker.signals.error.connect(self.on_analysis_error)
        self.worker.signals.progress.connect(self.update_progress)

        # Start the thread
        self.worker_thread.start()

    def update_progress(self, message):
        # Update both the log and the progress dialog label
        self.log_status(message)
        if self.progress_dialog:
            # Use invokeMethod for thread safety when updating GUI from signal
             QMetaObject.invokeMethod(self.progress_dialog, "setLabelText", Qt.ConnectionType.QueuedConnection, Q_ARG(str, message))

    def on_analysis_complete(self, pil_image):
        self.log_status("Analysis finished.")
        if self.progress_dialog:
            self.progress_dialog.close() # Close progress dialog cleanly

        if pil_image:
            try:
                # Convert PIL Image to QPixmap for display
                img_byte_array = io.BytesIO()
                pil_image.save(img_byte_array, format='PNG') # Save to bytes buffer
                img_byte_array.seek(0)

                qimage = QImage.fromData(img_byte_array.read())
                if qimage.isNull():
                    self.log_status("ERROR: Failed to load image data into QImage.")
                    self.map_label.setText("Error displaying map.")
                else:
                    pixmap = QPixmap.fromImage(qimage)
                    # Scale pixmap to fit the label while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(self.map_label.size(),
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
                    self.map_label.setPixmap(scaled_pixmap)
                    self.log_status("Map preview displayed.")

            except Exception as e:
                self.log_status(f"ERROR displaying image: {e}")
                self.map_label.setText("Error displaying map.")
        else:
            # No significant loss detected, or an error occurred where no image was generated
             self.map_label.setText("No significant loss detected\nor map could not be generated.")
             # Status already logged by worker


        self.analyze_button.setEnabled(True)
        self.worker_thread = None # Clear thread reference
        self.worker = None

    def on_analysis_error(self, error_message):
        self.log_status(f"ERROR: {error_message}")
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n{error_message}")
        self.map_label.setText("Analysis failed.")
        self.analyze_button.setEnabled(True)
        self.worker_thread = None
        self.worker = None

    def cancel_analysis(self):
        self.log_status("Analysis cancelled by user.")
        if self.worker:
            self.worker.is_cancelled = True # Signal the worker to stop
        if self.worker_thread and self.worker_thread.is_alive():
             # Give the thread a moment to potentially notice the cancel flag
             self.worker_thread.join(timeout=0.5) # Don't wait indefinitely

        self.analyze_button.setEnabled(True)
        self.map_label.setText("Analysis Cancelled.")
        self.worker_thread = None
        self.worker = None
        # Progress dialog closes automatically due to setAutoClose(True)

    # Ensure worker thread is handled if window is closed early
    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.is_alive():
             self.log_status("Window closing, attempting to cancel ongoing analysis...")
             self.cancel_analysis() # Attempt graceful cancellation
             if self.worker_thread and self.worker_thread.is_alive():
                 self.log_status("Analysis thread still active, waiting briefly...")
                 self.worker_thread.join(timeout=1.0) # Wait a bit longer
                 if self.worker_thread.is_alive():
                     self.log_status("Warning: Analysis thread did not terminate cleanly.")
        event.accept()


if __name__ == "__main__":
    # Need to enable high DPI scaling for better visuals on some systems
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    monitor_app = GalamseyMonitorApp()
    monitor_app.show()
    sys.exit(app.exec())