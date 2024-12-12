import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, QWidget)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

class ImageMatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize UI and variables
        self.initUI()
        self.camera_on = False
        self.template_image = None
        self.display_markers = True
        self.connect_markers = False

    def initUI(self):
        # Set up the main application window
        self.setWindowTitle("ORB Image Matcher")

        # Video feed label
        self.video_label = QLabel("Camera feed")
        self.video_label.setFixedSize(960, 720)

        # Buttons to load template image, start, and stop the camera
        self.load_button = QPushButton("Load Template Image")
        self.load_button.clicked.connect(self.load_template_image)

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)

        # Checkboxes to toggle display options
        self.marker_checkbox = QCheckBox("Display Markers")
        self.marker_checkbox.setChecked(True)
        self.marker_checkbox.stateChanged.connect(self.toggle_markers)

        self.connect_checkbox = QCheckBox("Connect Markers")
        self.connect_checkbox.setChecked(False)
        self.connect_checkbox.stateChanged.connect(self.toggle_connect_markers)

        # Layout setup
        layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.marker_checkbox)
        checkbox_layout.addWidget(self.connect_checkbox)
        layout.addLayout(checkbox_layout)

        layout.addWidget(self.video_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer for video feed updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None

    def load_template_image(self):
        # Load the template image from a file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Template Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            self.template_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.template_image is not None:
                print("Template image loaded successfully.")
                # Resize template image if larger than 500 pixels in any dimension
                height, width = self.template_image.shape[:2]
                if max(height, width) > 450:
                    scaling_factor = 450 / max(height, width)
                    self.template_image = cv2.resize(self.template_image, (int(width * scaling_factor), int(height * scaling_factor)))
            else:
                print("Failed to load template image.")

    def start_camera(self):
        # Start the camera and begin capturing frames
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot access the camera.")
            return
        self.camera_on = True
        self.timer.start(30)

    def stop_camera(self):
        # Stop the camera and release resources
        if self.cap:
            self.cap.release()
        self.camera_on = False
        self.timer.stop()
        self.video_label.clear()

    def toggle_markers(self):
        # Toggle display of keypoints
        self.display_markers = self.marker_checkbox.isChecked()

    def toggle_connect_markers(self):
        # Toggle connecting keypoints between images
        self.connect_markers = self.connect_checkbox.isChecked()

    def update_frame(self):
        # Capture and process a frame from the camera
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                return

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.template_image is not None:
                # Detect keypoints and descriptors using ORB
                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(self.template_image, None)
                kp2, des2 = orb.detectAndCompute(gray_frame, None)

                if des1 is not None and des2 is not None:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)

                    if self.connect_markers:
                        # Draw matches between the template and camera feed
                        scale_factor = 1
                        template_resized = cv2.resize(self.template_image, (0, 0), fx=scale_factor, fy=scale_factor)
                        result_image = cv2.drawMatches(template_resized, kp1, frame, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                        # Convert to QImage for display
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        image = QImage(result_image_rgb.data, result_image_rgb.shape[1], result_image_rgb.shape[0], result_image_rgb.strides[0], QImage.Format_RGB888)
                        self.video_label.setPixmap(QPixmap.fromImage(image))
                        return

                if self.display_markers:
                    # Draw keypoints on the frame
                    frame_with_keypoints = cv2.drawKeypoints(frame, kp2, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT)

                    # Convert to QImage for display
                    frame_rgb = cv2.cvtColor(frame_with_keypoints, cv2.COLOR_BGR2RGB)
                    image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(image))
                    return

            # Default display without markers or matches
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        # Ensure resources are released on closing the application
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = ImageMatcherApp()
    window.show()
    app.exec_()