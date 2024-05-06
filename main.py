import cv2
import pyqtgraph
import qdarkstyle
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import numpy as np
import os
from os import path
import sys
import numpy as np
from scipy.spatial.distance import cdist
# import pyqtgraph as pg
from PIL import Image
import matplotlib.pyplot as plt
import skimage.exposure as exposure
from scipy.ndimage import filters
import scipy
from io import BytesIO
import time
from scipy.signal import convolve2d
from scipy.spatial.distance import pdist, squareform
from sift import computeKeypointsAndDescriptors
from scipy.cluster.hierarchy import linkage, fcluster
from skimage.color import rgb2gray
from skimage.segmentation import active_contour

FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "main.ui"))


class ImageConverter:
    @staticmethod
    def numpy_to_pixmap(array):
        # array = (array - array.min()) / (array.max() - array.min()) * 255
        # array = array.astype(np.uint8)
        # # Check if the array is 2D or 3D
        # if len(array.shape) == 2:
        #     # For 2D arrays
        #     height, width = array.shape
        #     bytes_per_line = width
        #     img = QImage(array.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        # else:
        #     print("Unsupported array shape.")
        #     return None
        # return QPixmap.fromImage(img)

        # Normalize the input array to the range [0, 255] by scaling its values
        array = (array - array.min()) / (array.max() - array.min()) * 255
        array = array.astype(np.uint8)

        # Check if the array is 2D or 3D
        if len(array.shape) == 2:
            # For 2D arrays (grayscale)
            height, width = array.shape
            bytes_per_line = width
            img = QImage(array.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(array.shape) == 3 and array.shape[2] == 3:
            # For 3D arrays (RGB color images)
            height, width, channels = array.shape
            bytes_per_line = width * channels
            img = QImage(array.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            print("Unsupported array shape.")
            return None

        # Convert the QImage to a QPixmap and return it
        return QPixmap.fromImage(img)


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        # self.initialize()
        self.imagePath = None
        self.initializeUI()
        self.Handle_Buttons()
        pyqtgraph.setConfigOptions(antialias=True)
        self.first_image_combobox.currentIndexChanged.connect(self.choose_hybrid_mood)
        self.create_hybrid_button.clicked.connect(self.choose_hybrid_mood)

        self.filtered_images_array = [[None, None], [None, None]]
        self.set_default_values()
        self.initializeThresholding()
        self.cvalue_slider.hide()
        self.cvalue_label.hide()
        self.cvalue_lcd.hide()
        self.global_radiobutton.toggled.connect(self.toggleGlobalThresholdingWidgets)
        self.last_local_threshold_value = None
        self.last_global_threshold_value = None
        self.local_radiobutton.toggled.connect(self.toggleLocalThresholdingWidgets)
        self.cvalue_slider.valueChanged.connect(self.applyThresholding)  # task4
        self.segmentation_spinBox.valueChanged.connect(self.get_spinbox_value)  # task4
        self.segmentation_type_comboBox.currentIndexChanged.connect(self.change_segmentation_mode)  # task4
        self.faster_check_box.setVisible(False) #task 4
        self.normalize_button.clicked.connect(self.normalise_image)
        self.hist_equalize_button.clicked.connect(self.hist_equalize)
        layout = QVBoxLayout(self.distribution_curve_graph)
        # self.dist_curve_widget = pg.PlotWidget()
        # layout.addWidget(self.dist_curve_widget)
        # layout2 = QVBoxLayout(self.histogram_graph)
        # self.hist_curve_widget = pg.PlotWidget()
        # layout2.addWidget(self.hist_curve_widget)
        # layout3 = QVBoxLayout(self.processed_image_distribution)
        # self.processed_image_distribution = pg.PlotWidget()
        # layout3.addWidget(self.processed_image_distribution)
        # layout3 = QVBoxLayout(self.original_image_distribution)
        # self.original_image_distribution = pg.PlotWidget()
        # layout3.addWidget(self.original_image_distribution)
        self.kernel_slider.setMinimum(1)
        self.kernel_slider.setSingleStep(2)
        self.kernel_slider.setMaximum(50)
        self.sigma_slider.setMinimum(1)
        self.sigma_slider.setMaximum(10)

        ## task 3
        self.sift_btn.clicked.connect(lambda: self.options_selector.setCurrentIndex(1))
        self.harris_btn.clicked.connect(lambda: self.options_selector.setCurrentIndex(0))
        self.feat_matching_btn.clicked.connect(lambda: self.options_selector.setCurrentIndex(2))
        self.calc_sift_btn.clicked.connect(lambda: self.calc_sift())
        self.sigma_slider.setValue(5)

        ## Task 4
        self.thresholding_btn.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.segmentation_btn.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.before_segmentation_image.mousePressEvent = self.get_mouse_coordinates
        self.region_growing_x_pos = 0
        self.region_growing_y_pos = 0

        # layout = QVBoxLayout(self.R_hist)
        # self.R_hist = pg.PlotWidget()
        # layout.addWidget(self.R_hist)
        # layout = QVBoxLayout(self.G_hist)
        # self.G_hist = pg.PlotWidget()
        # layout.addWidget(self.G_hist)
        # layout = QVBoxLayout(self.B_hist)
        # self.B_hist = pg.PlotWidget()
        # layout.addWidget(self.B_hist)

        ## filters
        self.filters = {
            "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            "roberts_x": np.array([[1, 0], [0, -1]]),
            "roberts_y": np.array([[0, 1], [-1, 0]]),
            "prewitt_x": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "prewitt_y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        }

        self.directions = {
            (0, -1): 0,  # North
            (1, -1): 1,  # Northeast
            (1, 0): 2,  # East
            (1, 1): 3,  # Southeast
            (0, 1): 4,  # South
            (-1, 1): 5,  # Southwest
            (-1, 0): 6,  # West
            (-1, -1): 7  # Northwest
        }
        self.max_iterations_slider.setMinimum(0)
        self.max_iterations_slider.setMaximum(400)
        self.max_iterations_slider.setValue(250)

        self.alpha_slider.setMinimum(1)
        self.alpha_slider.setMaximum(1200)
        self.alpha_slider.setValue(1100)

        self.beta_slider.setMinimum(1)
        self.beta_slider.setMaximum(1200)
        self.beta_slider.setValue(1100)

        self.kvalue_spinbox.setMinimum(0.01)
        self.kvalue_spinbox.setMaximum(10)
        self.kvalue_spinbox.setValue(0.04)
        self.threshold_spinbox.setMinimum(0)
        self.threshold_spinbox.setMaximum(400)
        self.threshold_spinbox.setValue(150)

        self.types_of_filters = ["sobel", "roberts", "prewitt", "canny"]
        self.filter_options = ["Horizontal", "Vertical", "Magnitude"]

        self.populate_combobox()

        self.matching_lines_slider.setMinimum(1)
        self.matching_lines_slider.setMaximum(100)
        self.matching_lines_slider.setValue(10)

        ## Task 4
        self.block_size_slider.setMinimum(5)
        self.block_size_slider.setMaximum(90)
        self.block_size_slider.setValue(50)
        self.local_radio_btn.setChecked(True)
        # self.index = self.thresholding_combobox.currentIndex()
        self.num_thresholds_label.hide()
        self.num_thresholds_slider.hide()
        self.num_thresholds_lcd.hide()

        self.segmentation_spinbox_value = 1

    # def calc_sift(self):
    #     image = self.image
    #     # start timing
    #     start_time = time.time()
    #     # calc sift from sift file
    #     keypoints, descriptors = computeKeypointsAndDescriptors(image)
    #     image_with_keypoints = cv2.drawKeypoints(self.image_color, keypoints, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     end_time = time.time()  # End timing
    #     computation_time = end_time - start_time  # Compute total time taken
    #     self.sift_lcd.display(computation_time)
    #     # Convert the image to QImage
    #     image_with_keypoints_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
    #     height, width, channel = image_with_keypoints_rgb.shape
    #     bytesPerLine = 3 * width
    #     qImg = QImage(image_with_keypoints_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
    #
    #     # Convert QImage to QPixmap
    #     pixmap = QPixmap.fromImage(qImg)
    #     pixmap = pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #     scaled_pixmap = pixmap
    #     self.sift_key_points_image.setPixmap(scaled_pixmap)
    #     self.sift_key_points_image.setScaledContents(False)
    #     print(keypoints[1])
    def populate_combobox(self):
        self.type_combobox.addItems(self.types_of_filters)
        self.alignment_combobox.addItems(self.filter_options)

    def initializeUI(self):
        self.setWindowTitle("Edge Detection  and Filtering")

    def Handle_Buttons(self):
        self.actionUpload_Image.triggered.connect(lambda: self.openImageDialog(mode='1'))
        self.actionUpload_Second_Image.triggered.connect(lambda: self.openImageDialog(mode='2'))
        self.gaussian_noise.clicked.connect(self.add_gaussian_noise)
        self.salt_pepper_noise.clicked.connect(self.add_salt_and_pepper_noise)
        self.average_noise.clicked.connect(self.add_average_noise)
        self.median_filter_button.clicked.connect(self.apply_median_filter)
        self.gaussian_filter_button.clicked.connect(self.apply_gaussian_filter)
        self.average_filter_button.clicked.connect(self.apply_average_filter)
        self.histogram_radiobutton.toggled.connect(self.dis_hist)
        self.DF_radiobutton.toggled.connect(self.display_pdf)
        self.CDF_radiobutton.toggled.connect(self.display_cdf)
        self.submit_button.clicked.connect(self.choose_filter)
        self.type_combobox.currentIndexChanged.connect(self.hide_canny)
        self.apply_changes_btn.clicked.connect(self.contour_image)
        self.detection_button.clicked.connect(self.handle_detection)
        self.corner_detection_btn.clicked.connect(self.harrisCornerDetector)  ## task 3
        self.features_extraction_button.clicked.connect(self.match_features_and_visualize)  ## task 3
        self.corner_detection_btn.clicked.connect(self.shiTomasiCornerDetection)  ## task 3
        self.submit_thresholding_btn.clicked.connect(self.perform_thresholding)  ## Task 4
        self.global_radio_btn.toggled.connect(self.thresholding_radio_buttons_toggled)  ## Task 4
        self.thresholding_combobox.currentIndexChanged.connect(self.update_sliders)  ## Task 4
        self.submit_segmentation_btn.clicked.connect(self.submit_segmentation_btn_clicked)  ## Task 4
        self.luv_button.clicked.connect(self.rgb_to_luv)  ## Task 4

    ############ Task4 #############
    def change_segmentation_mode(self):
        segmentation_type = self.segmentation_type_comboBox.currentText()

        # Perform action based on selected value
        if segmentation_type == "K-means":
            self.segmentation_label.setText("Clusters")
            self.segmentation_spinBox.setSingleStep(1)
            self.segmentation_spinBox.setMinimum(1)  # Minimum value
            self.segmentation_spinBox.setMaximum(20)  # Maximum value

        elif segmentation_type == "Mean Shift":
            self.segmentation_label.setText("Band Width")
            self.segmentation_spinBox.setSingleStep(10)
            self.segmentation_spinBox.setMinimum(1)  # Minimum value
            self.segmentation_spinBox.setMaximum(200)  # Maximum value
            self.faster_check_box.setVisible(False)
        elif segmentation_type == "Agglomerative":
            self.segmentation_label.setText("Clusters")
            self.segmentation_spinBox.setSingleStep(1)
            self.segmentation_spinBox.setMinimum(1)  # Minimum value
            self.segmentation_spinBox.setMaximum(20)  # Maximum value
            self.faster_check_box.setVisible(True)
        else:
            self.segmentation_label.setText("Threshold")
            self.segmentation_spinBox.setSingleStep(1)
            self.segmentation_spinBox.setMinimum(1)  # Minimum value
            self.segmentation_spinBox.setMaximum(20)  # Maximum value
            self.faster_check_box.setVisible(False)


    def submit_segmentation_btn_clicked(self):
        segmentation_type = self.segmentation_type_comboBox.currentText()
        # Perform action based on selected value
        if segmentation_type == "K-means":
            self.apply_k_means()
        elif segmentation_type == "Mean Shift":
            self.apply_mean_shift()
        elif segmentation_type == "Agglomerative":
            self.apply_agglomerative(self.image)
        elif segmentation_type == "Region Growing":
            self.apply_region_growing(self.image)

    def openImageDialog(self, mode):
        # Open a file dialog to select an image
        imagePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if mode == '1':
            if imagePath:
                self.imagePath = imagePath
                self.image = cv2.imread(self.imagePath, cv2.IMREAD_GRAYSCALE)
                self.image_color = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)
                # Convert the image from BGR to RGB
                self.IMG = cv2.imread(self.imagePath)
                # Convert the image from BGR to RGB
                self.img_rgb = cv2.cvtColor(self.IMG, cv2.COLOR_BGR2RGB)
                self.show_image(QImage(imagePath), mode)
                self.apply_fourier_transform(mode, imagePath)
                self.applyThresholding()
                self.draw_curves()
                self.filtered_image.clear()
                # self.contour_image()
        else:
            if imagePath:
                self.show_image(QImage(imagePath), mode)
                self.apply_fourier_transform(mode, imagePath)
                self.imagePath2 = imagePath

    ####################################### Task 4 Functions ############################################
    def update_sliders(self, index):
        self.block_size_slider.setMinimum(5)
        self.block_size_slider.setMaximum(90)
        self.block_size_slider.setValue(50)
        if index == 2:  # Check if the index is 2
            self.num_thresholds_label.show()
            self.num_thresholds_slider.show()
            self.num_thresholds_lcd.show()
            self.num_thresholds_slider.setMinimum(1)
            self.num_thresholds_slider.setMaximum(10)
            self.num_thresholds_slider.setValue(2)
        else:
            self.num_thresholds_label.hide()
            self.num_thresholds_slider.hide()
            self.num_thresholds_lcd.hide()

    def perform_thresholding(self):
        self.index = self.thresholding_combobox.currentIndex()
        if self.index == 0:
            if self.local_radio_btn.isChecked():
                self.local_optimal_thresholding()
            elif self.global_radio_btn.isChecked():
                self.global_optimal_thresholding()
        elif self.index == 1:
            if self.local_radio_btn.isChecked():
                self.local_otsu_thresholding()
            if self.global_radio_btn.isChecked():
                self.global_otsu_thresholding()
        else:
            if self.local_radio_btn.isChecked():
                self.local_multi_level_thresholding()
            elif self.global_radio_btn.isChecked():
                self.global_multi_level_thresholding()

    def thresholding_radio_buttons_toggled(self, checked):
        if checked:
            self.block_size_slider.hide()
            self.block_size_lcd.hide()
            self.block_size_label.hide()
        else:
            self.block_size_slider.show()
            self.block_size_lcd.show()
            self.block_size_label.show()

    def global_optimal_thresholding(self):
        # Initialize the threshold and previous threshold
        threshold = 0
        previous_threshold = threshold + 1  # Initialize to a value that ensures the loop runs at least once

        # Iterate until convergence
        while abs(previous_threshold - threshold) != 0:
            previous_threshold = threshold  # Update the previous threshold before updating the current one

            # Compute the mean values of the background and object regions
            background_mean = np.mean([self.image[0, 0], self.image[0, -1], self.image[-1, 0], self.image[-1, -1]])
            object_mean = np.mean(self.image[1:-1, 1:-1])

            # Update the threshold using the means of the background and object
            threshold = (background_mean + object_mean) / 2

        # Apply thresholding to the image
        thresholded_image = (self.image > threshold) * 255

        self.apply_thresholding_on_image(thresholded_image)

    def local_optimal_thresholding(self):
        print("optimal")  # Debugging or informational message

        # Get image dimensions
        block_size = self.block_size_slider.value()
        height, width = self.image.shape

        # Create a copy of the image to store the thresholded result
        thresholded_image = np.zeros_like(self.image)

        # Iterate over blocks
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Adjust block size for the boundary blocks
                current_block = self.image[i:i + block_size, j:j + block_size]

                # Initialize the threshold and previous threshold for convergence
                threshold = 0
                previous_threshold = threshold + 1  # Initialize to ensure loop runs at least once

                # Iterate until convergence
                while abs(previous_threshold - threshold) != 0:
                    previous_threshold = threshold  # Update the previous threshold before updating the current one

                    # Compute mean values of the background and object regions within the block
                    background_mean = np.mean(
                        [current_block[0, 0], current_block[0, -1], current_block[-1, 0], current_block[-1, -1]])
                    object_mean = np.mean(current_block[1:-1, 1:-1])

                    # Compute local threshold for the block
                    local_threshold = (background_mean + object_mean) / 2

                # Apply thresholding to the block
                thresholded_block = (current_block > local_threshold) * 255

                # Update the thresholded image with the thresholded block
                thresholded_image[i:i + block_size, j:j + block_size] = thresholded_block

        self.apply_thresholding_on_image(thresholded_image)

    def global_otsu_thresholding(self):
        # Compute histogram
        histogram, _ = np.histogram(self.image.flatten(), bins=256, range=(0, 256))

        # Total number of pixels
        total_pixels = self.image.shape[0] * self.image.shape[1]

        # Initialize variables
        max_sigma = 0
        threshold = 0

        # Iterate through all possible thresholds
        for t in range(256):
            # Background
            background_pixels = np.sum(histogram[:t])
            background_probability = background_pixels / total_pixels
            if background_pixels == 0:
                continue

            # Foreground
            foreground_pixels = np.sum(histogram[t:])
            foreground_probability = foreground_pixels / total_pixels
            if foreground_pixels == 0:
                break

            # Mean background intensity
            background_mean = np.sum(np.arange(t) * histogram[:t]) / background_pixels

            # Mean foreground intensity
            foreground_mean = np.sum(np.arange(t, 256) * histogram[t:]) / foreground_pixels

            # Calculate inter-class variance
            sigma = background_probability * foreground_probability * (background_mean - foreground_mean) ** 2

            # Update threshold if variance is higher
            if sigma > max_sigma:
                max_sigma = sigma
                threshold = t

        # Apply thresholding to the image
        thresholded_image = (self.image > threshold) * 255

        self.apply_thresholding_on_image(thresholded_image)

    def local_otsu_thresholding(self):
        print("otsu")
        block_size = self.block_size_slider.value()
        # Get image dimensions
        height, width = self.image.shape

        # Create a copy of the image to store the thresholded result
        thresholded_image = np.zeros_like(self.image)

        # Iterate over blocks
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Adjust block size for the boundary blocks
                current_block = self.image[i:i + block_size, j:j + block_size]

                # Compute histogram for the current block
                histogram, _ = np.histogram(current_block.flatten(), bins=256, range=(0, 256))
                total_pixels = current_block.shape[0] * current_block.shape[1]

                # Initialize variables for local Otsu thresholding
                max_sigma = 0
                local_threshold = 0

                # Iterate through all possible thresholds
                for t in range(256):
                    # Background
                    background_pixels = np.sum(histogram[:t])
                    background_probability = background_pixels / total_pixels
                    if background_pixels == 0:
                        continue

                    # Foreground
                    foreground_pixels = np.sum(histogram[t:])
                    foreground_probability = foreground_pixels / total_pixels
                    if foreground_pixels == 0:
                        break

                    # Mean background intensity
                    background_mean = np.sum(np.arange(t) * histogram[:t]) / background_pixels

                    # Mean foreground intensity
                    foreground_mean = np.sum(np.arange(t, 256) * histogram[t:]) / foreground_pixels

                    # Calculate inter-class variance
                    sigma = background_probability * foreground_probability * (background_mean - foreground_mean) ** 2

                    # Update local threshold if variance is higher
                    if sigma > max_sigma:
                        max_sigma = sigma
                        local_threshold = t

                # Apply local thresholding to the block
                thresholded_block = (current_block > local_threshold) * 255

                # Update the thresholded image with the thresholded block
                thresholded_image[i:i + block_size, j:j + block_size] = thresholded_block

        self.apply_thresholding_on_image(thresholded_image)

    def global_multi_level_thresholding(self):
        print("global")
        num_thresholds = self.num_thresholds_slider.value()
        # Compute histogram
        histogram, _ = np.histogram(self.image.flatten(), bins=256, range=(0, 256))

        # Total number of pixels
        total_pixels = self.image.shape[0] * self.image.shape[1]

        # Initialize thresholds list
        thresholds = []
        print("before")
        # Iterate until the desired number of thresholds are found
        while len(thresholds) < num_thresholds:
            # Initialize variables for finding next threshold
            max_sigma = 0
            next_threshold = 0

            # Iterate through all possible thresholds
            for t in range(256):
                # Background
                background_pixels = np.sum(histogram[:t])
                background_probability = background_pixels / total_pixels
                if background_pixels == 0:
                    continue

                # Foreground
                foreground_pixels = np.sum(histogram[t:])
                foreground_probability = foreground_pixels / total_pixels
                if foreground_pixels == 0:
                    break

                # Mean background intensity
                background_mean = np.sum(np.arange(t) * histogram[:t]) / background_pixels

                # Mean foreground intensity
                foreground_mean = np.sum(np.arange(t, min(256, len(histogram))) * histogram[t:]) / foreground_pixels

                # Calculate inter-class variance
                sigma = background_probability * foreground_probability * (background_mean - foreground_mean) ** 2

                # Update next threshold if variance is higher
                if sigma > max_sigma:
                    max_sigma = sigma
                    next_threshold = t

            # Add next threshold to the list
            thresholds.append(next_threshold)

            # Update histogram for next iteration (remove pixels assigned to current threshold)
            histogram = histogram[next_threshold:]

        # Apply multi-level thresholding to the image
        thresholded_image = np.zeros_like(self.image)
        for i in range(num_thresholds + 1):
            if i == 0:
                thresholded_image[self.image <= thresholds[i]] = i * (255 // num_thresholds)
            elif i == num_thresholds:
                thresholded_image[self.image > thresholds[i - 1]] = 255
            else:
                thresholded_image[(self.image > thresholds[i - 1]) & (self.image <= thresholds[i])] = i * (
                            255 // num_thresholds)

        self.apply_thresholding_on_image(thresholded_image)

    def local_multi_level_thresholding(self):
        # Get the values from the sliders
        num_thresholds = self.num_thresholds_slider.value()
        block_size = self.block_size_slider.value()

        # Initialize thresholds list
        thresholds = []

        # Iterate over blocks
        for i in range(0, self.image.shape[0], block_size):
            for j in range(0, self.image.shape[1], block_size):
                # Get the block
                block = self.image[i:i + block_size, j:j + block_size]

                # Compute histogram for the block
                block_histogram, _ = np.histogram(block.flatten(), bins=256, range=(0, 256))

                # Initialize variables for finding next threshold
                thresholds_found = 0
                while thresholds_found < num_thresholds:
                    # Initialize variables for finding next threshold
                    max_sigma = 0
                    next_threshold = 0

                    # Iterate through all possible thresholds
                    for t in range(256):
                        # Background
                        background_pixels = np.sum(block_histogram[:t])
                        background_probability = background_pixels / (block_size * block_size)
                        if background_pixels == 0:
                            continue

                        # Foreground
                        foreground_pixels = np.sum(block_histogram[t:])
                        foreground_probability = foreground_pixels / (block_size * block_size)
                        if foreground_pixels == 0:
                            break

                        # Mean background intensity
                        background_mean = np.sum(np.arange(t) * block_histogram[:t]) / background_pixels

                        # Mean foreground intensity
                        foreground_mean = np.sum(
                            np.arange(t, min(256, len(block_histogram))) * block_histogram[t:]) / foreground_pixels

                        # Calculate inter-class variance
                        sigma = background_probability * foreground_probability * (
                                    background_mean - foreground_mean) ** 2

                        # Update next threshold if variance is higher
                        if sigma > max_sigma:
                            max_sigma = sigma
                            next_threshold = t

                    # Add next threshold to the list
                    thresholds.append(next_threshold)

                    # Update histogram for next iteration (remove pixels assigned to current threshold)
                    block_histogram = block_histogram[next_threshold:]

                    # Increment the count of thresholds found
                    thresholds_found += 1

        # Apply multi-level thresholding to the image
        thresholded_image = np.zeros_like(self.image)
        for i in range(num_thresholds + 1):
            if i == 0:
                thresholded_image[self.image <= thresholds[i]] = i * (255 // num_thresholds)
            elif i == num_thresholds:
                thresholded_image[self.image > thresholds[i - 1]] = 255
            else:
                thresholded_image[(self.image > thresholds[i - 1]) & (self.image <= thresholds[i])] = i * (
                            255 // num_thresholds)

        self.apply_thresholding_on_image(thresholded_image)

    def apply_thresholding_on_image(self, thresholded_image):
        resulted_pixmap = ImageConverter.numpy_to_pixmap(thresholded_image)
        scaled_pixmap = resulted_pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.after_thresholding_image.setPixmap(scaled_pixmap)
        self.after_thresholding_image.setScaledContents(False)


######################################## Task 4 RGB to LUV Mapping ############################################

    def rgb_to_xyz(self, rgb_image):
        """
        Convert an RGB image to XYZ color space using transformation matrices.
        """
        # Define the transformation matrix from sRGB to XYZ
        transformation_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])

        # Normalize and convert sRGB to linear RGB
        linear_rgb_image = rgb_image / 255.0
        linear_rgb_image = np.where(
            linear_rgb_image <= 0.04045,
            linear_rgb_image / 12.92,
            ((linear_rgb_image + 0.055) / 1.055) ** 2.4
        )

        # Apply the matrix to convert linear RGB to XYZ
        xyz_image = np.dot(linear_rgb_image, transformation_matrix.T)

        return xyz_image

    def xyz_to_luv(self, xyz_image):
        # Reference white point for D65 illuminant
        ref_white = np.array([0.95047, 1.0, 1.08883])

        # Compute u' and v' for reference white
        u_prime_ref = (4 * ref_white[0]) / (ref_white[0] + 15 * ref_white[1] + 3 * ref_white[2])
        v_prime_ref = (9 * ref_white[1]) / (ref_white[0] + 15 * ref_white[1] + 3 * ref_white[2])

        # Compute XYZ normalized values
        X, Y, Z = xyz_image[..., 0], xyz_image[..., 1], xyz_image[..., 2]
        Xn, Yn, Zn = ref_white

        Y_Yn = Y / Yn
        L = np.where(Y_Yn > 0.008856, (116 * np.cbrt(Y_Yn)) - 16, 903.3 * Y_Yn)

        d = X + 15 * Y + 3 * Z
        u_prime = (4 * X) / d
        v_prime = (9 * Y) / d

        u = 13 * L * (u_prime - u_prime_ref)
        v = 13 * L * (v_prime - v_prime_ref)

        luv_image = np.stack((L, u, v), axis=-1)
        return luv_image

    def luv_to_xyz(self, luv_image):
        # Reference white point for D65 illuminant
        ref_white = np.array([0.95047, 1.0, 1.08883])

        # Extract L, u, and v from LUV image
        L, u, v = luv_image[..., 0], luv_image[..., 1], luv_image[..., 2]

        # Compute u' and v' for reference white
        u_prime_ref = (4 * ref_white[0]) / (ref_white[0] + 15 * ref_white[1] + 3 * ref_white[2])
        v_prime_ref = (9 * ref_white[1]) / (ref_white[0] + 15 * ref_white[1] + 3 * ref_white[2])

        # Calculate Y
        Y = np.where(L > 7.999624799, ((L + 16) / 116) ** 3 * ref_white[1], L / 903.2962962 * ref_white[1])

        # Calculate u' and v'
        u_prime = u / (13 * L) + u_prime_ref
        v_prime = v / (13 * L) + v_prime_ref

        # Calculate X and Z
        X = Y * 9 * u_prime / (4 * v_prime)
        Z = Y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime)

        xyz_image = np.stack((X, Y, Z), axis=-1)
        return xyz_image

    def xyz_to_rgb(self, xyz_image):
        """
        Convert an XYZ image to RGB color space using transformation matrices.
        """
        # Define the inverse transformation matrix from XYZ to sRGB
        inverse_transformation_matrix = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ])

        # Apply the matrix to convert XYZ to linear RGB
        linear_rgb_image = np.dot(xyz_image, inverse_transformation_matrix.T)

        # Convert linear RGB to sRGB with proper gamma correction
        srgb_image = np.where(
            linear_rgb_image <= 0.0031308,
            linear_rgb_image * 12.92,
            1.055 * np.power(linear_rgb_image, 1 / 2.4) - 0.055
        )

        # Clip the values to the [0, 1] range and convert to 8-bit integers
        srgb_image = np.clip(srgb_image, 0, 1) * 255.0

        return srgb_image.astype(np.uint8)

    def rgb_to_luv(self):
        """
        Convert an RGB image to LUV without using OpenCV and display it
        """
        rgb_image = cv2.imread(self.imagePath)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        xyz_image = self.rgb_to_xyz(rgb_image)
        luv_image = self.xyz_to_luv(xyz_image)

        xyz_image2 = self.luv_to_xyz(luv_image)
        rgb_image2 = self.xyz_to_rgb(xyz_image2)

        # luv_image = cv2.cvtColor(luv_image, cv2.COLOR_Luv2RGB)

        self.show_image(
            QImage(luv_image.clip(0, 255).astype(np.uint8).data, luv_image.shape[1], luv_image.shape[0],
                   luv_image.shape[1] * 3, QImage.Format_RGB888), mode='luv')

        self.show_image(
            QImage(rgb_image2.clip(0, 255).astype(np.uint8).data, rgb_image2.shape[1], rgb_image2.shape[0],
                   rgb_image2.shape[1] * 3, QImage.Format_RGB888), mode='luv_from_rgb')
    

    ####################################### Task 4 Segmentation ############################################

    ########################################## K-Means #####################################################
    def get_spinbox_value(self, value):
        self.segmentation_spinbox_value = value

    def apply_k_means(self):
        # Flatten the image to be 2D
        pixels = self.IMG.reshape((-1, 3))
        clusters = self.segmentation_spinbox_value
        centroids = self.initialize_centroids(self.IMG, clusters)
        labels = self.run_kmeans(pixels, centroids)
        self.visualize_image(labels, clusters)

    def initialize_centroids(self, image, n_clusters):
        # Get image dimensions
        height, width = image.shape[:2]
        # Select number of random centroids equal to number of needed clusters
        indices = np.random.choice(height * width, n_clusters, replace=False)
        # Initialize an empty array to store centroids
        centroids = np.empty((n_clusters, 3))
        for i, index in enumerate(indices):
            centroids[i] = image.reshape((-1, 3))[index]
        return centroids.reshape(-1, 3)  # Reshape centroids to (n_clusters, 3)

    def run_kmeans(self, pixels, centroids, max_iterations=300):
        for _ in range(max_iterations):
            # Assign each pixel to the nearest centroid
            labels = self.assign_labels(pixels, centroids)
            # Update centroids
            new_centroids = self.update_centroids(pixels, labels, centroids)
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return labels

    def assign_labels(self, pixels, centroids):
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, pixels, labels, centroids):
        new_centroids = np.zeros_like(centroids)
        for i in range(len(centroids)):
            new_centroids[i] = np.mean(pixels[labels == i], axis=0)
        return new_centroids

    def visualize_image(self, labels, clusters):
        # Map labels to colors
        colors = np.random.randint(0, 255, size=(clusters, 3), dtype=np.uint8)
        # Map labels to colors
        segmented_image = colors[labels]
        # Reshape segmented image to original shape
        segmented_image = segmented_image.reshape(self.IMG.shape)
        resulted_pixmap = ImageConverter.numpy_to_pixmap(segmented_image)
        scaled_pixmap = resulted_pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.after_segmentation_image.setPixmap(scaled_pixmap)
        self.after_segmentation_image.setScaledContents(False)

    ########################################## Mean Shift ##########################################
    def apply_mean_shift(self):
        image = self.IMG
        bandwidth = self.segmentation_spinbox_value
        # Convert the input image from BGR to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        self.run_mean_shift(lab_image,bandwidth)

    def run_mean_shift(self,image,band):
        lab_image=image
        bandwidth=band
        # Compute the mean shift vector for each pixel
        height, width, channels = lab_image.shape
        ms_image = np.zeros((height, width), dtype=np.int32)
        label = 1
        # Iterate over each pixel in the image
        for y in range(height):
            for x in range(width):
                # Check if the pixel has already been assigned a label
                if ms_image[y, x] == 0:
                    ms_vector = np.zeros(channels, dtype=np.float32)
                    pixel_vector = lab_image[y, x].astype(np.float32)
                    count = 0

                    # Perform mean shift clustering
                    while True:
                        prev_ms_vector = ms_vector
                        prev_count = count

                        # Iterate over the neighboring pixels within the bandwidth
                        for i in range(max(y - bandwidth, 0), min(y + bandwidth, height)):
                            for j in range(max(x - bandwidth, 0), min(x + bandwidth, width)):
                                if ms_image[i, j] == 0 and np.linalg.norm(
                                        pixel_vector - lab_image[i, j]) < bandwidth:
                                    ms_vector += lab_image[i, j]
                                    count += 1
                                    ms_image[i, j] = label

                        # Calculate the mean shift vector
                        ms_vector /= count

                        # Check for convergence
                        if np.linalg.norm(ms_vector - prev_ms_vector) < 1e-5 or count == prev_count:
                            break

                    # Assign the label to all pixels in the cluster
                    ms_image[ms_image == label] = count
                    label += 1
        self.assign_labels_to_colors(ms_image,lab_image)
    def assign_labels_to_colors(self,mean_image,original_image):
        # Convert the mean shift labels to a color image using the LAB color space
        unique_labels = np.unique(mean_image)
        n_clusters = len(unique_labels)
        ms_color_image = np.zeros_like(original_image)
        for i, label in enumerate(unique_labels):
            mask = mean_image == label
            ms_color_image[mask] = np.mean(original_image[mask], axis=0)
        self.show_result(ms_color_image)

    def show_result(self,mean_image):
        # Convert the LAB color image back to BGR for visualization
        ms_color_image_bgr = cv2.cvtColor(mean_image.astype(np.uint8), cv2.COLOR_LAB2BGR)
        # Create a QImage from the BGR image data
        height, width, channels = ms_color_image_bgr.shape
        bytes_per_line = channels * width
        q_image = QImage(ms_color_image_bgr.data, width, height, bytes_per_line, QImage.Format_BGR888)

        # Create a QPixmap from the QImage
        resulted_pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = resulted_pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.after_segmentation_image.setPixmap(scaled_pixmap)
        self.after_segmentation_image.setScaledContents(False)

    ########################################## Agglomerative ##########################################


    def merge_closest_clusters(self, clusters):
        # Filter out empty or invalid clusters
        clusters = [cluster for cluster in clusters if len(cluster) > 0]

        # Calculate cluster means
        cluster_means = np.array([np.mean(cluster) for cluster in clusters])

        # Perform hierarchical clustering using linkage
        Z = linkage(cluster_means[:, np.newaxis], method='single')

        # Get the indices of the clusters with the minimum similarity
        min_similarity_idx = np.argmin(Z[:, 2])

        # Merge the clusters with the minimum similarity
        merge_indices = (int(Z[min_similarity_idx, 0]), int(Z[min_similarity_idx, 1]))
        merged_cluster = np.concatenate((clusters[merge_indices[0]], clusters[merge_indices[1]]))
        clusters.pop(max(merge_indices))
        clusters.pop(min(merge_indices))
        clusters.append(merged_cluster)

        return clusters

    def apply_agglomerative(self, image):
        num_clusters = self.segmentation_spinbox_value
        flattened_image = self.image.flatten()
        # print(len(flattened_image))
        clusters = None
        if self.faster_check_box.isChecked():
            # Initialize clusters with initial colors/groups
            clusters = self.initial_clusters(flattened_image, int(0.5*len(flattened_image)))
        else:
            clusters = [[pixel] for pixel in flattened_image]

        while len(clusters) > num_clusters:
            print(len(clusters))
            clusters = self.merge_closest_clusters(clusters)

        segmented_image = np.zeros_like(flattened_image)
        for i, cluster in enumerate(clusters):
            segmented_image[np.isin(flattened_image, cluster)] = i * (255 / (len(clusters) - 1))

        segmented_image = segmented_image.reshape(self.image.shape)
        self.show_image_after_segmentation(segmented_image, self.after_segmentation_image)

    def initial_clusters(self,image_clusters, k):
        """
        Initializes the clusters with k colors by grouping the image pixels based on their values.
        """
        # Calculate the range for pixel values
        pixel_range = np.ptp(image_clusters)

        # Determine the step size for dividing the range into k intervals
        step_size = pixel_range / k

        # Initialize clusters with empty lists
        clusters = [[] for _ in range(k)]

        # Assign pixels to clusters based on their values
        for pixel in image_clusters:
            # Calculate the index of the cluster for the current pixel
            cluster_index = int((pixel - np.min(image_clusters)) / step_size)

            # Ensure that the index falls within the range of clusters
            cluster_index = min(max(cluster_index, 0), k - 1)

            # Add the pixel to the corresponding cluster
            clusters[cluster_index].append(pixel)

        # Filter out empty clusters
        clusters = [cluster for cluster in clusters if len(cluster) > 0]

        return clusters

    ########################################### Region Growing #################################################
    def apply_region_growing(self, image):
        print(image.shape)
        seed_point = [self.region_growing_x_pos, self.region_growing_y_pos]
        threshold = self.segmentation_spinbox_value
        print(threshold)
        segmented_region = self.calculate_region_growing(image, seed_point, threshold)
        print(len(segmented_region))
        segmented_image = self.get_segmented_image(image, segmented_region)
        self.show_image_after_segmentation(segmented_image, self.after_segmentation_image)
    def calculate_region_growing(self, image, seed_point, threshold):
        # Initialize empty list to store region pixels
        region = []
        # Initialize queue for pixels to be visited
        queue = []
        queue.append(seed_point)
        # Get image dimensions
        height, width = image.shape

        while len(queue) > 0:
            # Get pixel from queue
            current_point = queue.pop(0)
            # Add pixel to region
            region.append(current_point)
            # Get pixel value
            value = image[current_point[1], current_point[0]]
            # Get neighbors
            neighbors = self.get_neighbors(current_point)
            # Add neighbors to queue if they are within image bounds and their value is within the threshold
            for neighbor in neighbors:
                if neighbor not in region:
                    if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                        if abs(int(image[neighbor[1], neighbor[0]]) - int(value)) < threshold:
                            queue.append(neighbor)
                            region.append(neighbor)

        return region

    def get_neighbors(self, current_point):
        neighbors = []
        offsets = [-1, 0, 1]

        for dx in offsets:
            for dy in offsets:
                if dx == 0 and dy == 0:
                    continue
                x = current_point[0] + dx
                y = current_point[1] + dy
                neighbors.append((x, y))

        return neighbors

    def get_mouse_coordinates(self, event):
        x = event.pos().x()
        y = event.pos().y()

        print(f"Mouse clicked at coordinates: ({x}, {y})")
        self.region_growing_x_pos = x * 2
        self.region_growing_y_pos = int(y * 1.65)

    def get_segmented_image(self, image, region):
        segmented_image = image.copy()
        for point in region:
            segmented_image[point[1], point[0]] = 250
        return segmented_image

    def show_image_after_segmentation(self, image, label):
        label.clear
        pixmap = ImageConverter.numpy_to_pixmap(image)
        pixmap = pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)


    ####################################### Task 3 Functions ############################################

    def harrisCornerDetector(self):
        k = self.kvalue_spinbox.value()
        thr = self.threshold_spinbox.value()
        start_time = time.time()
        # Calculate Image gradients
        Ixx, Iyy, Ixy = self.getGradients(self.image)

        # Calculate Harris Response
        det = Ixx * Iyy - Ixy ** 2  # product of eigen values
        tr = Ixx + Iyy  # sum of eigen values
        response = det - k * (tr ** 2)

        # Spot corners on original image
        output = self.image_color.copy()
        result = self.drawResponse(response, output, thr)

        end_time = time.time()
        computation_time = end_time - start_time
        self.harris_computation_time_lcd.display(computation_time)

        image_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        resulted_pixmap = ImageConverter.numpy_to_pixmap(image_result)
        scaled_pixmap = resulted_pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.harris_resulted_image.setPixmap(scaled_pixmap)
        self.harris_resulted_image.setScaledContents(False)

    def shiTomasiCornerDetection(self):
        thr = self.threshold_spinbox.value()
        start_time = time.time()
        # Calculate Image gradients
        Ixx, Iyy, Ixy = self.getGradients(self.image)

        # Compute the eigenvalues
        tr_shi = Ixx + Iyy
        det_shi = Ixx * Iyy - Ixy ** 2

        lambda1 = 0.5 * (tr_shi + np.sqrt(tr_shi ** 2 - 4 * det_shi))
        lambda2 = 0.5 * (tr_shi - np.sqrt(tr_shi ** 2 - 4 * det_shi))
        R = np.minimum(lambda1, lambda2)

        # Spot corners on the original image
        output = self.image_color.copy()
        result = self.drawResponse(R, output, thr)

        end_time = time.time()
        computation_time = end_time - start_time
        self.shi_tomasi_computation_time_lcd.display(computation_time)

        image_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        resulted_pixmap = ImageConverter.numpy_to_pixmap(image_result)
        scaled_pixmap = resulted_pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.shi_tomasi_resulted_image.setPixmap(scaled_pixmap)
        self.shi_tomasi_resulted_image.setScaledContents(False)

    def getGradients(self, gray):
        gray_smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        sobel_x = self.filters["sobel_x"]
        sobel_y = self.filters["sobel_y"]
        Ix = convolve2d(gray_smoothed, sobel_x, mode='same', boundary='symm')
        Iy = convolve2d(gray_smoothed, sobel_y, mode='same', boundary='symm')
        Ixx = convolve2d(Ix ** 2, np.ones((3, 3)), mode='same',
                         boundary='symm')  # used in matrix of structure tensor which is related to the eigen values
        Iyy = convolve2d(Iy ** 2, np.ones((3, 3)), mode='same',
                         boundary='symm')  # used in matrix of structure tensor which is related to the eigen values
        Ixy = convolve2d(Ix * Iy, np.ones((3, 3)), mode='same',
                         boundary='symm')  # used in matrix of structure tensor which is related to the eigen values

        return Ixx, Iyy, Ixy

    def drawResponse(self, response, output, thr):
        # Normalize the response values from 0 to 255 only
        response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        print(response)

        # check the harris response for each pixel and decide either it's a corner or not
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if response[i, j] > thr:
                    cv2.circle(output, (j, i), 3, (255, 0, 0), 2)
        return output

    def draw_matches_custom(self, img1, kp1, img2, kp2, matches, thickness=2):

        # (1) Check the number of channels in the images
        if len(img1.shape) == 3:
            channels1 = img1.shape[2]  # colored
        else:
            channels1 = 1  # grayscale

        if len(img2.shape) == 3:
            channels2 = img2.shape[2]
        else:
            channels2 = 1

        # (2)If grayscale, they are converted to color (BGR format) for visualization purposes.
        if channels1 == 1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if channels2 == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # (3) Create a new output image (empty image (out)) that concatenates the two images together
        # and is large enough to place the two input images side by side.
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]

        # The height of the new image is the maximum of the two input images' heights, and the width is the sum of their widths.
        out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

        # (4) Place the first image on the left
        out[:rows1, :cols1] = img1
        # Place the next image to the right of it
        out[:rows2, cols1:] = img2

        # (5) Generate a random color for each match
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(matches))]

        # (6) For each pair of points we have between both images draw circles, then connect a line between them with a unique color
        # For each match tuple
        for match, color in zip(matches, colors):
            # Unpack the tuple matches: (index1, index2, distance)
            # It retrieves the indices (img1_idx, img2_idx) of the matched keypoints in each image.
            img1_idx, img2_idx, _ = match

            # Using these indices, it finds the coordinates of the keypoints (x1, y1 for the first image and x2, y2 for the second).
            # x - columns, y - rows (make sure the keypoints are converted to int)
            (x1, y1) = map(int, kp1[img1_idx].pt)
            (x2, y2) = map(int, kp2[img2_idx].pt)

            # Adjust the position of image2's keypoints
            x2 += cols1

            # A line is drawn between the corresponding keypoints, and a circle is drawn at each keypoint location.
            # A unique color is used for each match for differentiation.
            cv2.circle(out, (x1, y1), 4, color, thickness)
            cv2.circle(out, (x2, y2), 4, color, thickness)
            cv2.line(out, (x1, y1), (x2, y2), color, thickness)

        return out

    def match_descriptors(self, descriptors1, descriptors2, mode="SSD"):
        matches = []
        for i in range(len(descriptors1)):
            best_j = -1
            if mode == "SSD":
                # For SSD, For each descriptor in descriptors1, find the descriptor in descriptors2 with the smallest Euclidean
                # distance (SSD). These pairs are considered matches, and their indices, along with the distance, are stored in
                # the matches list.
                min_dist = float('inf')
                for j in range(len(descriptors2)):
                    dist = np.linalg.norm(descriptors1[i] - descriptors2[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_j = j
                matches.append((i, best_j, min_dist))

            elif mode == "NCC":
                # For NCC, find the match with the highest normalized correlation
                max_corr = float('-inf')
                for j in range(len(descriptors2)):
                    # Normalize descriptors to zero mean and unit variance
                    descriptor1_norm = (descriptors1[i] - np.mean(descriptors1[i])) / np.std(descriptors1[i])
                    descriptor2_norm = (descriptors2[j] - np.mean(descriptors2[j])) / np.std(descriptors2[j])
                    # Use np.dot to calculate normalized cross-correlation
                    corr = np.dot(descriptor1_norm, descriptor2_norm)
                    if corr > max_corr:
                        max_corr = corr
                        best_j = j
                # The pair with the highest dot product (indicating the highest correlation) is considered a match.
                # The indices and correlation values are stored in the matches list.
                matches.append((i, best_j, max_corr))

        return matches

    def match_features_and_visualize(self):
        lines_no = self.matching_lines_slider.value()
    
        start_time = time.time()  # Start timing
    
        # Load images
        image1 = cv2.imread(self.imagePath)
        image2 = cv2.imread(self.imagePath2)
    
        image2_grey = cv2.imread(self.imagePath2, cv2.IMREAD_GRAYSCALE)
    
        keypoints1, descriptors1 = computeKeypointsAndDescriptors(self.image)
        keypoints2, descriptors2 = computeKeypointsAndDescriptors(image2_grey)
        # Initialize the brute force matcher with different norms depending on the combobox index
        if self.features_combobox.currentIndex() == 0:
            mode = "SSD"
    
        elif self.features_combobox.currentIndex() == 1:
            mode = "NCC"
        else:
            raise ValueError("Invalid combobox index.")
    
        # Match descriptors using the custom match_descriptors function
        matches = self.match_descriptors(descriptors1, descriptors2, mode)
    
    
        # Sort matches based on mode , If in SSD mode, matches with smaller distances are considered better, so they are
        # sorted in ascending order. In NCC mode, matches with higher correlations are better, so they are sorted in descending order.
        if mode == "SSD":
            matches_sorted = sorted(matches, key=lambda x: x[2])  # For SSD, sort by distance, ascending
        elif mode == "NCC":
            matches_sorted = sorted(matches, key=lambda x: x[2],
                                    reverse=True)  # For NCC, sort by correlation, descending
    
    
        # Now we need to filter out the matches to only keep the best ones
        # we will take the number of sorted matches based on the slider
        best_matches = matches_sorted[:lines_no]
    
        # Draw custom matches
        matched_image = self.draw_matches_custom(image1, keypoints1, image2, keypoints2, best_matches, thickness=3)
    
        end_time = time.time()  # End timing
        computation_time = end_time - start_time  # Compute total time taken
    
        # Display computation time on the LCD
        self.match_lcd.display(computation_time)
    
        # Convert the matched image to QImage and display it
        matched_image = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        self.show_image(QImage(matched_image.data, matched_image.shape[1], matched_image.shape[0],
                               matched_image.strides[0], QImage.Format_RGB888), mode='match')

    ################################################### Task 2 Functions ###################################
    ################################################# Active Contour ######################################################

    def contour_image(self):
        max_iterations = self.max_iterations_slider.value()
        alpha = self.alpha_slider.value()
        beta = self.beta_slider.value()
        contour_image = self.image.copy()
        # Apply Gaussian blur for noise reduction
        blurred_image = cv2.GaussianBlur(contour_image, (9, 9), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_image = clahe.apply(blurred_image)
        # Perform Canny edge detection
        canny_edges = cv2.Canny(equalized_image, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        contour_edges = cv2.dilate(canny_edges, kernel, iterations=1)
        contour_edges = cv2.erode(contour_edges, kernel, iterations=1)
        contour_edges=cv2.GaussianBlur(contour_edges, (9, 9), 0)
        snake_points = self.initialize_snake(contour_image,200)
        final_contour = self.optimize_contour(np.copy(snake_points), np.copy(contour_edges), alpha, beta, max_iterations, energy_tolerance=1e-5)
        area = self.contour_area(final_contour)
        self.area_value.setText("{:.2f}".format(area))
        perimeter = self.contour_perimeter(final_contour)
        self.perimeter_value.setText("{:.2f}".format(perimeter))
        chain_code = self.contour_to_chain_code(final_contour)
        chain_code_string = ''.join(map(str, chain_code))
        self.chain_code_values.setText(chain_code_string)
        print("Chain Code: ", chain_code)
        # Plot the optimized contour
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(self.image_color, cmap=plt.cm.gray)
        ax.plot(final_contour[:, 1], final_contour[:, 0], '-b', lw=3)
        ax.axis('off')
        # Save the plot to a BytesIO object
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        # Convert the BytesIO object to a QPixmap
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())
        pixmap = pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Set the QPixmap as the pixmap for the QLabel
        self.contoured_image.setPixmap(pixmap)

    def initialize_snake(self, image, num_points=250):
        width, height = image.shape[:2]

        # Calculate radius to cover as much area as possible
        radius_x = (width - 1) / 2
        radius_y = (height - 1) / 2

        # Use the smaller radius to ensure the ellipse fits inside the rectangular image
        radius = min(radius_x, radius_y)

        # Calculate center of the image
        center_x = (width - 1) / 2
        center_y = (height - 1) / 2

        angles = np.linspace(0, 2 * np.pi, num_points)

        # Calculate initial contour points for the ellipse
        contour_points = []
        for angle in angles:
            x = int(center_x + radius_x * np.cos(angle))
            y = int(center_y + radius_y * np.sin(angle))
            contour_points.append((x, y))

        initial_contour = np.array(contour_points, dtype=np.int32)
        return initial_contour

    def convolve(self,image, kernel):
        # Get the dimensions of the image and kernel
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Calculate the padding needed
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Pad the image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        # Initialize the output image
        output = np.zeros_like(image)

        # Perform the convolution
        for i in range(image_height):
            for j in range(image_width):
                output[i, j] = np.sum(padded_image[i:i + kernel_height, j:j + kernel_width] * kernel)

        return output

    def calculate_gradient_magnitude(self, image):
        # Calculate gradient magnitude using Sobel operator
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gradient_x = scipy.signal.convolve2d(image, sobel_x)
        gradient_y = scipy.signal.convolve2d(image, sobel_y)
        gradient_x = scipy.signal.convolve2d(gradient_x, sobel_x)
        gradient_y = scipy.signal.convolve2d(gradient_y, sobel_y)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (7, 7), 20)
        gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (7, 7), 20)

        return gradient_magnitude

    def internal_energy(self, contour, i, alpha, beta):
        # Calculate internal energy based on the curvature of the contour
        x, y = contour[i]
        x_prev, y_prev = contour[i - 1]
        x_next, y_next = contour[(i + 1) % len(contour)]

        angle_prev = np.arctan2(y - y_prev, x - x_prev)
        angle_next = np.arctan2(y_next - y, x_next - x)

        curvature = angle_next - angle_prev

        # Internal energy is a combination of curvature and distance between consecutive points
        dist_prev = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
        dist_next = np.sqrt((x_next - x) ** 2 + (y_next - y) ** 2)

        internal_energy = alpha * (curvature ** 2) + beta * ((dist_prev + dist_next) / 2) ** 2

        return internal_energy

    def move_contour_point(self, x, y, energy_total, image_shape, alpha, beta):
        min_energy = energy_total
        new_x, new_y = x, y

        # Calculate the center of the image
        center_x = image_shape[0] // 2
        center_y = image_shape[1] // 2

        # Iterate over neighboring positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Skip the current position
                if dx == 0 and dy == 0:
                    continue

                # Calculate energy change for moving to the neighboring position
                new_x_temp, new_y_temp = x + dx, y + dy
                energy_temp = self.internal_energy([(new_x_temp, new_y_temp)], 0, alpha, beta) + self.external_energy[
                    new_x_temp, new_y_temp]

                # Calculate penalty term for moving away from the center of the image
                distance_to_center = np.sqrt((new_x_temp - center_x) ** 2 + (new_y_temp - center_y) ** 2)
                penalty_weight = 0.7
                center_penalty = penalty_weight * distance_to_center

                # Apply Gaussian blur to the penalty term for smoothing
                sigma = 1.0
                smoothed_center_penalty = scipy.ndimage.gaussian_filter(center_penalty, sigma)

                # Apply penalty term to the energy calculation
                energy_temp += smoothed_center_penalty

                # Update if energy reduced
                if energy_temp < min_energy:
                    min_energy = energy_temp
                    new_x, new_y = new_x_temp, new_y_temp

        return new_x, new_y

    def optimize_contour(self, initial_contour, image, alpha, beta, max_iterations=50, energy_tolerance=1e-5):
        self.external_energy = self.calculate_gradient_magnitude(image)
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.ion()  # Turn on interactive mode

        converged = False
        iteration = 0
        while not converged and iteration < max_iterations:
            converged = True
            energy_change = 0
            for i, (x, y) in enumerate(initial_contour):
                energy_total = self.internal_energy(initial_contour, i, alpha, beta) + self.external_energy[x, y]
                new_x, new_y = self.move_contour_point(x, y, energy_total, image.shape, alpha, beta)
                if new_x != x or new_y != y:
                    converged = False
                    initial_contour[i] = (new_x, new_y)
                    energy_change += abs(energy_total - self.internal_energy(initial_contour, i, alpha, beta) - self.external_energy[x, y])

            # Plot the current contour overlaying the original image
            ax.clear()
            ax.set_title('Snake Contour Visualization')
            ax.imshow(self.image, cmap=plt.cm.gray)
            ax.plot(initial_contour[:, 1], initial_contour[:, 0], '-b', lw=3)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, image.shape[1], image.shape[0], 0])
            plt.pause(0.1)  # Pause for a short duration to allow for visualization
            iteration += 1
            if energy_change < energy_tolerance:
                converged = True

            # Ensure that the initial and final contour points are within the image boundaries
            initial_contour[:, 0] = np.clip(initial_contour[:, 0], 0, image.shape[0] - 1)
            initial_contour[:, 1] = np.clip(initial_contour[:, 1], 0, image.shape[1] - 1)

        plt.ioff()  # Turn off interactive mode after visualization
        plt.show()
        return initial_contour

    def contour_area(self, contour_points):
        area = 0.0
        num_points = len(contour_points)
        j = num_points - 1
        for i in range(num_points):
            area += (contour_points[j][0] + contour_points[i][0]) * (contour_points[j][1] - contour_points[i][1])
            j = i
        return abs(area / 2.0)

    def contour_perimeter(self, contour_points):
        distance_sum = 0
        num_points = len(contour_points)
        for i in range(num_points):
            next_point = (i + 1) % num_points
            distance = np.sqrt((contour_points[next_point][0] - contour_points[i][0]) ** 2 + (
                        contour_points[next_point][1] - contour_points[i][1]) ** 2)
            distance_sum += distance
        return distance_sum

    def contour_to_chain_code(self, contour):
        chain_code = []
        num_points = len(contour)

        # Compute chain code for each pair of adjacent points
        for i in range(num_points):
            point1 = contour[i]
            point2 = contour[(i + 1) % num_points]  # Wrap around for the last point

            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]

            # Compute the differential chain code based on dx and dy
            if dx == 0:
                if dy > 0:
                    chain_code.append(0)
                else:
                    chain_code.append(4)
            elif dx > 0:
                if dy == 0:
                    chain_code.append(2)
                elif dy > 0:
                    chain_code.append(1)
                else:
                    chain_code.append(7)
            else:
                if dy == 0:
                    chain_code.append(6)
                elif dy > 0:
                    chain_code.append(3)
                else:
                    chain_code.append(5)

        return chain_code
    ######################################### End of task 2 Functions #################################

    ##################################### Canny Edge Detection #################
    def canny_edge_detection(self, image, low_threshold, high_threshold):

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Step 1: Compute gradients using Sobel operator
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # Step 2: Compute gradient magnitude and angle
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        gradient_direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

        # Step 3: Non-maximum suppression
        suppressed = self.non_max_suppression(gradient_magnitude, gradient_direction)

        # Step 4: Double thresholding
        thresholded = self.double_threshold(suppressed, low_threshold, high_threshold)

        # Step 5: Edge tracking by hysterisis
        edges = self.edge_tracking_by_hysteresis(thresholded, low_threshold, high_threshold)

        return edges.astype(np.uint8) * 255

    def non_max_suppression(self, gradient_magnitude, gradient_direction):
        rows, cols = gradient_magnitude.shape
        suppressed = np.zeros_like(gradient_magnitude)
        angle = gradient_direction % 180

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                q = 255
                r = 255

                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                # Angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                # Angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                # Angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    suppressed[i, j] = gradient_magnitude[i, j]
                else:
                    suppressed[i, j] = 0

        return suppressed

    def double_threshold(self, image, low_threshold, high_threshold):
        high_indices = image >= high_threshold
        low_indices = (image >= low_threshold) & (image < high_threshold)
        thresholded = np.zeros_like(image)
        thresholded[high_indices] = 255
        thresholded[low_indices] = 127
        return thresholded

    def edge_tracking_by_hysteresis(self, image, low_threshold, high_threshold):
        rows, cols = image.shape
        weak = 127

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if image[i, j] == weak:
                    if (
                            (image[i - 1, j - 1] == 255) or
                            (image[i - 1, j] == 255) or
                            (image[i - 1, j + 1] == 255) or
                            (image[i, j - 1] == 255) or
                            (image[i, j + 1] == 255) or
                            (image[i + 1, j - 1] == 255) or
                            (image[i + 1, j] == 255) or
                            (image[i + 1, j + 1] == 255)
                    ):
                        image[i, j] = 255
                    else:
                        image[i, j] = 0

        strong_indices = image == 255
        image = np.zeros_like(image)
        image[strong_indices] = 255

        return image

    def hide_canny(self):
        if self.type_combobox.currentText() == "canny":
            self.alignment_combobox.hide()
        else:
            self.alignment_combobox.show()

    #################################################################################################################################

    ######################################### Ellipses, Lines and Circles Detection #################################################
    def handle_detection(self):
        selected_index = self.detection_combobox.currentIndex()

        if selected_index == 0:  # Detect lines
            # Perform line detection
            self.detect_lines()

        elif selected_index == 1:  # Detect circles
            # Perform circle detection
            self.detect_circles()
        elif selected_index == 2:  # Detect ellipses
            # Perform ellipse detection
            self.detect_ellipses()

    def find_lines(self, edges, rho=1, theta=np.pi / 180, threshold=100):
        height, width = edges.shape
        max_rho = int(np.sqrt(height ** 2 + width ** 2))
        rhos = np.arange(-max_rho, max_rho + 1, rho)
        thetas = np.deg2rad(np.arange(0, 180, theta))
        num_thetas = len(thetas)
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        accumulator = np.zeros((2 * len(rhos), num_thetas), dtype=np.uint64)
        y_indices, x_indices = np.nonzero(edges)
        for i in range(num_thetas):
            rho_values = x_indices * cos_thetas[i] + y_indices * sin_thetas[i]
            rho_values += max_rho
            rho_values = rho_values.astype(np.int64)
            np.add.at(accumulator[:, i], rho_values, 1)
        candidates_indices = np.argwhere(accumulator >= threshold)
        candidate_values = accumulator[candidates_indices[:, 0], candidates_indices[:, 1]]
        sorted_indices = np.argsort(candidate_values)[::-1][: len(candidate_values)]
        candidates_indices = candidates_indices[sorted_indices]
        return candidates_indices, rhos, thetas

    def draw_lines(self, image, candidates_indices, rhos, thetas):
        for rho_idx, theta_idx in candidates_indices:
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image

    def detect_lines(self):
        image = cv2.imread(self.imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        candidates_indices, rhos, thetas = self.find_lines(edges, rho=1, theta=np.pi / 180, threshold=100)
        result = self.draw_lines(image, candidates_indices, rhos, thetas)

        image_with_lines = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        qimage = QImage(image_with_lines.data, image_with_lines.shape[1], image_with_lines.shape[0],
                        image_with_lines.shape[1] * image_with_lines.shape[2],
                        QImage.Format_RGB888 if image_with_lines.shape[2] > 1 else QImage.Format_Grayscale8)
        # Assuming `show_image` is implemented elsewhere in this class
        self.show_image(qimage, mode="detection")

    def custom_hough_circles(self, image, dp=1, min_dist=50, param1=200, param2=30, min_radius=0, max_radius=0):
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)

        # Initialize accumulator array to store circle candidates
        accumulator = np.zeros((image.shape[0] + 2 * max_radius, image.shape[1] + 2 * max_radius), dtype=np.uint64)

        # Initialize list to store detected circles
        circles = []
        l_space = np.linspace(0, 2 * np.pi, 100)  # Increased the resolution for better circle fitting

        # Iterate over each pixel in the edge image
        for y in range(edges.shape[0]):
            for x in range(edges.shape[1]):
                if edges[y, x] != 0:  # If it's an edge pixel
                    # Try different radii for circles
                    for r in range(min_radius, max_radius + 1):
                        # For each radius, generate a circle candidate
                        for theta in l_space:
                            b = int(x - r * np.cos(theta)) + max_radius
                            a = int(y - r * np.sin(theta)) + max_radius
                            accumulator[a, b] += 1  # Increment the accumulator

        # Find circle candidates with enough support
        # Ensure to search for circles within the original image dimensions in the accumulator
        for y in range(max_radius, accumulator.shape[0] - max_radius):
            for x in range(max_radius, accumulator.shape[1] - max_radius):
                if accumulator[y, x] >= param1:  # Check against the threshold
                    # Local maxima check to ensure one vote per candidate circle (non-maximum suppression)
                    neigh_x, neigh_y = np.meshgrid(np.arange(-max_radius, max_radius + 1, 1),
                                                   np.arange(-max_radius, max_radius + 1, 1))
                    if np.all(accumulator[y, x] >= accumulator[y + neigh_y, x + neigh_x]):
                        circles.append((x - max_radius, y - max_radius,
                                        accumulator[y, x]))  # Store the circle center and vote count

        # Only keep the circles with the most votes (i.e., highest likelihood of being a circle)
        circles = sorted(circles, key=lambda x: x[2], reverse=True)
        confirmed_circles = []

        for circle in circles:
            if len(confirmed_circles) == 0 or min(
                    [np.linalg.norm(np.array(circle[:2]) - np.array(c[:2])) for c in confirmed_circles]) > min_dist:
                confirmed_circles.append(circle)

        average_radius = (min_radius + max_radius) // 2
        confirmed_circles = [(c[0], c[1], average_radius) for c in confirmed_circles]

        return np.array(confirmed_circles)

    def detect_circles(self):
        # Load the image
        image = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur the image to reduce noise and improve detection
        gray_blurred = cv2.medianBlur(gray, 5)

        # Apply Hough Circle Transform
        circles = self.custom_hough_circles(gray_blurred, dp=1, min_dist=50, param1=225, param2=50, min_radius=50,
                                            max_radius=80)

        # Draw circles on the original image
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles:  # Iterate over circles without indexing
                x, y, radius = circle
                # Draw the outer circle
                cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        # return image
        image_with_circles = image
        # Convert the image to a QImage object
        qimage = QImage(image_with_circles.data, image_with_circles.shape[1], image_with_circles.shape[0],
                        image_with_circles.shape[1] * image_with_circles.shape[2],
                        QImage.Format_RGB888 if image_with_circles.shape[2] > 1 else QImage.Format_Grayscale8)

        # Show the image with detected circles using the show_image function
        self.show_image(qimage, mode="detection")

    def detect_ellipses(self):
        # Load the image
        image = cv2.imread(self.imagePath)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Step 2: Fit ellipses directly to the edges
        ellipses = self.fit_ellipses_directly(edges)

        # Draw the detected ellipses on the original image
        for ellipse in ellipses:
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)

        # return image
        # Convert BGR image to RGB
        image_with_ellipses = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a QImage object
        qimage = QImage(image_with_ellipses.data, image_with_ellipses.shape[1], image_with_ellipses.shape[0],
                        image_with_ellipses.shape[1] * image_with_ellipses.shape[2],
                        QImage.Format_RGB888 if image_with_ellipses.shape[2] > 1 else QImage.Format_Grayscale8)

        # Show the image with detected ellipses using the show_image function
        self.show_image(qimage, mode="detection")

    def fit_ellipses_directly(self, edges):
        # Find indices of non-zero elements (edges) in the edge image
        nonzero_indices = np.column_stack(np.nonzero(edges))

        # Fit ellipses directly to the non-zero elements
        ellipses = []
        if len(nonzero_indices) >= 5:
            # Convert indices to points in (x, y) format
            points = nonzero_indices[:, ::-1]  # Swap columns to get (x, y) format
            ellipse = cv2.fitEllipse(points)
            ellipses.append(ellipse)
        return ellipses

    ################################################ End of Task 2 Functions ###############################################################

    def normalise_image(self):
        normalized_image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the NumPy array to QImage
        normalized_qimage = QImage(normalized_image.data, normalized_image.shape[1], normalized_image.shape[0],
                                   normalized_image.shape[1], QImage.Format_Grayscale8)

        # Convert QImage to QPixmap
        normalized_pixmap = QPixmap.fromImage(normalized_qimage)
        normalized_pixmap = normalized_pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Display the normalized image in QLabel
        self.processed_image.setPixmap(normalized_pixmap)
        self.processed_image.setScaledContents(False)
        hist, bins = np.histogram(normalized_image.flatten(), bins=256, range=[0, 256])

        # Normalize histogram values
        hist = hist / hist.sum()
        self.draw_dist(self.processed_image_distribution, hist, bins)

    def hist_equalize(self):
        # Perform histogram equalization
        equalized_image = cv2.equalizeHist(self.image)

        # Convert the NumPy array to QImage
        equalized_qimage = QImage(equalized_image.data, equalized_image.shape[1], equalized_image.shape[0],
                                  equalized_image.shape[1], QImage.Format_Grayscale8)

        # Convert QImage to QPixmap
        equalized_pixmap = QPixmap.fromImage(equalized_qimage)
        equalized_pixmap = equalized_pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Display the equalized image in QLabel
        self.processed_image.setPixmap(equalized_pixmap)
        self.processed_image.setScaledContents(False)
        hist, bins = np.histogram(equalized_image.flatten(), bins=256, range=[0, 256])

        # Normalize histogram values
        hist = hist / hist.sum()
        self.draw_dist(self.processed_image_distribution, hist, bins)

    def draw_curves(self):
        hist, bins = np.histogram(self.image.flatten(), bins=256, range=[0, 256])

        # Normalize histogram values
        hist = hist / hist.sum()
        # Plot distribution curve

        # self.distribution_curve_graph = pg.PlotWidget
        self.dist_curve_widget.clear()
        # self.curve_widget.plot([1,2,3],[5,6,7])
        # print(len(bins[:-1]), len(hist))

        self.draw_dist(self.dist_curve_widget, hist, bins)
        self.draw_dist(self.original_image_distribution, hist, bins)
        # Plot histogram
        self.hist_curve_widget.clear()
        self.hist_curve_widget.plot(bins[1:-1], hist[1:], pen='r')
        self.hist_curve_widget.setTitle("Histogram")
        self.hist_curve_widget.setLabel('left', "Frequency")
        self.hist_curve_widget.setLabel('bottom', "Pixel Intensity")

    def draw_dist(self, widget, hist, bins):
        widget.clear()
        widget.plot(bins[1:], hist[1:], stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))
        widget.setTitle("Distribution Curve")
        widget.setLabel('left', "Frequency")
        widget.setLabel('bottom', "Pixel Intensity")

    def show_image(self, image, mode):
        # Convert the QImage to a QPixmap once, outside the loop
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(700, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if mode == '1':
            labels = [
                self.original_image,
                self.original_image_tab2,
                self.original_image_tab4,
                self.original_image_tab5,
                self.original_image_tab7,
                self.original_image_tab8,
                self.original_image_tab9,
                self.harris_original_image,  ## task 3
                self.orginal_image_tab10,
                self.sift_og_image,
                self.before_thresholding_image,  ## Task 4
                self.before_segmentation_image,
                self.orginal_image_luv
            ]
            # Iterate over the list and set the pixmap on each QLabel
            for label in labels:
                scaled_pixmap = pixmap
                label.setPixmap(scaled_pixmap)
                label.setScaledContents(False)

        if mode == '2':
            self.second_original_image_tab7.setPixmap(pixmap)
            self.second_original_image_tab7.setScaledContents(False)
            self.second_original_image_tab10.setPixmap(pixmap)
            self.second_original_image_tab10.setScaledContents(False)

        if mode == 'noisy':
            scaled_pixmap = pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.noisy_image.setPixmap(scaled_pixmap)
            self.noisy_image.setScaledContents(False)

        if mode == 'filter':
            scaled_pixmap = pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.filtered_image_tab1.setPixmap(scaled_pixmap)
            self.filtered_image_tab1.setScaledContents(False)

        if mode == 'detection':
            scaled_pixmap = pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.detected_shapes_image.setPixmap(scaled_pixmap)
            self.detected_shapes_image.setScaledContents(False)
            # self.detected_shapes_image.setScaledContents(True)

        if mode == 'match':
            scaled_pixmap = pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.matched_image.setPixmap(scaled_pixmap)
            self.matched_image.setScaledContents(True)

        if mode == 'luv':
            scaled_pixmap = pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.luv_image.setPixmap(scaled_pixmap)
            self.luv_image.setScaledContents(True)

        if mode == 'luv_from_rgb':
            scaled_pixmap = pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.luv_to_rgb_image.setPixmap(scaled_pixmap)
            self.luv_to_rgb_image.setScaledContents(True)

    def apply_fourier_transform(self, mode, path):
        if path:
            # Read the image using OpenCV
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # Perform Fourier transform
            f_transform = np.fft.fft2(image)
            f_transform_shifted = np.fft.fftshift(f_transform)
            high_pass = self.filter_image(f_transform_shifted, 'h')
            low_pass = self.filter_image(f_transform_shifted, 'l')

            low_pixmap = ImageConverter.numpy_to_pixmap(low_pass)
            low_pixmap = low_pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            high_pixmap = ImageConverter.numpy_to_pixmap(high_pass)
            high_pixmap = high_pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if mode == '1':
                self.high_pass_original_image.setPixmap(high_pixmap)
                self.low_pass_original_image.setPixmap(low_pixmap)
                self.filtered_images_array[0][0] = high_pass
                self.filtered_images_array[0][1] = low_pass
            else:
                self.second_high_pass_original_image.setPixmap(high_pixmap)
                self.second_low_pass_original_image.setPixmap(low_pixmap)
                self.filtered_images_array[1][0] = high_pass
                self.filtered_images_array[1][1] = low_pass

    def filter_image(self, image, filter):
        rows, columns = image.shape
        crows, ccolumns = rows // 2, columns // 2
        mask = np.ones((rows, columns), np.uint8)
        center = [crows, ccolumns]
        radius = 15
        x, y = np.ogrid[:rows, :columns]
        if filter == 'h':
            mask_area = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius * radius
        else:
            mask_area = ((x - center[0]) ** 2 + (y - center[1]) ** 2) > radius * radius
        mask[mask_area] = 0
        filtered_image = image * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(filtered_image)).real
        return img_back

    def create_hybrid_image(self, high_pass_image1, low_pass_image2):
        # Combine high-frequency components of the first image with low-frequency components of the second image
        hybrid_image = high_pass_image1 + low_pass_image2

        # Display or save the hybrid image
        hybrid_pixmap = ImageConverter.numpy_to_pixmap(hybrid_image)
        hybrid_pixmap = hybrid_pixmap.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_image.setPixmap(hybrid_pixmap)

    def choose_hybrid_mood(self, index):
        selected_mood = self.first_image_combobox.currentText()
        if selected_mood == "High Pass":
            self.second_image_label2.setText("Low Pass")
            self.create_hybrid_image(self.filtered_images_array[0][0], self.filtered_images_array[1][1])
        else:
            self.second_image_label2.setText("High Pass")
            self.create_hybrid_image(self.filtered_images_array[0][1], self.filtered_images_array[1][0])

    def set_default_values(self):
        default_value = 128
        self.tvalue_slider.setMinimum(1)
        self.tvalue_slider.setMaximum(255)
        self.tvalue_slider.setValue(default_value)
        self.global_radiobutton.setChecked(True)

    def applyThresholding(self):
        image = self.image
        original_image_array = np.array(image)
        threshold_value = self.tvalue_slider.value()
        window_size = self.tvalue_slider.value()
        constant_offset = self.cvalue_slider.value()
        if self.global_radiobutton.isChecked():
            thresholded_image = self.globalThresholding(original_image_array, threshold_value)
        else:
            thresholded_image = self.localThresholding(original_image_array, window_size, constant_offset)

        thresholded_qimage = QImage(thresholded_image.data, thresholded_image.shape[1], thresholded_image.shape[0],
                                    thresholded_image.shape[1], QImage.Format_Grayscale8)
        thresholded_qimage = thresholded_qimage.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.after_threshold_image.setPixmap(QPixmap.fromImage(thresholded_qimage))

    def localThresholding(self, image_array, window_size, constant_offset):
        rows, cols = image_array.shape
        thresholded_image = np.zeros_like(image_array)
        for i in range(rows):
            for j in range(cols):
                start_row = max(0, i - window_size // 2)
                end_row = min(rows, i + window_size // 2 + 1)
                start_col = max(0, j - window_size // 2)
                end_col = min(cols, j + window_size // 2 + 1)
                window_mean = np.mean(image_array[start_row:end_row, start_col:end_col])
                local_threshold = window_mean - constant_offset

                if image_array[i, j] < local_threshold:
                    thresholded_image[i, j] = 0
                else:
                    thresholded_image[i, j] = 255

        return thresholded_image

    def globalThresholding(self, image_array, threshold_value):
        thresholded_image = np.zeros_like(image_array)
        rows, cols = image_array.shape
        for i in range(rows):
            for j in range(cols):
                if image_array[i, j] < threshold_value:
                    thresholded_image[i, j] = 0
                else:
                    thresholded_image[i, j] = 255
        return thresholded_image

    def toggleGlobalThresholdingWidgets(self, checked):
        if checked:
            self.cvalue_slider.hide()
            self.cvalue_label.hide()
            self.cvalue_lcd.hide()
            self.tvalue_label.setText("Threshold Value")
            self.tvalue_slider.setMinimum(1)
            self.tvalue_slider.setMaximum(255)

            if self.last_global_threshold_value is not None:
                self.tvalue_slider.setValue(self.last_global_threshold_value)
            else:
                self.tvalue_slider.setValue(128)

    def toggleLocalThresholdingWidgets(self, checked):
        if checked:
            self.cvalue_slider.show()
            self.cvalue_label.show()
            self.cvalue_lcd.show()
            self.tvalue_label.setText("Window Size")
            self.tvalue_slider.setMinimum(1)
            self.tvalue_slider.setMaximum(30)
            self.cvalue_slider.setMinimum(0)
            self.cvalue_slider.setMaximum(20)
            self.cvalue_slider.setValue(10)

            if self.last_local_threshold_value is None:
                self.tvalue_slider.setValue(15)
            else:
                self.tvalue_slider.setValue(self.last_local_threshold_value)
                self.cvalue_slider.setValue(10)

    def initializeThresholding(self):
        self.tvalue_slider.valueChanged.connect(self.applyThresholding)
        self.global_radiobutton.toggled.connect(self.applyThresholding)

    def add_salt_and_pepper_noise(self):
        if self.imagePath:
            original_image = cv2.imread(self.imagePath)
            salt_value = self.salt_slider.value()
            pepper_value = self.pepper_slider.value()

            # Generate salt and pepper noise
            noisy_image = original_image.copy()
            salt_mask = np.random.random(original_image.shape[:2]) < salt_value / 100
            pepper_mask = np.random.random(original_image.shape[:2]) < pepper_value / 100
            # Add salt noise
            noisy_image[salt_mask] = [255, 255, 255]
            # Add pepper noise
            noisy_image[pepper_mask] = [0, 0, 0]

            # Convert BGR to RGB for displaying
            noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
            self.noisy_image_data = noisy_image_rgb

            # Display the noisy image in the "noisy_image" QLabel
            self.show_image(QImage(noisy_image_rgb.data, noisy_image_rgb.shape[1], noisy_image_rgb.shape[0],
                                   noisy_image_rgb.shape[1] * 3, QImage.Format_RGB888), mode='noisy')

    def add_gaussian_noise(self):
        if self.imagePath:
            original_image = cv2.imread(self.imagePath)
            mean = self.mean_slider.value()
            variance = self.variance_slider.value()

            # Generate separate noise for each color channel
            noisy_image = np.zeros_like(original_image)
            for i in range(original_image.shape[2]):  # Iterate over color channels
                gaussian_noise = np.random.normal(mean, variance, original_image[:, :, i].shape).astype(np.uint8)
                noisy_channel = cv2.add(original_image[:, :, i], gaussian_noise)
                noisy_image[:, :, i] = noisy_channel

        # Convert BGR to RGB for displaying
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        self.noisy_image_data = noisy_image_rgb

        # Display the noisy image in the "noisy_image" QLabel
        self.show_image(QImage(noisy_image_rgb.data, noisy_image_rgb.shape[1], noisy_image_rgb.shape[0],
                               noisy_image_rgb.shape[1] * 3, QImage.Format_RGB888), mode='noisy')

    def add_average_noise(self):
        if self.imagePath:
            original_image = cv2.imread(self.imagePath)
            noise_amount = 30

            # Generate average noise
            # This part generates random integers within the range[-noise_amount, noise_amount](inclusive).It creates an array of random integers
            # with the same shape as the original_image.This array represents the noise to be added to the original image.Each pixel
            # in this array will contain a randomly generated noise value within the specified range.
            noise = np.random.randint(-noise_amount, noise_amount + 1, original_image.shape).astype(np.uint8)
            noisy_image = cv2.add(original_image, noise)

        # Convert BGR to RGB for displaying
        noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        self.noisy_image_data = noisy_image_rgb

        # Display the noisy image in the "noisy_image" QLabel
        self.show_image(QImage(noisy_image_rgb.data, noisy_image_rgb.shape[1], noisy_image_rgb.shape[0],
                               noisy_image_rgb.shape[1] * 3, QImage.Format_RGB888), mode='noisy')

    def apply_median_filter(self):
        if self.noisy_image_data is not None:
            # Get kernel size from kernel_slider
            kernel_size = self.kernel_slider.value()
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Assuming noisy_image_data is a numpy array
            img = self.noisy_image_data
            img_height, img_width = img.shape[:2]
            # Create an empty array for the filtered image
            filtered_image = np.zeros_like(img)

            # Calculate the margin (the number of pixels to ignore around the image edges to avoid boundary issues) based on the kernel size.
            margin = kernel_size // 2

            # Handle different image types (grayscale or RGB)
            if len(img.shape) == 2:  # Grayscale image
                channels = 1
            else:
                # If the image is not grayscale, it's assumed to be RGB. In that case, img.shape[2] returns the number of
                # channels, which is typically 3 for RGB images (red, green, and blue channels).
                channels = img.shape[2]

            # it iterates over each channel of the image( if it's RGB) and over each pixel in the image, excluding the margins to avoid boundary issues.
            for channel in range(channels):
                for y in range(margin, img_height - margin):
                    for x in range(margin, img_width - margin):
                        # Extract the kernel window
                        if channels == 1:
                            window = img[y - margin:y + margin + 1, x - margin:x + margin + 1]
                        else:
                            window = img[y - margin:y + margin + 1, x - margin:x + margin + 1, channel]

                        # Find the median of the window and apply it to the filtered image
                        median_value = np.median(window)
                        if channels == 1:
                            filtered_image[y, x] = median_value
                        else:
                            filtered_image[y, x, channel] = median_value

            # Display the filtered image
            self.show_image(
                QImage(filtered_image.data, filtered_image.shape[1], filtered_image.shape[0],
                       filtered_image.shape[1] * channels,
                       QImage.Format_RGB888 if channels > 1 else QImage.Format_Grayscale8), mode='filter')

    def apply_gaussian_filter(self):
        if self.noisy_image_data is not None:
            # Get kernel size from kernel_slider
            kernel_size = self.kernel_slider.value()
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Assuming noisy_image_data is a numpy array
            img = self.noisy_image_data
            img_height, img_width = img.shape[:2]
            # Create an empty array for the filtered image
            filtered_image = np.zeros_like(img)

            # Calculate the margin to avoid boundary issues
            margin = kernel_size // 2

            # Gaussian kernel
            # sigma = 5
            sigma = self.sigma_slider.value()
            gauss_kernel = self.gaussian_kernel(kernel_size, sigma)

            # Handle different image types (grayscale or RGB)
            if len(img.shape) == 2:  # Grayscale image
                channels = 1
            else:
                channels = img.shape[2]

            for channel in range(channels):
                for y in range(margin, img_height - margin):
                    for x in range(margin, img_width - margin):
                        # Extract the kernel window
                        if channels == 1:
                            window = img[y - margin:y + margin + 1, x - margin:x + margin + 1]
                        else:
                            window = img[y - margin:y + margin + 1, x - margin:x + margin + 1, channel]

                        # The Gaussian filter is applied to the window by element-wise multiplication with the Gaussian kernel,
                        # followed by summing up all the resulting values. This operation effectively convolves the kernel with the window.
                        filtered_value = np.sum(window * gauss_kernel)
                        if channels == 1:
                            filtered_image[y, x] = filtered_value
                        else:
                            filtered_image[y, x, channel] = filtered_value

            # Display the filtered image
            self.show_image(
                QImage(filtered_image.data, filtered_image.shape[1], filtered_image.shape[0],
                       filtered_image.shape[1] * channels,
                       QImage.Format_RGB888 if channels > 1 else QImage.Format_Grayscale8), mode='filter')
            # shape tuple have three elements: (height, width, channels).

    def gaussian_kernel(self, size, sigma):
        # Create an empty kernel matrix
        kernel = np.zeros((size, size))

        # Calculate the center of the kernel
        center = size // 2

        # Iterate over each element in the kernel matrix
        for x in range(size):
            for y in range(size):
                # calculate the squared Euclidean distance of the current point (x, y) from the center of the kernel.
                # It's divided by 2 * sigma ** 2, which is a scaling factor based on the sigma value.
                distance = ((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)

                # Calculate the value of the Gaussian function at this point (x, y) using the calculated distance.
                value = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-distance)

                # Assign the value to the kernel matrix
                kernel[x, y] = value

        # Normalize the kernel by dividing each element by the sum of all elements This ensures that the kernel's weights
        # sum up to 1, making it suitable for convolution operations.
        kernel /= np.sum(kernel)

        return kernel

    def apply_average_filter(self):
        if self.noisy_image_data is not None:
            # Get kernel size from kernel_slider
            kernel_size = self.kernel_slider.value()
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Assuming noisy_image_data is a numpy array
            img = self.noisy_image_data
            img_height, img_width = img.shape[:2]
            # Create an empty array for the filtered image
            filtered_image = np.zeros_like(img)

            # Calculate the margin to avoid boundary issues
            margin = kernel_size // 2

            # Handle different image types (grayscale or RGB)
            if len(img.shape) == 2:  # Grayscale image
                channels = 1
            else:
                channels = img.shape[2]

            for channel in range(channels):
                for y in range(margin, img_height - margin):
                    for x in range(margin, img_width - margin):
                        # Extract the kernel window
                        if channels == 1:
                            window = img[y - margin:y + margin + 1, x - margin:x + margin + 1]
                        else:
                            window = img[y - margin:y + margin + 1, x - margin:x + margin + 1, channel]

                        # Find the average of the window and apply it to the filtered image
                        average_value = np.mean(window)
                        if channels == 1:
                            filtered_image[y, x] = average_value
                        else:
                            filtered_image[y, x, channel] = average_value

            # Display the filtered image
            self.show_image(
                QImage(filtered_image.data, filtered_image.shape[1], filtered_image.shape[0],
                       filtered_image.shape[1] * channels,
                       QImage.Format_RGB888 if channels > 1 else QImage.Format_Grayscale8), mode='filter')

    def dis_hist(self):
        img = self.img_rgb
        # print(img.shape)
        red_his, green_his, blue_his = self.rgb_hist_fun(img)
        self.draw_rgb(red_his, green_his, blue_his, 'hist')

    def display_pdf(self):
        img = self.img_rgb
        red_his, green_his, blue_his = self.rgb_hist_fun(img)
        red_cdf, red_pdf = self.calculate_cdf_and_pdf(red_his)
        # print(f"this is the red pdf {red_pdf}")
        green_cdf, green_pdf = self.calculate_cdf_and_pdf(green_his)
        # print(f"this is the green pdf {green_pdf.shape}")
        blue_cdf, blue_pdf = self.calculate_cdf_and_pdf(blue_his)
        # print(f"this is the blue pdf {blue_pdf.shape}")
        self.draw_rgb(red_pdf, green_pdf, blue_pdf)

    def display_cdf(self):
        img = self.img_rgb
        red_his, green_his, blue_his = self.rgb_hist_fun(img)
        red_cdf, red_pdf = self.calculate_cdf_and_pdf(red_his)
        green_cdf, green_pdf = self.calculate_cdf_and_pdf(green_his)
        blue_cdf, blue_pdf = self.calculate_cdf_and_pdf(blue_his)
        self.draw_rgb(red_cdf, green_cdf, blue_cdf)

    def rgb_hist_fun(self, img):
        # print(f"this is the rgb hist {img}")
        pixels_2d = img.reshape(-1, 3)
        red_his = pixels_2d[:, 0]
        green_his = pixels_2d[:, 1]
        blue_his = pixels_2d[:, 2]
        return red_his, green_his, blue_his

    def calculate_cdf_and_pdf(self, color_array):
        # Flatten the color array
        flat_array = color_array.flatten()
        # Sort the flattened array
        sorted_array = np.sort(flat_array)
        # Calculate the PDF
        pdf, _ = np.histogram(sorted_array, bins=256, range=(0, 255), density=True)
        # Calculate the CDF (cumulative distribution function)
        cdf = np.cumsum(pdf)
        return cdf, pdf

    def draw_rgb(self, red, green, blue, mode='pc'):
        # Calculate histograms
        self.B_hist.clear()
        self.R_hist.clear()
        self.G_hist.clear()
        if mode == 'hist':
            hist_red, bin_edges_1 = np.histogram(red, bins=256, range=(0, 255))
            hist_green, bin_edges_2 = np.histogram(green, bins=256, range=(0, 255))
            hist_blue, bin_edges_3 = np.histogram(blue, bins=256, range=(0, 255))
            self.plot_histogram(hist_red, bin_edges_1, self.R_hist, 'red', mode='hist')
            self.plot_histogram(hist_green, bin_edges_2, self.G_hist, 'green', mode='hist')
            self.plot_histogram(hist_blue, bin_edges_3, self.B_hist, 'blue', mode='hist')
        else:
            hist_red = red
            hist_green = green
            hist_blue = blue
            self.plot_histogram(data=hist_red, widget=self.R_hist, color='red')
            self.plot_histogram(data=hist_green, widget=self.G_hist, color='green')
            self.plot_histogram(data=hist_blue, widget=self.B_hist, color='blue')

    # Plot histograms
    def plot_histogram(self, data, bin_edges=None, widget=None, color=None, mode='pc'):
        # Plot histogram with bin edges
        if mode == 'hist':
            widget.plot(x=bin_edges, y=data, stepMode=True, fillLevel=0, brush=color)
        else:
            widget.plot(data, pen=color)

    # convolve function takes the image and filter as input and returns the convolved image
    def apply_filter(self, image, filter_name):
        filtered_image = []
        # add GaussianBlur
        image = cv2.GaussianBlur(src=image, ksize=(0, 0), sigmaX=1.3, dst=1.3)
        # Get the filter from the dictionary
        filter_x = self.filters[f"{filter_name}_x"]
        filter_y = self.filters[f"{filter_name}_y"]
        # Apply the filter
        output_x = cv2.filter2D(image, cv2.CV_32F, filter_x)
        # add the used filter in the filtered_image list
        filtered_image.append(output_x)
        output_y = cv2.filter2D(image, cv2.CV_32F, filter_y)
        filtered_image.append(output_y)
        # print(f"x {output_x.shape}")
        # print(f"y {output_y.shape}")
        # Combine the outputs
        output = np.sqrt(output_x ** 2 + output_y ** 2)
        # print(output.shape)
        # Normalize the output to the range [0, 255]
        filtered_magnitude = exposure.rescale_intensity(output, in_range='image', out_range=(0, 255)).clip(0,
                                                                                                           255).astype(
            np.uint8)
        # print(filtered_magnitude.shape)
        filtered_image.append(filtered_magnitude)
        return filtered_image

    def choose_filter(self):
        self.filtered_image.clear()
        # Get the selected filter and alignment from the combo boxes
        filter_name = self.type_combobox.currentText()
        alignment = self.alignment_combobox.currentIndex()
        if filter_name == "canny":
            low_threshold = 50
            high_threshold = 150
            image_to_display = self.canny_edge_detection(self.image, low_threshold, high_threshold)
        else:
            # Apply the selected filter
            filtered_image = self.apply_filter(self.image, filter_name)
            image_to_display = filtered_image[alignment]
        # Convert the NumPy array to QImage
        image_to_display = ImageConverter.numpy_to_pixmap(image_to_display)
        image_to_display = image_to_display.scaled(500, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.filtered_image.setPixmap(image_to_display)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
