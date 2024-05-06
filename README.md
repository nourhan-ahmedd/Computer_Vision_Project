# Computer Vision Project

## Table of Contents

1. [Filtering and Edge Detection](#filtering-and-edge-detection)
2. [Edge and Boundary Detection](#edge-and-boundary-detection)
3. [Harris Operator](#harris-operator)
4. [SIFT](#sift)
5. [Feature Matching](#feature-matching)
6. [Segmentation](#segmentation)
7. [Thresholding](#thresholding)

## 1. Filtering and Edge Detection

Description: Given standard images (grayscale and color), the following tasks are implemented:

- Adding additive noise to the image (e.g., Uniform, Gaussian, and Salt & Pepper noise).
- Filtering the noisy image using low-pass filters (e.g., Average, Gaussian, and Median filters).
- Detecting edges using various edge detection masks (e.g., Sobel, Roberts, Prewitt, and Canny).
- Drawing histogram and distribution curve.
- Equalizing the image.
- Normalizing the image.
- Performing local and global thresholding.
- Transforming color image to grayscale and plot histograms of R, G, and B channels with distribution functions.
- Applying frequency domain filters (high pass and low pass).
- Creating hybrid images.
    
![Filtering and noise](https://drive.google.com/uc?export=download&id=1vlK6lHjAeqAHO6eGvNAK_FiJUtPx0ZLQ)

![Normalization and hist equalization](https://drive.google.com/uc?export=download&id=1PtLo5rxZh4s2ZHOBa3bSKtmn3IpnLklr)

![Edge detection](https://drive.google.com/uc?export=download&id=1zfA2J4TWo5ArPjbwjM0u2MNomxRtDeoo)

![Global and local thresholding](https://drive.google.com/uc?export=download&id=1nN0qXlbr4n0Z9uGzL7fOpE3WNC6pmRKf)

![Hybrid image](https://drive.google.com/uc?export=download&id=1BxjBJnGqtTPrxVas948IE5104v-0f5XT)

## 2. Edge and Boundary Detection

Description: For given images (grayscale and color), the following tasks are implemented:

- Detecting edges using Canny edge detector.
- Detecting lines, circles, ellipses (if any) and superimposing the detected shapes on the images.
- Initializing contour for a given object and evolve the Active Contour Model (SNAKE) using the greedy algorithm. Represent the output as chain code and compute perimeter and area inside these contours.

![Active contour](https://drive.google.com/uc?export=download&id=1FFqSmoX-oY-DhTHJ70VAdHT6mla6ZtKa)
![Shapes Detection](https://drive.google.com/uc?export=download&id=1JtIdu8Od59UOxQS4jOSwHazRKeiObr9N)


## 3. Harris Operator

Description: For given images (grayscale and color), the following is implemented:
- Extracting unique features using Harris operator and λ-. Report computation times.

![Harris Operator](https://drive.google.com/uc?export=download&id=1XpAM9SLqlw8XfSAwrNPUEa02OH4vQsFA)

## 4. SIFT

Description: For given images (grayscale and color), the following is implemented:
- Generating feature descriptors using Scale Invariant Features (SIFT). Report computation time.

![SIFT](https://drive.google.com/uc?export=download&id=1JaputP6mSr6JBsIDBxKsI2z4oRZhS_VI)

## 5. Feature Matching

Description: For given images (grayscale and color), the following is implemented:
- Matching image set features using Sum of Squared Differences (SSD) and Normalized Cross Correlations. Report matching computation time.

![Feature Matching](https://drive.google.com/uc?export=download&id=1Wj9HT4IU3unuQZvP5RJZxNPGlLl4A-Jv)

## 6. Segmentation

Description: For given images (grayscale), the following is implemented:
- Performing unsupervised segmentation on supplied Gray/Color images using k-means, region growing, agglomerative, and mean shift methods.

![K-Means](https://drive.google.com/uc?export=download&id=1WA95ZsvZBAnacQ0O470d4bB3x9-DPoCU)
![Mean Shift](https://drive.google.com/uc?export=download&id=16FD8WCUAi_GeQ5Li4hy38gpczg4qMZBd)
![Agglomerative](https://drive.google.com/uc?export=download&id=1FHboIGugXG8jSwHpKzm1nxuMsnzOiJaI)

## 7. Thresholding

Description: For given images (grayscale), the following is implemented:
- Threshold supplied grayscale images using optimal thresholding, Otsu, and Multi-level thresholding (more than 2 modes) with 2 modes for each of them (local and global thresholding).

![Optimal Thresholding](https://drive.google.com/uc?export=download&id=1CXm_2P4ZP3Miz0nx2EWpbIMDlohrwon_)
![Otsu Thresholding](https://drive.google.com/uc?export=download&id=1h09a9cauIEDiswbGAUzDN8lvdx0XeUhB)
![Multi-level Thresholding](https://drive.google.com/uc?export=download&id=1F8QxnDP6JQytwmc2SuYedQRqQ2Q9sjrL)

## Contributors

We would like to thank the following individuals for their contributions to this project:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/OmarEmad101">
        <img src="https://github.com/OmarEmad101.png" width="100px" alt="@OmarEmad101">
        <br>
        <sub><b>Omar Emad</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Omarnbl">
        <img src="https://github.com/Omarnbl.png" width="100px" alt="@Omarnbl">
        <br>
        <sub><b>Omar Nabil</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/KhaledBadr07">
        <img src="https://github.com/KhaledBadr07.png" width="100px" alt="@KhaledBadr07">
        <br>
        <sub><b>Khaled Badr</b></sub>
      </a>
    </td>
  </tr> 
  <!-- New Row -->
  <tr>
    <td align="center">
      <a href="https://github.com/nourhan-ahmedd">
        <img src="https://github.com/nourhan-ahmedd.png" width="100px" alt="@nourhan-ahmedd">
        <br>
        <sub><b>Nourhan Ahmed </b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hanaheshamm">
        <img src="https://github.com/hanaheshamm.png" width="100px" alt="@hanaheshamm">
        <br>
        <sub><b>Hana Hesham</b></sub>
      </a>
    </td>
  </tr>
</table>
