from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging
import numpy as np

####################
# Global variables #
####################

logger = logging.getLogger(__name__)
float_tolerance = 1e-7

#################
# Main function #
#################

def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    image = image.astype('float32')
    base_image = upsaple_image(image, sigma, assumed_blur)
    gaussian_kernels = gen_sigmas(sigma, num_intervals)
    gaussian_images = pop_gaussian_pyramid(base_image, 4, gaussian_kernels)
    dog_images = pop_DoG_images(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = gen_descriptors(keypoints, gaussian_images)
    return keypoints,descriptors

def upsaple_image(image, sigma, assumed_blur):

    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    # we add th effect of the 1.6 blur at a higher size
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur


def gen_sigmas(sigma, num_intervals):
    # we add more iterations than what we want 1 at the beggning 2 at the end 
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
        
    print(gaussian_kernels)
    return gaussian_kernels

def pop_gaussian_pyramid(image, num_octaves = 4, gaussian_kernels = None):
    
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        # resize the image in half and add it to the beggning 
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
    return gaussian_images

def pop_DoG_images(gaussian_images):

    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            # subtract the image from the one next to it
            dog_images_in_octave.append(subtract(second_image, first_image))
            print(f"Dog images: {dog_images_in_octave}")
        dog_images.append(dog_images_in_octave)
    return dog_images

def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)
    # threshold = 2
    print("Da el threshold: ",threshold)
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints

def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    center_pixel_value = second_subimage[1, 1]  # deh el heya el centre sub image  1,1 el heya centrha b2aa
    # DoG1 =I2−I1
    # DoG2=I3−I2
    # print("el centre:::",center_pixel_value)
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            # betkaren be kol el 1st we 3rd subimage we ba3den el second bekaren be kolo ma3ada [1,1]
            return all(center_pixel_value >= first_subimage) and \
                   all(center_pixel_value >= third_subimage) and \
                   all(center_pixel_value >= second_subimage[0, :]) and \
                   all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return all(center_pixel_value <= first_subimage) and \
                   all(center_pixel_value <= third_subimage) and \
                   all(center_pixel_value <= second_subimage[0, :]) and \
                   all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False



def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2] #bageeb awel we tany we talet soora
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube) #1st derivative 3shan a3raf el intensity
        hessian = computeHessianAtCenterPixel(pixel_cube)  # me7tageen el hessian matrix 3shan ne3raf stable extrema wla gya men noisy to get local intensity curvature
        # stable if the ratio of the largest eigenvalue to the smallest eigenvalue exceeds a certain threshold usin eigenvalues directions
        # print(hessian.shape)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]  #function solves linear eqns of 1st and 2nd derrivatives Hx+b h:hessian,b:gradient
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold: #ensures that the extremum has sufficient contrast to be considered a keypoint.
        # print(hessian.shape)
        xy_hessian = hessian[:2, :2] #only the upper-left 2x2 submatrix (corresponding to the spatial dimensions) is considered.
        xy_hessian_trace = trace(xy_hessian) # da laplacian trace bey2ees intensity surface curvature fe el xy plane
        xy_hessian_det = det(xy_hessian) # A positive determinant indicates a minimum or maximum, while a negative determinant suggests a saddle point.
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            keypoint = KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum) #da el abs value bs 3hsan yebayen el strength
            return keypoint, image_index
    return None

def computeGradientAtCenterPixel(pixel_array):
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])

def computeHessianAtCenterPixel(pixel_array):

    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])


def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36,
                                     peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    print(f"keypoint {keypoint}")
    logger.debug('Computing keypoint orientations...')
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / float32(
        2 ** (octave_index + 1))  # el area around the key point depends on the size of it
    radius = int(round(radius_factor * scale))
    print(f" raduis:{radius}")
    weight_factor = -0.5 / (scale ** 2)  # gradients closer to the keypoint center have a higher influence
    print(f"weight: {weight_factor}")
    smooth_histogram = zeros(num_bins)
    # function to create the hist
    raw_histogram = create_orientation_histogram(radius, keypoint, octave_index, gaussian_image, num_bins)

    # el histogram et3mal for this key point
    print(f"raw Hist {raw_histogram}")
    # smooth  [1, 4, 6, 4, 1] / 16
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) +
                               raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    max_orientation_index = np.argmax(smooth_histogram)
    max_orientation_value = smooth_histogram[max_orientation_index]
    orientation = 360. - (max_orientation_index * 360.) / num_bins
    # quantization
    if abs(orientation - 360.) < float_tolerance:
        orientation = 0
    # add the new orientation
    new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
    keypoints_with_orientations.append(new_keypoint)

    return keypoints_with_orientations


def create_orientation_histogram(radius, keypoint, octave_index, gaussian_image, num_bins):
    y_coor = []
    raw_histogram = zeros(num_bins)
    image_shape = gaussian_image.shape
    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        y_coor.append(region_y)
        if region_y > 0 and region_y < image_shape[0] - 1:  # within the shape of image
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    # can be done by sobel bs kda ashal w asra3
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    # calc mag and orientation
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    print(f"orientation : {gradient_orientation}")
                    # quantize before adding
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += gradient_magnitude

    return raw_histogram


def convertKeypointsToInputImageSize(keypoints):

    converted_keypoints = []
    for keypoint in keypoints:
        # 2alel el size by 0.5
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5

        # calculations 3shan a8yar el lower 8 bits
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)

        converted_keypoints.append(keypoint)

    return converted_keypoints




def unpack(keypoint):
    # Extract the octave and layer information from the keypoint's octave

    octave = keypoint.octave & 255  # geb el last 8 bits
    layer = (keypoint.octave >> 8) & 255  # hat el 8 elle b3dhom 3shan ageb el layer info

    #
    if octave >= 128:
        octave = octave | -128

    # Calculate the scale factor based on the octave value
    if octave >= 0:
        scale = 1 / float32(1 << octave)  # for positive  scale = 1 / 2^octave
    else:
        scale = float32(1 << -octave)  # for negative scale = 2^(-octave)

    return octave, layer, scale
def  pop_desc_vector(row_bin_list, col_bin_list,magnitude_list, orientation_bin_list, num_bins , histogram_tensor):
    for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list,
                                                            orientation_bin_list):
        # Smoothing via trilinear interpolation
        # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
        # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
        row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
        row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
        if orientation_bin_floor < 0:
            orientation_bin_floor += num_bins
        if orientation_bin_floor >= num_bins:
            orientation_bin_floor -= num_bins
        # calculate el contripution of each pixel dpending on the fraction
        c1 = magnitude * row_fraction
        c0 = magnitude * (1 - row_fraction)
        c11 = c1 * col_fraction
        c10 = c1 * (1 - col_fraction)
        c01 = c0 * col_fraction
        c00 = c0 * (1 - col_fraction)
        c111 = c11 * orientation_fraction
        c110 = c11 * (1 - orientation_fraction)
        c101 = c10 * orientation_fraction
        c100 = c10 * (1 - orientation_fraction)
        c011 = c01 * orientation_fraction
        c010 = c01 * (1 - orientation_fraction)
        c001 = c00 * orientation_fraction
        c000 = c00 * (1 - orientation_fraction)

        histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
        histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
        histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
        histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
        histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
        histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
        histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
        histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

    return (histogram_tensor)
def gen_descriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):

    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpack(keypoint)
        gaussian_image = gaussian_images[octave + 1][layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # make the
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     #  half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                # btcheck enha lsa gowa el image
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    # b5le el size to be relvant to the whole image msh el key point bs
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:

                        #bnfs el tre2a elle fo2
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        # bt2ked en el orientation msh akbar mn 360 lw akbar 3ed w kda
                        gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                        weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        histogram_tensor = pop_desc_vector(row_bin_list, col_bin_list,magnitude_list, orientation_bin_list, num_bins , histogram_tensor)

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')