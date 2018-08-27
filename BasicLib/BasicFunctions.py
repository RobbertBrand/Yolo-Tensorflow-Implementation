"""
This module contains tools which can be used in many different applications and are simple in essence.
"""

import matplotlib as plt
import matplotlib.pyplot as pyplot
import sklearn.cluster
import glob
import cv2
import numpy as np
import datetime


###################
# GENERAL LIBRARY #
###################


def get_filenames_by_wildcard(file_name_wildcard):
    """Returns a list with filenames matching the wildcard."""
    return glob.glob(file_name_wildcard)


def format_list_of_floats(float_list, length_per_value=6, digit_after_point=4):
    """Returns a formatted string"""
    string = "["
    str_format = " {:" + str(length_per_value) + "." + str(digit_after_point) + "f},"
    for value in float_list:
        string += str_format.format(value)
    string += "]"
    return string


def get_time():
    """Returns a string with the date and time."""
    return datetime.datetime.now().strftime("%I.%M%p on %B %d %Y")


def get_random_color(normalized=False):
    """
    Generates a random color.
    :param normalized: False returns color value's between 0 and 255. True returns color value's between -1.0 and 1.0
    :return: list with 3 random value's between 1 and 255
    """
    if normalized:
        return ((np.random.random((3, )) * 2) - 1).tolist()
    else:
        return np.random.randint(256, size=3).tolist()


def compute_thresholded_count(value_list, threshold_list):
    """
    Counts value's in value_list, higher than threshold value's in threshold_list.
    :param value_list: list of value's
    :param threshold_list: list of thresholds
    :return: list of counts per threshold
    """
    count_per_threshold = []
    for threshold in threshold_list:
        count_per_threshold.append(sum(i > threshold for i in value_list))
    return np.array(count_per_threshold)


def cluster_data(data_list, n_clusters, random_state=0):
    """
    This function can be used to cluster (group) data points which are most alike, in a given amount of clusters. The
    function returns the center point (average) of each data cluster and a nested list. Each inner list of the nested
    list containing the data points of that cluster.
    :param data_list: list of data points in format:
                        datapoint 1        [[feature_1, feature_2, ..., feature_n],
                        datapoint 2         [feature_1, feature_2, ..., feature_n],
                        datapoint n          ...]
    :param n_clusters: sets the amount of clusters
    :param random_state: sets a random number for the group search
    :return: list of cluster center points and a nested list of clustered data points.
            center_point_list in format:
                [[cluster_1_feature_1_centerpoint, cluster_1_feature_2_centerpoint, ...],
                ...,
                [cluster_n_feature_1_centerpoint, cluster_n_feature_2_centerpoint, ...]]

            data_point_clusters_list in format:
                cluster 1 datapoint 1        [ [[feature_1, feature_2, ..., feature_n],
                cluster 1 datapoint 2           [feature_1, feature_2, ..., feature_n],
                                                ...],
                cluster n datapoint 1          [[feature_1, feature_2, ..., feature_n],
                cluster n datapoint 2           [feature_1, feature_2, ..., feature_n],
                                                ...]]
    """
    k_means = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=random_state).fit(data_list)
    center_point_list = k_means.cluster_centers_
    data_point_clusters_list = [[] for _ in range(n_clusters)]
    for lbl, point in zip(k_means.labels_, data_list):
        data_point_clusters_list[lbl].append(point)
    return center_point_list, data_point_clusters_list


def compute_iou(p_x1, p_y1, p_x2, p_y2, l_x1, l_y1, l_x2, l_y2):
    """
    Computes Intersection over Union (iou) over two area's defined by two points each
    :param p_x1: area 1, point 1, x
    :param p_y1: area 1, point 1, y
    :param p_x2: area 1, point 2, x
    :param p_y2: area 1, point 2, y
    :param l_x1: area 2, point 1, x
    :param l_y1: area 2, point 1, y
    :param l_x2: area 2, point 2, x
    :param l_y2: area 2, point 2, y
    :return: iou
    """
    iou = 0.0

    l_x_min = np.minimum(l_x1, l_x2)
    l_y_min = np.minimum(l_y1, l_y2)
    l_x_max = np.maximum(l_x1, l_x2)
    l_y_max = np.maximum(l_y1, l_y2)
    l_area = ((l_x_max - l_x_min) * (l_y_max - l_y_min))

    p_x_min = np.minimum(p_x1, p_x2)
    p_y_min = np.minimum(p_y1, p_y2)
    p_x_max = np.maximum(p_x1, p_x2)
    p_y_max = np.maximum(p_y1, p_y2)
    p_area = ((p_x_max - p_x_min) * (p_y_max - p_y_min))

    overlap_x = np.minimum(l_x_max, p_x_max) - np.maximum(l_x_min, p_x_min)
    overlap_y = np.minimum(l_y_max, p_y_max) - np.maximum(l_y_min, p_y_min)

    if overlap_x > 0 and overlap_y > 0:
        overlap_area = overlap_x * overlap_y
        total_area = l_area + p_area - overlap_area
        iou = overlap_area / total_area
    return iou


def compute_iou_over_bbox_size(bbox_1, bbox_2):
    """
    Determine iou of bboxes with same center point. Note that only bbox width and height are required, since center
    points match.
    :param bbox_1: bbox in format [width, height]
    :param bbox_2: bbox in format [width, height]
    :return: iou as float between 0.0 and 1.0
    """
    area_overlap = min(bbox_1[0], bbox_2[0]) * min(bbox_1[1], bbox_2[1])
    total_area = (bbox_1[0] * bbox_1[1]) + (bbox_2[0] * bbox_2[1]) - area_overlap
    return area_overlap / total_area


def resize_image(img, outp_img_size=(640, 480)):
    """
    Resizes an image to the desired output size.
    :param img: loaded image
    :param outp_img_size: output size in format (width, height)
    :return: resized image
    """
    return cv2.resize(img, outp_img_size)


def resize_bbox_list(label_list, img_w, img_h, outp_img_size):
    """
    Resizes a list of bboxes to the desired output size. If an image is resized, should also the corresponding bboxes be
    resized. The img_w and img_h specify the original image size. Outp_img_size specifies the image size after resizing.
    :param label_list: list of bboxes in format [[[x,y,width,height], bbox_category_id], ...]
    :param img_w: input width of bboxes. This is the original size of the image for which the bbox list is meant.
    :param img_h: input height of bboxes. This is the original size of the image for which the bbox list is meant.
    :param outp_img_size: output size in format (width, height). This is the image size after resizing.
    :return: list of resized bboxes in format [[[x,y,width,height], bbox_category_id], ...]
    """
    resize_factor_x = outp_img_size[0] / img_w
    resize_factor_y = outp_img_size[1] / img_h

    resize_label = []
    for bbox, category in label_list:
        resize_label.append([[bbox[0] * resize_factor_x, bbox[1] * resize_factor_y, bbox[2] * resize_factor_x,
                              bbox[3] * resize_factor_y], category])
    return resize_label


def crop_image(img, crop_window):
    """
    Crops image to given window.
    :param img: loaded image
    :param crop_window: window definition in format [x1, y1, x2, y2]
    :return: cropped image
    """
    return img[crop_window[1]:crop_window[3], crop_window[0]:crop_window[2]]


def crop_bbox(bbox_list, crop_window):
    """
    Crop bboxes in bbox_list to given crop_window
    :param bbox_list: list of bboxes in format [[[x,y,width,height], bbox_category_id], ...]
    :param crop_window: crop window definition in format [x1, y1, x2, y2]
    :return: cropped bbox list in format [[[x,y,width,height], bbox_category_id], ...]
    """
    x1_crop = crop_window[0]
    y1_crop = crop_window[1]
    x2_crop = crop_window[2]
    y2_crop = crop_window[3]

    cropped_bbox_list = []
    for bbox, cat in bbox_list:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]

        if (x1_crop < x1 < x2_crop or x1_crop < x2 < x2_crop or (x1 < x1_crop and x2_crop < x2)) and \
                (y1_crop < y1 < y2_crop or y1_crop < y2 < y2_crop or (y1 < y1_crop and y2_crop < y2)):

            x1 = np.maximum(x1, x1_crop)
            y1 = np.maximum(y1, y1_crop)
            x2 = np.minimum(x2, x2_crop)
            y2 = np.minimum(y2, y2_crop)

            w = x2 - x1
            h = y2 - y1

            x = x1 - x1_crop
            y = y1 - y1_crop

            cropped_bbox_list.append([[x, y, w, h], cat])
    return cropped_bbox_list


def compute_crop_window_size(img_w, img_h, output_shape_ratio):
    """
    Computes the width and height of a crop window, to crop (cut) an image to a given (width / height) ratio in a way
    to lose as little pixels as possible.
    :param img_w: image width in pixels
    :param img_h: image height in pixels
    :param output_shape_ratio: defines the (width / height) ratio of the crop_window. Ratio is computed as following:
                                        output_shape_ratio = output_width / output_height
    :return: ideal_crop_window_width, ideal_crop_window_height
    """
    outp_h = img_h
    outp_w = img_h * output_shape_ratio

    var = np.minimum(img_h / outp_h, img_w / outp_w)
    if var < 1.0:
        outp_h *= var
        outp_w *= var

    return int(outp_w), int(outp_h)


def compute_crop_pix_position(img_w, img_h, outp_size, crop_offset_w=None, crop_offset_h=None):
    """
    Computes the ideal crop size of an image with img_w (width) and img_h (height) to match after cropping the
    "width / height" ratio of the outp_size (output size). After the image is cropped to the size, specified by this
    function, does the image only have to be resized to match the desired output size. The amount of lost pixels and
    image distortion will be reduced to a minimum after cropping and resizing the input image to the specified output
    size.
    :param img_w: width of the "to be cropped" image
    :param img_h: height of the "to be cropped" image
    :param outp_size: a tuple (width, height)
    :param crop_offset_w: Sets the horizontal offset of the cropping window as a factor of the pixels which couldn't fit
                            in the window. The factor will be random when parameter is set to None
    :param crop_offset_h: Sets the vertical offset of the cropping window as a factor of the pixels which couldn't fit
                            in the window. The factor will be random when parameter is set to None
    :return: ideal crop output size in format [x1, y1, x2, y2]
    """
    outp_w = outp_size[0]
    outp_h = outp_size[1]

    if crop_offset_w is None:
        crop_offset_w = np.random.rand()

    if crop_offset_h is None:
        crop_offset_h = np.random.rand()

    outp_ratio = outp_w / outp_h

    crop_w, crop_h = compute_crop_window_size(img_w, img_h, outp_ratio)

    crop_outp_size = [0, 0, 0, 0]
    crop_outp_size[0] = int(crop_offset_w * (img_w - crop_w))
    crop_outp_size[1] = int(crop_offset_h * (img_h - crop_h))
    crop_outp_size[2] = crop_outp_size[0] + crop_w
    crop_outp_size[3] = crop_outp_size[1] + crop_h
    return crop_outp_size


def safe_divide_list(x_list, y_list):
    """
    Divides x_list by y_list. Returns list of division results, where a divide by zero results in a zero.
    :param x_list: list of value's
    :param y_list: list of value,s
    :return: list of x value's divided by y value's. results in 0 value when y is zero.
    """
    result = []
    for x1, y1 in zip(x_list, y_list):
        if y1 != 0.0:
            result.append(x1 / y1)
        else:
            result.append(0.0)
    return result


def safe_divide_list_by_value(x_list, y_value):
    """
    Divides x_list by y_value. Returns list of division results, where a divide by zero results in a zero.
    :param x_list: list of value's
    :param y_value: single value
    :return: list of x value's divided by y value's. results in 0 value when y is zero.
    """
    y_list = np.array([y_value] * len(x_list))
    return safe_divide_list(x_list, y_list)


#####################
# IMAGE LIBRARY
#####################

def show_image(image, label='', vmin=0, vmax=255, bgr=False, blocking=True):
    """
    Displays an image
    :param image: loaded image
    :param label: name
    :param vmin: minimum pixel value
    :param vmax: maximum pixel value
    :param bgr: color in rgb (False) of bgr (True)
    :param blocking: block program execution until window is closed when True
    """
    norm = plt.colors.Normalize(vmin=vmin, vmax=vmax)

    if bgr:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image

    if len(image.shape) == 2:
        cmap = 'gray'
    else:
        cmap = None

    pyplot.figure(figsize=(15, 9))
    pyplot.title(label)
    pyplot.imshow(norm(rgb_image), cmap=cmap)
    if not blocking:
        pyplot.ion()
    else:
        pyplot.ioff()
    pyplot.show()


def normalize_image(image):
    """
    Normalizes an image with pixel value's between 0 and 255 to -1.0 and +1.0.
    :param image: loaded image
    :return: Normalized image
    """
    image = np.array(image, dtype=np.float)
    image /= 127.5
    image -= 1.0
    return image


def load_image(file_location, normalize=False):
    """
    Returns the loaded images from the given file locations.
    :param file_location: sets the image file location
    :param normalize: normalizes pixel values to (-1.0 to 1.0)
    :return: loaded image
    """
    image = cv2.cvtColor(cv2.imread(file_location), cv2.COLOR_BGR2RGB)
    if normalize:
        image = normalize_image(image)
    return image


def save_image(file_location, image, vmin=0, vmax=255, bgr=False):
    """
    Stores an image in given file location
    :param file_location: sets the image file location
    :param image: loaded image
    :param vmin: minimum pixel value
    :param vmax: maximum pixel value
    :param bgr: color in rgb (False) of bgr (True)
    """
    norm = plt.colors.Normalize(vmin=vmin, vmax=vmax)

    if bgr:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image

    if len(image.shape) == 2:
        cmap = 'gray'
    else:
        cmap = None

    pyplot.imsave(file_location, norm(rgb_image), cmap=cmap)


def to_grayscale(img, bgr=False):
    """Returns an image in gray scale"""
    if bgr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def to_hsvscale(img, bgr=False):
    """Returns an image in hsv scale"""
    if bgr:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv


def to_hlsscale(img, bgr=False):
    """Returns an image in hls scale"""
    if bgr:
        hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    else:
        hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hsl


def to_ycrcbscale(img, bgr=False):
    """Returns an image in YCrCb scale"""
    if bgr:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return ycrcb


def to_labscale(img, bgr=False):
    """Returns an image in lab scale"""
    if bgr:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return lab


def to_luvscale(img, bgr=False):
    """Returns an image in lab scale"""
    if bgr:
        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    else:
        luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    return luv


def gaussian_blur(img, kernel_size=5):
    """Applies a Gaussian Noise kernel, blurring the original image"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def optimize_contrast_hist(raw_img):
    """Optimizes image contrast using histogram equalization.

    Returns same image with better contrast"""
    ycrcb_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    y = cv2.equalizeHist(y)
    ycrcb_img = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)


def optimize_contrast_clahe(raw_img):
    """Optimize contrast using Adaptive histogram equalization.

    Returns same image with contrast localy optimized"""
    lab_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(25, 25))
    l = clahe.apply(l)
    lab_img = cv2.merge((l, a, b))

    return cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)


def weighted_img(img1, img2, alfa=0.8, beta=1., gamma=0.):
    """Merges two images based on alfa, beta and gamma inputs.
    `img1` could be a blank image (all black) with lines drawn on it.\n",
    `img2` could be the image before any processing.\n",
    The result image is computed as follows:\n",

    initial_img * alfa + img * beta + gamma\n",
    NOTE: initial_img and img must be the same shape!"""
    return cv2.addWeighted(img2, alfa, img1, beta, gamma)


def detect_edges(image, blur=0, sobel_kernel_size=3, adaptive_threshold_kernel_size=10,
                 adaptive_threshold_negative_offset=-30):
    """Detect edges in image
    'image' contains the images to find edges in
    'blur' set blur level
    'sobel_kernel_size' set edge detection kernel size
    'adaptive_threshold_kernel_size' set the edge detection neighbourhood area size
    'adaptive_threshold_negative_offset' set the threshold value for edge detection in the neighbourhood area

    Returns the single channel image with the pixels value's representing edge certainty.
    """
    process_image = np.copy(image)
    if blur != 0:
        process_image = gaussian_blur(process_image, kernel_size=blur)

    sobelx_process_image = cv2.Sobel(process_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=sobel_kernel_size)
    sobely_process_image = cv2.Sobel(process_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=sobel_kernel_size)

    sobelx_process_image = cv2.convertScaleAbs(sobelx_process_image)
    sobely_process_image = cv2.convertScaleAbs(sobely_process_image)

    sobelx_process_image = cv2.adaptiveThreshold(sobelx_process_image, maxValue=1,
                                                 adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 thresholdType=cv2.THRESH_BINARY,
                                                 blockSize=adaptive_threshold_kernel_size,
                                                 C=adaptive_threshold_negative_offset)

    sobely_process_image = cv2.adaptiveThreshold(sobely_process_image, maxValue=1,
                                                 adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 thresholdType=cv2.THRESH_BINARY,
                                                 blockSize=adaptive_threshold_kernel_size,
                                                 C=adaptive_threshold_negative_offset)

    process_image = sobelx_process_image + sobely_process_image

    return sobelx_process_image, sobely_process_image, process_image
