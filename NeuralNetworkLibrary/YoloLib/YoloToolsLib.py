"""
Library contains tools for yolo data preparation and vizualization
"""

from BasicLib.BasicFunctions import *


############################
# yolo visualization tools #
############################


def show_yolo_result(image, certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list, certainty_threshold, n_class):
    """
    Show image with predicted bboxes.
    NOTE: this function does not accept batches, only separate images with their prediction.
    :param image: loaded images
    :param certainty_list: list of bbox certainty value's
    :param x1_list: list of bbox x1 value's
    :param y1_list: list of bbox y1 value's
    :param x2_list: list of bbox x2 value's
    :param y2_list: list of bbox y2 value's
    :param cat_list: list of bbox numeric category value's
    :param certainty_threshold: predictions with lower certainty the threshold are not drawn.
    :param n_class: amount of classes
    """
    cat_color = [get_random_color()[0] * (1 / 128.0) - 1 for i in range(n_class)]
    for certainty, x1, y1, x2, y2, cat in zip(certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list):
        if certainty > certainty_threshold:
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))

            cv2.rectangle(image, pt1, pt2, cat_color[cat], 2)
    show_image(image, vmin=-1, vmax=1)


###############################
# yolo data preparation tools #
###############################


def resize_yolo_data(img, label_list, outp_img_size=(640, 480)):
    """
    Resizes image and bbox labels to given output size, regardless of input size.
    :param img: contains a loaded image
    :param label_list: contains the labels in format [[[x,y,width,height], bbox_category_id], ...]
    :param outp_img_size: returns the resize image and labels in same format
    :return: resized image and resized labels
    """
    # determine resize factores
    resize_factor_x = outp_img_size[0] / len(img[0])
    resize_factor_y = outp_img_size[1] / len(img)

    # resize image
    resize_image = cv2.resize(img, outp_img_size)

    # resize labels
    resize_label = []
    for bbox, category in label_list:
        resize_label.append([[bbox[0] * resize_factor_x, bbox[1] * resize_factor_y, bbox[2] * resize_factor_x,
                              bbox[3] * resize_factor_y], category])

    return resize_image, resize_label


def crop_yolo_data(img, label_list, outp_size):
    """
    Crop yolo data to output size width / height ratio and resize to desired pixel output size.
    :param img: loaded image
    :param label_list: list of bboxes in format [[[x,y,width,height], bbox_category_id], ...]
    :param outp_size: a tuple (width, height)
    :return: processed_image, processed_labels
    """
    img_w = len(img[0])
    img_h = len(img)

    outp_w = outp_size[0]
    outp_h = outp_size[1]

    crop_offset_w = np.random.rand()
    crop_offset_h = np.random.rand()

    outp_ratio = outp_w / outp_h

    crop_w, crop_h = compute_crop_window(img_w, img_h, outp_ratio)

    crop_outp_size = [0, 0, 0, 0]
    crop_outp_size[0] = int(crop_offset_w * (img_w - crop_w))
    crop_outp_size[1] = int(crop_offset_h * (img_h - crop_h))
    crop_outp_size[2] = crop_outp_size[0] + crop_w
    crop_outp_size[3] = crop_outp_size[1] + crop_h

    cropped_image = crop_image(img, crop_outp_size)
    cropped_bbox_list = crop_bbox(label_list, crop_outp_size)
    return resize_yolo_data(cropped_image, cropped_bbox_list, outp_img_size=outp_size)


class DataAugmenter:
    """
    Class to augment data. augment_yolo_data() can be used to supply to the data generator from the DataGenerationLib
    as parameter.
    """

    def __init__(self, outp_img_size=(640, 480)):
        """
        Initializes the class.
        :param outp_img_size: specifies the output image size, regardless of input size.
        """
        self.output_image_size = outp_img_size

    def augment_yolo_data(self, img, label_list):
        """
        Applies data augmentation, based on parameters set during class initialization.
        :param img: contains the loaded image
        :param label_list: contains the labels list for image in format [[[x,y,width,height], bbox_category_id], ...]
        :return: returns the augmented data in the same format
        """
        # return resize_yolo_data(image, labels_list, output_image_size=self.output_image_size)
        return crop_yolo_data(img, label_list, self.output_image_size)
        # TODO: ADD MORE AUGMENTATIONS


def bbox_splitter(label_list):
    """
    Bbox splitter is a DataGenerator label creator, which returns the data in 5 list. Can be used to supply to the data
    generator from the DataGenerationLib as label creator.
    :param label_list: contains the labels in format [[[x,y,width,height], bbox_category_id], ...]
    :return: 5 lists: x1, y1, x2, y2, category
    """
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    category = []

    for bbox_label, category_label in label_list:
        x1.append(bbox_label[0])
        y1.append(bbox_label[1])
        x2.append(bbox_label[0] + bbox_label[2])
        y2.append(bbox_label[1] + bbox_label[3])
        category.append(category_label)

    return x1, y1, x2, y2, category


class YoloLabelCreator:
    """
    Class to create a yolo label from bbox data.
    """

    def __init__(self, cat_count, img_size, cell_grid, anchor_descriptors_list, flatten=False):
        """
        Initializes yolo label creator.
        :param cat_count: amount of categories
        :param img_size: image size in format (x, y) in pixels
        :param cell_grid: amount of output cells in format (horizontal_cell_count, vertical_cell_count)
        :param anchor_descriptors_list: list of anchor box descriptors applied to each cell
                                        in format [(height, width), ...]
        """
        # assert if cell_grid does not match image size
        assert img_size[0] % cell_grid[0] == 0
        assert img_size[1] % cell_grid[1] == 0

        # load parameters in to class
        self.categories_count = cat_count
        self.image_size = img_size
        self.cell_grid = cell_grid
        self.anchor_descriptors_list = anchor_descriptors_list
        self.flatten = flatten

        # compute cell parameters
        self.cell_size_in_pixels_x = self.image_size[0] / self.cell_grid[0]
        self.cell_size_in_pixels_y = self.image_size[1] / self.cell_grid[1]

        # compute yolo_label dimensions
        self.anchor_descriptor_length = 5 + self.categories_count
        self.cell_descriptor_length = self.anchor_descriptor_length * len(self.anchor_descriptors_list)

    def get_shape(self):
        """
        Returns yolo data shape of a single data point (not a batch).
        :return: [cell_x_count, cell_y_count, depths] or when flat output [output_count]
        """
        if not self.flatten:
            print("not flatten", self.flatten)
            shape = [self.cell_grid[0], self.cell_grid[1], self.cell_descriptor_length]
        else:
            print("flatten", self.flatten)
            shape = [self.cell_grid[0] * self.cell_grid[1] * self.cell_descriptor_length]
        return shape

    def get_category_count(self):
        """
        Returns amount of classes or detections categories as integer
        :return: amount of classes or detections categories as integer
        """
        return self.categories_count

    def bbox_to_yolo_label(self, label_list):
        """
        Creates yolo label from bbox data. Used to train neural networks.
        :param label_list: list of labels in format [[[x,y,width,height], bbox_category_id], ...]
        :return: yolo_label in format:
                [[x_cell_1 [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]],
                ...
                [x_cell_n [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]]]

                with anchor in format:
                [object_detected 1 or 0, bbox_center_x, bbox_center_y, bbox_width, bbox_height, category_0, category_n]
        """

        # create empty yolo_label
        yolo_label = np.zeros((self.cell_grid[0], self.cell_grid[1], self.cell_descriptor_length))

        # add bbox to empty yolo_label
        for bbox, category in label_list:
            # compute bbox width, height and center point.
            bbox_w = bbox[2]
            bbox_h = bbox[3]
            bbox_x = bbox[0] + bbox_w / 2
            bbox_y = bbox[1] + bbox_h / 2

            # alocate bbox centerpoint to cell, compute bbox positions relative to cell,
            # compute bbox size relative to cell
            bbox_in_cell_x = np.floor(bbox_x / self.cell_size_in_pixels_x).astype(np.uint)
            bbox_in_cell_y = np.floor(bbox_y / self.cell_size_in_pixels_y).astype(np.uint)
            bbox_cell_position_x = (bbox_x - (self.cell_size_in_pixels_x * bbox_in_cell_x)) / self.cell_size_in_pixels_x
            bbox_cell_position_y = (bbox_y - (self.cell_size_in_pixels_y * bbox_in_cell_y)) / self.cell_size_in_pixels_y
            bbox_cell_height = bbox_h / self.cell_size_in_pixels_x
            bbox_cell_width = bbox_w / self.cell_size_in_pixels_y

            # compute "iou", which is a factor describing how good the bbox and anchor box match
            anchor_descriptor_match = []
            for anchor_descriptor in self.anchor_descriptors_list:
                bbox_anchor_overlap_area = min(anchor_descriptor[0], bbox_w) * min(anchor_descriptor[1], bbox_h)
                bbox_anchor_total_area = (anchor_descriptor[0] * anchor_descriptor[1]) + (
                        bbox_w * bbox_h) - bbox_anchor_overlap_area
                iou = bbox_anchor_overlap_area / bbox_anchor_total_area

                anchor_descriptor_match.append(iou)

            # assign bbox to best fitting available anchor box
            # TODO: find better solution for taken anchor boxes
            # TODO: improve message reporting
            for i in range(len(self.anchor_descriptors_list)):
                anchor_match = np.argmax(anchor_descriptor_match)
                if yolo_label[bbox_in_cell_x][bbox_in_cell_y][int(anchor_match * self.anchor_descriptor_length)] == 0:
                    break
                anchor_descriptor_match[anchor_match] = 0
                # print("WARNING: Best anchor taken")
            else:
                # print("WARNING: All anchors taken")
                continue

            # load yolo_label
            yolo_label[bbox_in_cell_x][bbox_in_cell_y][anchor_match * self.anchor_descriptor_length] = \
                anchor_descriptor_match[anchor_match]
            yolo_label[bbox_in_cell_x][bbox_in_cell_y][
                (anchor_match * self.anchor_descriptor_length) + 1] = bbox_cell_position_x
            yolo_label[bbox_in_cell_x][bbox_in_cell_y][
                (anchor_match * self.anchor_descriptor_length) + 2] = bbox_cell_position_y
            yolo_label[bbox_in_cell_x][bbox_in_cell_y][
                (anchor_match * self.anchor_descriptor_length) + 3] = bbox_cell_width
            yolo_label[bbox_in_cell_x][bbox_in_cell_y][
                (anchor_match * self.anchor_descriptor_length) + 4] = bbox_cell_height
            yolo_label[bbox_in_cell_x][bbox_in_cell_y][
                (anchor_match * self.anchor_descriptor_length) + 5 + category] = 1

        # flatten label if necessary
        if self.flatten:
            yolo_label = yolo_label.ravel()
        return yolo_label
