"""
Library contains tools for yolo data preparation and visualization.
"""

from BasicLib.BasicFunctions import *


############################
# yolo visualization tools #
############################

def augment_image_with_yolo_prediction(image, certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list,
                                       certainty_threshold, cat_color, max_line_thick=5, cert_line_thick_factor=1):
    """
    Returns image with predicted bboxes drawn into the image. Higher certainty results in thicker bboxes
    NOTE: this function does not accept batches, only separate images with their prediction.
    :param image: loaded image
    :param certainty_list: list of bbox certainty value's
    :param x1_list: list of bbox x1 value's
    :param y1_list: list of bbox y1 value's
    :param x2_list: list of bbox x2 value's
    :param y2_list: list of bbox y2 value's
    :param cat_list: list of bbox numeric category value's
    :param certainty_threshold: predictions with lower certainty then the threshold are not drawn.
    :param cat_color: sets the colors of the categories in format [[123, 255, 58], [0, 0, 0], ...]. The numbers
                        representing rgb colors, where the value's should lay in the color value range of the image.
    :param max_line_thick: sets the maximum line thickness
    :param cert_line_thick_factor: sets the amount of times the prediction certainty has to be higher than the threshold
                                    to increase the line thickness by one pixel.
    :return:
    """
    for certainty, x1, y1, x2, y2, cat in zip(certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list):
        if certainty > certainty_threshold:
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))

            thickness = min(max_line_thick, int((certainty / certainty_threshold) / cert_line_thick_factor))
            cv2.rectangle(image, pt1, pt2, color=cat_color[cat], thickness=thickness)
    return image


def show_yolo_results_batch(image_list, prediction_list, certainty_threshold, cat_colors, vmin=0, vmax=255):
    """
    Show batch of images with predicted bboxes.
    :param image_list: batch of images as list
    :param prediction_list: yolo model output, as list of predictions:
                            Each prediction batch containing 5 lists:
                            x1, y1, x2, y2 and category, each of them containing a list for each batch. The x1 and y1
                            value's in that list item (that list for a batch) defines the left top positions of bboxes
                            for an image, x2 and y2 define the right bottom positions. Category specifies the categories
                            as a integer being 0 to (param_n_classes - 1).

                            labels batch = [certainty, x1, y1, x2, y2, cat]
                            cert = [[cert_batch_1_bbox_1, ...], [cert_batch_2_bbox_1, ...], ...]
                            x1, y1, x2 and y2 = [[pixel_pos_batch_1_bbox_1, ...], [pixel_pos_batch_2_bbox_1, ...], ...]
                            category = [[cat_batch_1_bbox_1, ...], [cat_batch_2_bbox_1, ...], ...]
    :param certainty_threshold: predictions with lower certainty then the threshold are not drawn.
    :param cat_colors: sets colors for categories in format [[123, 255, 58], [0, 0, 0], ...]. The numbers
                        representing rgb colors, where the value's should lay in the color value range of the image.
    :param vmin: minimum pixel value
    :param vmax: maximum pixel value
    """
    for image, certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list in zip(image_list, *prediction_list):
        image = augment_image_with_yolo_prediction(image, certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list,
                                                   certainty_threshold, cat_colors)
        show_image(image, vmin=vmin, vmax=vmax)


def show_yolo_test_batch(image_list, label_list, cat_colors, vmin=-1.0, vmax=1.0):
    """
    Show batch of images with predicted bboxes.
    :param image_list: batch of images as list
    :param label_list: test data generator label, as list:
                        Each labels batch 'list item' containing 5 lists:
                        x1, y1, x2, y2 and category. The x1 and y1 list define the left top positions of bboxes
                        for an image, x2 and y2 define the right bottom positions. Category specifies the category
                        as a integer being 0 to (param_n_classes - 1).

                        labels batch = [ [x1, y1, x2, y2, cat], ...]
                        x1 and y1 and x2 and y2 = [pixel_pos, ...]
                        category = [cat_nr, ...]
    :param cat_colors: sets colors for categories in format [[123, 255, 58], [0, 0, 0], ...]. The numbers
                        representing rgb colors, where the value's should lay in the color value range of the image.
    :param vmin: minimum pixel value
    :param vmax: maximum pixel value
    """
    for image, labels in zip(image_list, label_list):
        x1_list, y1_list, x2_list, y2_list, cat_list = labels
        # create list of certainties, because labels come without certainty since each label is correct and not a
        # prediction. Because each label is correct, is for each label a certainty created with value 1, which equals
        # 100%.
        certainty_list = []
        for _ in x1_list:
            certainty_list.append(1)
        image = augment_image_with_yolo_prediction(image, certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list,
                                                   0.9, cat_colors)
        show_image(image, vmin=vmin, vmax=vmax)


###############################
# yolo data preparation tools #
###############################


def yolo_prepare_data(img, label_list, outp_size):
    """
    Prepare and normalize image and label (list of bboxes) to match the desired pixel output size. This function can be
    used when input image data does not have the required size or to augment data for training a better generalized
    model. This function will most likely be used for training or testing of a yolo model, since it prepares images and
    labels.
    :param img: loaded image
    :param label_list: list of bboxes in format [[[x,y,width,height], bbox_category_id], ...]
    :param outp_size: a tuple (width, height) setting the desired output size.
    :return: processed_image, processed_labels
    """

    crop_outp_size = compute_crop_pix_position(len(img[0]), len(img), outp_size)

    image_norm = normalize_image(img)
    cropped_image = crop_image(image_norm, crop_outp_size)
    resized_image = resize_image(cropped_image, outp_size)

    cropped_bbox_list = crop_bbox(label_list, crop_outp_size)
    resize_label = resize_bbox_list(cropped_bbox_list, len(cropped_image[0]), len(cropped_image), outp_size)

    return resized_image, resize_label


def yolo_prepare_image(image, outp_shape, crop_offset_w=0.5, crop_offset_h=0.5):
    """
    Prepare and normalize images to match the desired pixel output size. This function can be used when input image data
    does not have the required size. Since this function only resizes images, will it most likely be used to prepare
    the data for prediction by the yolo model.
    :param image: loaded image
    :param outp_shape: a tuple (width, height) setting the desired output size.
    :param crop_offset_w: Sets the horizontal offset of the cropping window as a factor of the pixels which couldn't fit
                            in the window. The factor will be random when parameter is set to None
    :param crop_offset_h: Sets the vertical offset of the cropping window as a factor of the pixels which couldn't fit
                            in the window. The factor will be random when parameter is set to None
    :return: prepared_image (meant for user display) and prepared_normalized image (meant as model input)
    """
    crop = compute_crop_pix_position(len(image[0]), len(image), outp_shape,
                                     crop_offset_w=crop_offset_w, crop_offset_h=crop_offset_h)
    image = crop_image(image, crop)
    image = resize_image(image, outp_shape)
    image_norm = normalize_image(image)
    return image, image_norm


class YoloDataFormatter:
    """
    Class to process image and bbox data. The input data should be the following format:
     data_x will contain a image file location
     data_y will contain a list of bboxes in format [[[x,y,width,height], bbox_category_id], ...]. These bboxes form
            the label for the image supplied via data_y.

    The output of the class via the process_data_to_yolo_format() function will be a processed image and a yolo model
    output label. This yolo model output label equals the exact raw ideal model output.

    The processed image, is the input image cut and resized to a valid image size for the yolo model.

    NOTE: process_data_to_yolo_format() can be used to be supplied to the DataGenerationLib.DataGenerator as data
          processing function.
    """

    def __init__(self, label_processor_func, outp_img_size=(640, 480)):
        """
        Initializes the class.
        :param label_processor_func: contains the function which transforms the labels to the desired label format. The
                                        function takes one parameter which are the to be processed labels in
                                        format [[[x,y,width,height], bbox_category_id], ...]
                                        The function should return the processed labels.
        :param outp_img_size: specifies the output image size, regardless of input size.
        """
        self.output_image_size = outp_img_size
        self.yolo_lbl_creator = label_processor_func

    def process_data_to_yolo_format(self, data_x, data_y):
        """
        Applies data augmentation, based on parameters set during class initialization.
        :param data_x: contains a image file location
        :param data_y: contains the bbox list for the image from data_x. These bbox labels shouls be supplied in format:
                       [[[x,y,width,height], bbox_category_id], ...]
        :return: returns the loaded and processed image, the processed yolo_label
        """
        img = load_image(data_x)
        img, labels = yolo_prepare_data(img, data_y, self.output_image_size)
        yolo_label = self.yolo_lbl_creator(labels)
        return img, yolo_label
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
    Class to create a yolo label from bbox data. The yolo label generated by this class forms the desired output of the
    trained yolo model. The yolo loss function determines the error between this label and the actual model output. The
    model will be optimized based on the error computed by the loss function.

    A ideal yolo model would output the data as generated by this class.
    """

    def __init__(self, cat_count, img_size, cell_grid, anchor_descriptors_list, flatten=False):
        """
        Initializes yolo label creator.
        :param cat_count: amount of categories
        :param img_size: image size in format (x, y) in pixels
        :param cell_grid: amount of output cells in format (horizontal_cell_count, vertical_cell_count)
        :param anchor_descriptors_list: list of anchor box descriptors applied to each cell
                                        in format [(height, width), ...]
        :param flatten: creates a flat label when True
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
        """Returns amount of classes or detections categories as integer"""
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
                anchor_descriptor_match.append(compute_iou_over_bbox_size(anchor_descriptor, (bbox_w, bbox_h)))

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
