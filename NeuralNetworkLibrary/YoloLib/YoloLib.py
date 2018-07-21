"""
Module contains all functions to create, train, test and use a yolo network.
"""
from NeuralNetworkLibrary.NeuralNetworkLib import *


######################################
# Yolo output data processing layers #
######################################


def yolo_unpack_output(data, outp_cat_count, enable_activation):
    """
    Add's a yolo model output data preparation layer to the yolo tensorflow pipeline. Unpacks yolo data in to separate
    parts.

        yolo_label in format:
        [[x_cell_1 [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]],
        ...
        [x_cell_n [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]]]

        with anchor in format:
        [object_detected 1 or 0, bbox_center_x, bbox_center_y, bbox_width, bbox_height, category_0, category_n]

    :param data: yolo formatted data
    :param outp_cat_count: amount of classes or detections categories as integer
    :param enable_activation: enable the output activation
                                sigmoid in iou, x and y
                                exp on w and h
    :return: Tensorflow pipeline to unpack data in 6 tensor lists iou, x, y, w, h, classes. All in format:
             [[x_cell_1 [y_cell_1 [anchor_1_iou], [anchor_ndepth_iou]], [y_cell_n [anchor_1_iou], [anchor_ndepth_iou]]],
             ...
             [x_cell_n [y_cell_1 [anchor_1_iou], [anchor_ndepth_iou]], [y_cell_n [anchor_1_iou], [anchor_ndepth_iou]]]]
    """
    anchor_length = 5 + outp_cat_count
    output_shape = data.get_shape().as_list()

    iou = data[:, :, :, 0::anchor_length]
    x = data[:, :, :, 1::anchor_length]
    y = data[:, :, :, 2::anchor_length]
    w = data[:, :, :, 3::anchor_length]
    h = data[:, :, :, 4::anchor_length]

    classes = []
    for i in range(0, output_shape[3], anchor_length):
        classes.append(data[:, :, :, 5 + i:5 + outp_cat_count + i])
    classes = tf.stack(classes, 4)

    if enable_activation:
        iou = tf.sigmoid(iou)
        x = tf.sigmoid(x)
        y = tf.sigmoid(y)
        w = tf.exp(w)
        h = tf.exp(h)

    return iou, x, y, w, h, classes


def yolo_pred_to_bbox(logits, n_classes, cell_pixel_width, cell_pixel_height):
    """
    Add's a yolo model output data preparation layer to the yolo Tensorflow pipeline. This output layer add's:
     - translation from yolo bbox positions and sizes to image pixel bbox positions and sizes.
     - bbox coordinats in x1 y1 and x2 y2 coordinates, instead of Xcenter Ycenter Width Height.
     - translation from hot encoded category's to class number.
     - reshapes output data in flat list per batch
    :param logits: contains the yolo model
    :param n_classes: sets the number of yolo output classes
    :param cell_pixel_width: sets yolo output cell height in pixels
    :param cell_pixel_height: sets yolo output cell width in pixels
    :return: 6 lists: iou (confidence), x1, y1, x2, y2, class. All in format
             [[X1_bbox value's for first data point], [X1_bbox value's for second data point], ...]
    """
    iou, x, y, w, h, classes = yolo_unpack_output(logits, n_classes, True)

    cell_grid_size = x.get_shape()

    x_cell_offset = [[[i * cell_pixel_width]] for i in range(cell_grid_size[1])]
    y_cell_offset = [[i * cell_pixel_height] for i in range(cell_grid_size[2])]

    x_scaled = (x * cell_pixel_width) + x_cell_offset
    w_scaled = w * cell_pixel_width
    y_scaled = (y * cell_pixel_height) + y_cell_offset
    h_scaled = h * cell_pixel_height

    category_numeric = tf.argmax(classes, axis=3)

    x1 = x_scaled - (w_scaled / 2)
    x2 = x_scaled + (w_scaled / 2)
    y1 = y_scaled - (h_scaled / 2)
    y2 = y_scaled + (h_scaled / 2)

    iou = tf.reshape(iou, [-1, cell_grid_size[1] * cell_grid_size[2] * cell_grid_size[3]])
    x1 = tf.reshape(x1, [-1, cell_grid_size[1] * cell_grid_size[2] * cell_grid_size[3]])
    y1 = tf.reshape(y1, [-1, cell_grid_size[1] * cell_grid_size[2] * cell_grid_size[3]])
    x2 = tf.reshape(x2, [-1, cell_grid_size[1] * cell_grid_size[2] * cell_grid_size[3]])
    y2 = tf.reshape(y2, [-1, cell_grid_size[1] * cell_grid_size[2] * cell_grid_size[3]])
    category_numeric = tf.reshape(category_numeric, [-1, cell_grid_size[1] * cell_grid_size[2] * cell_grid_size[3]])

    return iou, x1, y1, x2, y2, category_numeric


######################
# YOLO LOSS FUNCTION #
######################


def yolo_loss(prediction, labels, outp_cat_count, coord_loss_compensation=5, no_object_compensation=0.5,
              verbose=False):
    """
    Add's a yolo loss function to the yolo tensorflow pipeline. Returns position_loss, size_loss, category_loss,
    confidence_loss for given prediction output. All loss results being a single loss value (float). The total loss
    is computed by adding all separate losses together.

    :param prediction: contains the nn prediction in yolo format:
        [[x_cell_1 [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]],
        ...
        [x_cell_n [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]]]

        with anchor in format:
        [object_detected 1 or 0, bbox_center_x, bbox_center_y, bbox_width, bbox_height, category_0, category_n]
    :param labels: contains labels in same yolo format
    :param outp_cat_count: amount of classes or detections categories as integer
    :param coord_loss_compensation: sets a loss compensation factor for bbox position and dimension loss
    :param no_object_compensation: sets a loss compensation factor for empty anchorboxes
    :param verbose: prints detailed info when True
    :return: Tensorflow tensors to compute: position_loss, size_loss, category_loss, confidence_loss
    """
    with tf.name_scope("yolo_loss"):
        predicted_iou, predicted_x, predicted_y, predicted_w, predicted_h, predicted_classes = yolo_unpack_output(
            prediction, outp_cat_count, True)
        label_iou, label_x, label_y, label_w, label_h, label_classes = yolo_unpack_output(labels, outp_cat_count,
                                                                                          False)
        object_in_anchor = tf.clip_by_value(tf.ceil(label_iou), 0, 1)
        no_object_in_anchor = (object_in_anchor * -1) + 1

        # anchor position loss
        position_loss = coord_loss_compensation * tf.reduce_sum(
            object_in_anchor * (tf.pow(label_x - predicted_x, 2) + tf.pow(label_y - predicted_y, 2)))

        # anchor size loss
        size_loss = coord_loss_compensation * tf.reduce_sum(object_in_anchor * (
                tf.pow(tf.sqrt(label_w) - tf.sqrt(predicted_w), 2) + tf.pow(tf.sqrt(label_h) - tf.sqrt(predicted_h),
                                                                            2)))

        # anchor category loss
        category_loss = tf.reduce_sum(object_in_anchor * tf.reduce_sum(tf.pow(label_classes - predicted_classes, 2), 3))

        # anchor confidence loss
        confidence_loss = tf.reduce_sum(object_in_anchor * tf.pow(label_iou - predicted_iou, 2)) + (
                no_object_compensation * tf.reduce_sum(no_object_in_anchor * tf.pow(label_iou - predicted_iou, 2)))

    if verbose:
        print("Predict output: ", prediction, predicted_iou, predicted_x, predicted_y, predicted_w, predicted_h,
              predicted_classes)
        print("Label output: ", labels, label_iou, label_x, label_y, label_w, label_h, label_classes)
        print("position_loss", position_loss)
        print("size_loss", size_loss)
        print("category_loss", category_loss)
        print("confidence_loss", confidence_loss)

    return position_loss, size_loss, category_loss, confidence_loss


##############
# Yolo model #
##############


def create_yolo_model(input_data, output_shape, verbose=False):
    """
    Returns a yolo model.
    :param input_data: contains the placeholder for the input features
    :param output_shape: contains a list with the yolo output dimensions of a prediction for a single data point.
            This should be a yolo compatible shape:
            [x, y, anchor_depth]
            NOTE: the batch dimension will be added by the function

            output date in format:
            [[x_cell_1 [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]],
            ...
            [x_cell_n [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]], ..., batch_n]

            with anchor in format:
            [object_detected 1 or 0, bbox_center_x, bbox_center_y, bbox_width, bbox_height, category_0, category_n]
    :param verbose: prints details is True
    :return: Tensorflow model
    """
    output_neuron_count = 1
    for i in output_shape:
        output_neuron_count *= i

    # input_data = tf.Print(input_data, [input_data], "input_data", summarize=150)

    process_value = create_conv(input_data, patch_size=5, filter_depth=16, stride=2)
    process_value = create_leaky_relu(process_value, tensorboard=False)
    process_value = create_pull(process_value, patch_size=2)
    print(process_value)
    # process_value = tf.Print(process_value, [process_value], "process_value", summarize=150)

    process_value = create_conv(process_value, patch_size=3, filter_depth=32, stride=1)
    process_value = create_leaky_relu(process_value, tensorboard=False)
    process_value = create_pull(process_value, patch_size=2)
    print(process_value)
    # process_value = tf.Print(process_value, [process_value], "process_value", summarize=150)

    process_value = create_conv(process_value, patch_size=3, filter_depth=64, stride=1)
    process_value = create_leaky_relu(process_value, tensorboard=False)
    process_value = create_pull(process_value, patch_size=2)
    print(process_value)
    # process_value = tf.Print(process_value, [process_value], "process_value", summarize=150)

    process_value = create_conv(process_value, patch_size=3, filter_depth=128, stride=1)
    process_value = create_leaky_relu(process_value, tensorboard=False)
    process_value = create_pull(process_value, patch_size=2)
    print(process_value)
    # process_value = tf.Print(process_value, [process_value], "process_value", summarize=150)

    process_value = create_conv(process_value, patch_size=3, filter_depth=256, stride=1)
    process_value = create_leaky_relu(process_value, tensorboard=False)
    # process_value = create_pull(process_value, patch_size=2)
    print(process_value)
    # process_value = tf.Print(process_value, [process_value], "process_value", summarize=150)

    process_value = create_conv(process_value, patch_size=3, filter_depth=256, stride=1)
    process_value = create_leaky_relu(process_value, tensorboard=False)
    # process_value = create_pull(process_value, patch_size=2)
    print(process_value)
    # process_value = tf.Print(process_value, [process_value], "process_value", summarize=150)

    process_value = tf.contrib.layers.flatten(process_value)
    process_value = create_fully_connected(process_value, output_count=5500)
    process_value = create_leaky_relu(process_value)
    print(process_value)
    # process_value = tf.Print(process_value, [process_value], "process_value", summarize=150)

    process_value = create_fully_connected(process_value, output_count=output_neuron_count)
    print(process_value)
    # process_value = tf.Print(process_value, [process_value], "process_value", summarize=150)

    process_value = tf.reshape(process_value, [-1] + output_shape)
    print(process_value)
    # process_value = tf.Print(process_value, [process_value], "process_value", summarize=150)

    if verbose:
        print_tf_variables()
    return process_value


#################
# Yolo frontend #
#################


class YoloObjectDetection:
    """
    create, train, test and use yolo model.
    """

    # TODO: auto compute cell_pix_size and remove parameter
    def __init__(self, param_input_img_shape, param_model_outp_shape, param_n_classes, yolo_cell_pix_size):
        """
        Initializes the Yolo front end.
        :param param_input_img_shape: defines the shape of the input images in format (width, height) in pixels
        :param param_model_outp_shape: contains a list with the yolo output dimensions of a prediction for a single data
            point. This should be a yolo compatible shape:
            [x, y, anchor_depth]
            NOTE: the batch dimension will be added by the function

            output date in format:
            [[x_cell_1 [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]],
            ...
            [x_cell_n [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]], ..., batch_n]

            with anchor in format:
            [object_detected 1 or 0, bbox_center_x, bbox_center_y, bbox_width, bbox_height, category_0, category_n]
        :param param_n_classes: sets amount of classes (categories) to predict
        :param yolo_cell_pix_size: sets the size of the yolo output cells in pixels
        """
        # reset tensor flow graph
        tf.reset_default_graph()

        # declare place holders
        with tf.name_scope("PlaceHolders"):
            # set place holders                  shape = [batch, in_height, in_width, in_channels]
            self.features = tf.placeholder("float32", shape=[None, param_input_img_shape[1], param_input_img_shape[0],
                                                             param_input_img_shape[2]], name="features")
            self.labels = tf.placeholder("float32", shape=[None, *param_model_outp_shape], name="labels")
            self.rate = tf.placeholder("float32", shape=[], name="learning_rate")

        # get model
        self.logits = create_yolo_model(self.features, param_model_outp_shape)

        # Build training pipeline
        with tf.name_scope("TrainPipeLine"):
            self.loss_ops = yolo_loss(self.logits, self.labels, param_n_classes)
            self.pos_loss_op, self.size_loss_op, self.cat_loss_op, self.conf_loss_op = self.loss_ops
            self.total_loss_op = self.pos_loss_op + self.size_loss_op + self.cat_loss_op + self.conf_loss_op
            self.training_op = tf.train.AdamOptimizer(learning_rate=self.rate, name="ADAM").minimize(self.total_loss_op)

        # Build prediction pipeline
        with tf.name_scope("PredictionPipeLine"):
            # self.iou_op, self.x1_op, self.y1_op, self.x2_op, self.y2_op, self.cat_op = self.prediction_ops
            self.prediction_ops = yolo_pred_to_bbox(self.logits, param_n_classes,
                                                    yolo_cell_pix_size[0], yolo_cell_pix_size[1])

        # declare model saver
        self.model_save_op = tf.train.Saver()

        self.sess = tf.Session()

    def init(self, pre_trained_model=''):
        """
        Initialize model variables. If model file location is given, the model will be loaded from disk. If no model
        file locations is gives, will all variables be initialized as a new untrained model.
        :param pre_trained_model: file location of the model.
        """
        if pre_trained_model != '':
            self.model_save_op.restore(self.sess, pre_trained_model)
        else:
            self.sess.run(tf.global_variables_initializer())

    def train(self, train_data_generator, learn_rate=0.001):
        """
        Train the model on the data provided by the data generator. This function runs until the data generator
        depletes. The data generator should return a x_batch containing a list of images and y_batch containing a list
        of labels like [[[[x,y,width,height], bbox_category_id], ...], [[[x,y,width,height], bbox_category_id], ...]]
        :param train_data_generator: contains data generator
        :param learn_rate: sets learning rate
        """
        for batch_x, batch_y in train_data_generator:
            # perform forward and backward propegation and gather data for tensorboard
            res_total_loss, res_position_loss, res_size_loss, res_category_loss, res_confidence_loss, _ = self.sess.run(
                [self.total_loss_op, *self.loss_ops, self.training_op],
                feed_dict={self.features: batch_x, self.labels: batch_y, self.rate: learn_rate})

            # display epoch details
            print("LOSS: pos {:8.3f} size {:8.3f} cat {:8.3f} conf {:8.3f} tot {:8.3f}".format(
                res_position_loss, res_size_loss, res_category_loss, res_confidence_loss, res_total_loss))

    def test(self, test_data_generator, cetainty_thresholds=(0.1, 0.2, 0.3, 0.4, 0.6)):
        """
        Test the model on the data provided by the data generator. This function runs until the data generator
        depletes. The data generator should return a x_batch containing a list om images and y_batch containing a list
        of labels like [[[[x,y,width,height], bbox_category_id], ...], [[[x,y,width,height], bbox_category_id], ...]]

        The function returns the precision and recall found.

        :param test_data_generator: contains data generator
        :param cetainty_thresholds: contains a list of certainty thresholds to determine precision and recall against.
        :return: precision and recall, per certainty threshold
        """
        test_recall_list = []
        test_precision_list = []
        for test_batch_x, test_batch_y in test_data_generator:

            # perform forward propagation
            predd_batch = self.sess.run([*self.prediction_ops], feed_dict={self.features: test_batch_x})

            for predd_iou, predd_x1, predd_y1, predd_x2, predd_y2, predd_cat, label in zip(*predd_batch,
                                                                                           test_batch_y):
                pred_precision, pred_recall = compute_precision_recall(predd_iou,
                                                                       predd_x1,
                                                                       predd_y1,
                                                                       predd_x2,
                                                                       predd_y2,
                                                                       label[0],
                                                                       label[1],
                                                                       label[2],
                                                                       label[3],
                                                                       cetainty_thresholds)
                test_precision_list.append(pred_precision)
                test_recall_list.append(pred_recall)

        test_precision = np.mean(test_precision_list, axis=0)
        test_recall = np.mean(test_recall_list, axis=0)
        return test_precision, test_recall

    def predict(self, input_image_batch):
        """
        Predict bboxes for batch of input images using yolo model. Prediction is returned in 6 lists:
        predd_certainty, predd_x1, predd_y1, predd_x2, predd_y2, predd_cat. Each list is builded as following:
            [[batch_first_image bbox value, batch_first_image bbox value, ...], [batch_second_image bbox value, ...]]
        :param input_image_batch:
        :return: lists: predd_certainty, predd_x1, predd_y1, predd_x2, predd_y2, predd_cat
        """
        return self.sess.run([*self.prediction_ops], feed_dict={self.features: input_image_batch})

    def save(self, save_path):
        """
        save model to given file location.
        :param save_path: file name and location to store the trained model
        """
        self.model_save_op.save(self.sess, save_path)
