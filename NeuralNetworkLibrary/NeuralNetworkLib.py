"""
Contains all basic neural network tools.
"""

from BasicLib.BasicFunctions import *
import tensorflow as tf


####################
# tensorflow tools #
####################


def print_tensor_process_values(trigger_tensor, tensor_list, tensor_name_list):
    """
    Prints all listed tensor shapes and values with the listed names, every time the trigger_tensor is executed.
    :param trigger_tensor: contains the tensor on which the print functions are added to it's pipe line.
    :param tensor_list: list of to be printed tensors
    :param tensor_name_list: lists op tensor names
    :return: trigger_tensor with print functions added to it's execution pipeline. Prints are executed, each time this
                return value is called in the tensorflow pipeline.
    """
    for tensor, name in zip(tensor_list, tensor_name_list):
        trigger_tensor = tf.Print(trigger_tensor, [tf.shape(tensor), tensor], name, summarize=150)
    return trigger_tensor


def print_tf_stats():
    """
    Prints Tensorflow related information.
    """
    print("tensorflow version: {}".format(tf.VERSION))
    print("gpu's found: {}".format(tf.test.gpu_device_name()))


def print_tf_variables():
    """
    Prints the declared tensorflow variables
    """
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(i)


##############
# bbox tools #
##############


def compute_bbox_positives(p_certainty_list, p_x1_list, p_y1_list, p_x2_list, p_y2_list,
                           l_x1_list, l_y1_list, l_x2_list, l_y2_list,
                           prediction_certainty_threshold=0.3, iou_threshold=0.6):
    """
    Computes positives between predictions and labels. returns for each positive prediction, the predicted certainty
    value in two lists. The first list contains the certainty for each positive. The second list contains the highest
    certainty per bbox in the label.
    NOTE: predicted bboxes, overlapping multiple label bboxes, might not be properly processed!

    :param p_certainty_list: list of predicted certainties
    :param p_x1_list: list of predicted x1 coord
    :param p_y1_list: list of predicted y1 coord
    :param p_x2_list: list of predicted x2 coord
    :param p_y2_list: list of predicted y2 coord
    :param l_x1_list: list of label x1 coord
    :param l_y1_list: list of label x1 coord
    :param l_x2_list: list of label x1 coord
    :param l_y2_list: list of label x1 coord
    :param prediction_certainty_threshold: ignores predictions with certainties below this threshold
    :param iou_threshold: minimum iou to count as prediction as positive
    :return: list of certainties with positive predictions, list of certainties with positive predictions per bbox
    """
    true_positive_certainties = []
    true_positive_certainties_per_bbox = len(p_certainty_list) * [0]

    for p_iou, p_x1, p_y1, p_x2, p_y2 in zip(p_certainty_list, p_x1_list, p_y1_list, p_x2_list, p_y2_list):
        if p_iou > prediction_certainty_threshold:
            for l_nr, l_x1, l_y1, l_x2, l_y2 in zip(range(len(l_x1_list)), l_x1_list, l_y1_list, l_x2_list, l_y2_list):
                iou = compute_iou(p_x1, p_y1, p_x2, p_y2, l_x1, l_y1, l_x2, l_y2)
                if iou >= iou_threshold:
                    true_positive_certainties.append(p_iou)
                    if p_iou > true_positive_certainties_per_bbox[l_nr]:
                        true_positive_certainties_per_bbox[l_nr] = p_iou
                    break

    return true_positive_certainties, true_positive_certainties_per_bbox


def compute_precision_recall(p_certainty_list, p_x1_list, p_y1_list, p_x2_list, p_y2_list,
                             l_x1_list, l_y1_list, l_x2_list, l_y2_list,
                             p_certainty_threshold=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8), iou_threshold=0.5):
    """
    Computes precision and recall for each certainty threshold
    :param p_certainty_list: list of predicted certainties
    :param p_x1_list: list of predicted x1 coord
    :param p_y1_list: list of predicted y1 coord
    :param p_x2_list: list of predicted x2 coord
    :param p_y2_list: list of predicted y2 coord
    :param l_x1_list: list of label x1 coord
    :param l_y1_list: list of label x1 coord
    :param l_x2_list: list of label x1 coord
    :param l_y2_list: list of label x1 coord
    :param p_certainty_threshold: ignores predictions with certainties below this threshold
    :param iou_threshold: minimum iou to count as prediction as positive
    :return: list with precision value's per certainty threshold, list with recall value's per certainty threshold
    """

    tp_certainty_list, tp_certainty_per_bbox_list = compute_bbox_positives(p_certainty_list, p_x1_list, p_y1_list,
                                                                           p_x2_list, p_y2_list,
                                                                           l_x1_list, l_y1_list, l_x2_list,
                                                                           l_y2_list,
                                                                           p_certainty_threshold[0],
                                                                           iou_threshold)

    tp_certainty_list = compute_thresholded_count(tp_certainty_list, p_certainty_threshold)
    tp_certainty_per_bbox_list = compute_thresholded_count(tp_certainty_per_bbox_list, p_certainty_threshold)
    positives_prediction_count_list = compute_thresholded_count(p_certainty_list, p_certainty_threshold)
    total_bbox_in_label = len(l_x1_list)

    precision = safe_divide_list(tp_certainty_list, positives_prediction_count_list)
    recall = safe_divide_list_by_value(tp_certainty_per_bbox_list, total_bbox_in_label)

    return precision, recall


##########################
# NEURAL NETWORK LIBRARY #
##########################


def create_conv(input_data, patch_size=5, filter_depth=6, stride=1, mu=0, sigma=0.05):
    """
    reates a convolutional layer, without activation.
    The funtion determins all input sizes based on the 'input_data' and can not be set by hand.

    'patch_size', The used patch is square and has a size defined by patch_size.
    'filter_depth', sets the filter depth of the created layer.
    'stride', sets the stride size for a single input channel.
    'mu' 'sigma', specifies the parameters for the random initialization of the weights.

    returns the created layer
    """
    with tf.name_scope("Conv"):
        #            [filter_height, filter_width, in_channels, out_channels]
        filter_def = [patch_size, patch_size, int(input_data.shape[3]), filter_depth]
        weights = tf.Variable(tf.truncated_normal(shape=filter_def, mean=mu, stddev=sigma), name="weights")
        biases = tf.Variable(tf.zeros(shape=filter_depth), name="biases")
        strides = [1, stride, stride, 1]
        padding = "VALID"
        return tf.nn.conv2d(input_data, weights, strides=strides, padding=padding, name="conv") + biases


def create_pull(input_data, patch_size=2, stride=None):
    """
    Creates a pulling layer.
    'patch_size', The used patch is square and has a size defined by patch_size.
    'stride', sets the stride size, and is set equal to patch_size when not defined.

    returns the created layer
    """
    if stride == None:
        stride = patch_size

    with tf.name_scope("Pull"):
        ksize = [1, patch_size, patch_size, 1]
        strides = [1, stride, stride, 1]
        padding = "VALID"
        return tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding=padding, name="pool")


def create_fully_connected(input_data, output_count, mu=0, sigma=0.05):
    """
    creates an fully connected layer.
    'output_count', sets the amount of outputs for the created layer.
    'mu' 'sigma', specifies the parameters for the random initialization of the weights.
    """
    with tf.name_scope("Full_Conn"):
        weights = tf.Variable(
            tf.truncated_normal(shape=[int(input_data.shape[1]), output_count], mean=mu, stddev=sigma), name="weights")
        biases = tf.Variable(tf.zeros(shape=output_count), name="biases")
        return tf.matmul(input_data, weights) + biases


def create_relu(input_data, tensorboard=False):
    """
    Creates a ReLu activation function.
    'tensorboard', stores relu output as picture in tensorboard. This function only works properly with a batch size = 1
    """
    with tf.name_scope("ReLu"):
        result = tf.nn.relu(input_data, name="relu")

        if tensorboard and len(result.shape) == 4:
            filter_depth = int(result.shape[3])
            # split filter (dimension=filter_depth) in 1 dimensional filters (images) and concatanate the split filters.
            # each filter is now split in to a different batch to be able to be shown as greyscale image by tensorboard.
            tf.summary.image(max_outputs=filter_depth, name="relu_image",
                             tensor=tf.concat(tf.split(value=result, num_or_size_splits=filter_depth, axis=3), 0))
        return result


def create_leaky_relu(input_data, alpha=0.1, tensorboard=False):
    """
    Creates a leaky ReLu activation function
    'tensorboard', stores leaky relu output in tensorboard. This function only works properly with a batch size = 1
    """
    with tf.name_scope("ReLu"):
        result = tf.nn.leaky_relu(input_data, alpha=alpha, name="leaky_relu")

        if tensorboard and len(result.shape) == 4:
            filter_depth = int(result.shape[3])
            # split filter (dimension=filter_depth) in 1 dimensional filters (images) and concatanate the split
            # filters. Each filter is now split in to a different batch to be able to be shown as greyscale image by
            # tensorboard.
            tf.summary.image(max_outputs=filter_depth, name="relu_image",
                             tensor=tf.concat(tf.split(value=result, num_or_size_splits=filter_depth, axis=3), 0))
        return result

# def get_model(input_data, n_classes, show=False):
#     """creates the entire model.
#     'show', displays the created tensorflow variables.
#     'tensorboard', stores relu output as picture in tensorboard.
#     """
#     process_value = create_conv(input_data, patch_size=5, filter_depth=8, stride=1)
#     process_value = create_relu(process_value, tensorboard=True)
#     process_value = create_pull(process_value, patch_size=2)
#
#     process_value = create_conv(process_value, patch_size=5, filter_depth=18, stride=1)
#     process_value = create_relu(process_value, tensorboard=True)
#     process_value = create_pull(process_value, patch_size=2)
#
#     process_value = tf.contrib.layers.flatten(process_value)
#     process_value = create_fully_connected(process_value, output_count=120)
#     process_value = create_relu(process_value)
#
#     process_value = create_fully_connected(process_value, output_count=84)
#     process_value = create_relu(process_value)
#
#     process_value = create_fully_connected(process_value, output_count=n_classes)
#
#     if show:
#         print_tf_variables()
#     return process_value
