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
    Prints all listed tensor shapes and values with the listed names, every time the trigger_tensor is executed. This
    printed information gives insight in the model during usages, training or testing. This insight can be used for
    debugging, optimization or creating a better understanding in the model.
    :param trigger_tensor: contains the tensor on which the print functions are added to it's pipe line.
    :param tensor_list: list of to be printed tensors
    :param tensor_name_list: lists op tensor names
    :return: trigger_tensor with print functions added to it's execution pipeline. Prints are executed, each time this
                return value is called in the tensorflow pipeline.
    """
    for tensor, name in zip(tensor_list, tensor_name_list):
        trigger_tensor = tf.Print(trigger_tensor, [tf.shape(tensor), tensor], name, summarize=150)
    return trigger_tensor


def get_tf_stats_as_pretty_string():
    """Returns a pretty, printable string. The string containing Tensorflow statistics."""
    string = "Tensorflow version: {} \n".format(tf.VERSION) + \
             "GPU's found: {}".format(tf.test.gpu_device_name())
    return string


def print_tf_stats():
    """Prints Tensorflow related information."""
    print(get_tf_stats_as_pretty_string())


def get_tf_variables_as_pretty_string(only_trainable=False):
    """
    Returns the declared Tensorflow variables as a pretty string.
    :param only_trainable: only returns trainable variables when True
    :return: string describing variables
    """
    string = ""

    if only_trainable:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    else:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    for i in variables:
        string += "{}\n".format(i)
    return string


def print_tf_variables(only_trainable=False):
    """
    Prints the declared Tensorflow variables.
    :param only_trainable: only prints trainable variables when True
    """
    print(get_tf_variables_as_pretty_string(only_trainable))


##############
# bbox tools #
##############


def compute_bbox_positives(p_certainty_list, p_x1_list, p_y1_list, p_x2_list, p_y2_list,
                           l_x1_list, l_y1_list, l_x2_list, l_y2_list,
                           prediction_certainty_threshold=0.3, iou_threshold=0.6):
    """
    Computes positives between predictions and labels. Returns for each positive prediction the predicted certainty
    value in two lists. The first list contains the certainty for each positive. The second list contains the highest
    certainty per bbox in the label.

    The returned value's are required to determine performance figures of the model like precision and recall.

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
    Computes precision and recall for each certainty threshold. Precision and Recall give insight in the models
    performance.
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

    # both safe_divide functions return zero when dividing by zero. This because when dividing by zero are there no
    # bboxes predicted at all, or the label did not contain any bboxes.
    precision = safe_divide_list(tp_certainty_list, positives_prediction_count_list)
    recall = safe_divide_list_by_value(tp_certainty_per_bbox_list, total_bbox_in_label)

    return precision, recall


def normalize_summarize_bbox_sizes(img_size_list, bbox_nested_list, normalized_outp_size):
    """
    Returns a flat list of normalized bbox sizes. The img_size_list is a list of image sizes, corresponding to the
    list of labels in bbox_nested_list. Each bbox_nested_list label is a list of bboxes. To normalize each bbox size, is
    the rescale factor determined between a img_size from img_size_list and the normalized_outp_size parameter. The
    rescale factor is than used to resize the bbox width and height. The resulting, normalized bbox width and height
    are than added to the output list which will be returned in the end.
    :param img_size_list: contains a list of image sizes in format [[width, heigth], ...]
    :param bbox_nested_list: contains a list of labels, each label being a list of bboxes. this in format:
                    [[[x,y,width,height], bbox_category_id], ...], [[x,y,width,height], bbox_category_id], ...], ...]
    :param normalized_outp_size: sets the output size to normalize the bboxes to.
    :return: flat list of normalized bbox sizes in format [[width, heigth], ...]
    """
    flat_bbox_size_list = []
    for size, bbox_list in zip(img_size_list, bbox_nested_list):
        w, h = compute_crop_window_size(size[0], size[1], normalized_outp_size[0] /
                                        normalized_outp_size[1])
        if w == size[0]:
            factor = normalized_outp_size[0] / w
        else:
            factor = normalized_outp_size[1] / h

        for bbox, cat in bbox_list:
            flat_bbox_size_list.append([bbox[2] * factor, bbox[3] * factor])

    return flat_bbox_size_list


def compute_anchor_bboxes(img_size_list, labels_list, img_outp_size,
                          quit_n_anchors=10, quit_min_iou_gain=0.01, quit_min_iou=0.45):
    """
    Returns a optimized set of anchor boxes for the given data set.
    :param img_size_list: contains a list of image sizes in format [[width, heigth], ...]
    :param labels_list: contains a list of labels, each label being a list of bboxes. this in format:
                    [[[x,y,width,height], bbox_category_id], ...], [[x,y,width,height], bbox_category_id], ...], ...]
    :param img_outp_size: sets the output size to normalize the bboxes to.
    :param quit_n_anchors: quit anchor box search when amount of anchor boxes hits 'quit_n_anchors'. (Minimum is 2)
    :param quit_min_iou_gain: quit anchor box search when 'quit_n_anchors' is not reached but the difference in the iou
                              between the second to last and last iou, computed over the anchor boxes and bboxes, is
                              less than 'quit_min_iou_gain'
    :param quit_min_iou: prevents quit by 'quit_min_iou_gain', until minimum iou is reached specified by 'quit_min_iou'.
    :return: anchor box specification in format:
            [[anchor_1_width, anchor_1_height], [anchor_2_width, anchor_2_height], ...]
    """
    centroid_list = [[]]
    last_iou = 0.0
    iou = 0.0

    bbox_size_list = normalize_summarize_bbox_sizes(img_size_list, labels_list, img_outp_size)
    for i in range(2, quit_n_anchors + 1):
        centroid_list, cluster_list = cluster_data(bbox_size_list, i)

        iou_coll = []
        for centroid, cluster in zip(centroid_list, cluster_list):
            for data_p in cluster:
                iou_coll.append(compute_iou_over_bbox_size(data_p, centroid))

        iou = np.mean(iou_coll)
        if iou - last_iou < quit_min_iou_gain and iou > quit_min_iou:
            break
        last_iou = iou
    centroid_list = centroid_list.astype(int).tolist()
    return centroid_list, iou


##########################
# NEURAL NETWORK LIBRARY #
##########################


def create_conv(input_data, patch_size=5, filter_depth=6, stride=1, mu=0, sigma=0.05, padding="VALID"):
    """
    Creates a convolutional layer.
    The function determines all input sizes based on the 'input_data' and can not be set by hand.

    'patch_size', The used patch is square and has a size defined by patch_size.
    'filter_depth', sets the filter depth of the created layer.
    'stride', sets the stride size for a single input channel.
    'mu' 'sigma', specifies the parameters for the random initialization of the weights.
    'padding', sets the padding using the string 'VALID' or 'SAME'.

    returns the created layer
    """
    with tf.name_scope("Conv"):
        #            [filter_height, filter_width, in_channels, out_channels]
        filter_def = [patch_size, patch_size, int(input_data.shape[3]), filter_depth]
        weights = tf.Variable(tf.truncated_normal(shape=filter_def, mean=mu, stddev=sigma), name="weights")
        biases = tf.Variable(tf.zeros(shape=filter_depth), name="biases")
        strides = [1, stride, stride, 1]
        return tf.nn.conv2d(input_data, weights, strides=strides, padding=padding, name="conv") + biases


def create_pull(input_data, patch_size=2, stride=None, padding="VALID"):
    """
    Creates a pulling layer.
    'patch_size', The used patch is square and has a size defined by patch_size.
    'stride', sets the stride size, and is set equal to patch_size when not defined.
    'padding', sets the padding using the string 'VALID' or 'SAME'.

    returns the created layer
    """
    if stride is None:
        stride = patch_size

    with tf.name_scope("Pull"):
        ksize = [1, patch_size, patch_size, 1]
        strides = [1, stride, stride, 1]
        return tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding=padding, name="pool")


def create_fully_connected(input_data, output_count, mu=0, sigma=0.05):
    """
    creates an fully connected layer.
    'output_count', sets the amount of outputs for the created layer.
    'mu' 'sigma', specifies the parameters for the random initialization of the weights.

    returns the created layer
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

    returns the created layer
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
    'alpha' slope of the activation function
    'tensorboard', stores leaky relu output in tensorboard. This function only works properly with a batch size = 1

    returns the created layer
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


def create_flatten(input_data):
    """Creates a layer which flattens the data shape."""
    with tf.name_scope("Flatten"):
        result = tf.contrib.layers.flatten(input_data)
        return result


def create_reshape(input_data, output_shape):
    """Creates a layer which reshapes the input data to a given shape. The shape is specifies as list of integers."""
    with tf.name_scope("Reshape"):
        result = tf.reshape(input_data, output_shape)
        return result


def create_iou(x_1, y_1, w_1, h_1, x_2, y_2, w_2, h_2):
    """
    Creates a layer which returns the iou's between two area's.
    :param x_1: bbox center position x
    :param y_1: bbox center position y
    :param w_1: bbox width
    :param h_1: bbox height
    :param x_2: bbox center position x
    :param y_2: bbox center position y
    :param w_2: bbox width
    :param h_2: bbox height
    :return: iou
    """
    min_w = tf.maximum(x_1 - (w_1 / 2.0), x_2 - (w_2 / 2.0))
    max_w = tf.minimum(x_1 + (w_1 / 2.0), x_2 + (w_2 / 2.0))
    min_h = tf.maximum(y_1 - (h_1 / 2.0), y_2 - (h_2 / 2.0))
    max_h = tf.minimum(y_1 + (h_1 / 2.0), y_2 + (h_2 / 2.0))

    overlap_w = max_w - min_w
    overlap_h = max_h - min_h

    overlap_w = create_negative_filter(overlap_w)
    overlap_h = create_negative_filter(overlap_h)

    overlap_area = (overlap_w * overlap_h)
    total_area = (w_1 * h_1) + (w_2 * h_2) - overlap_area

    return create_nan_filter(overlap_area / total_area)


def create_nan_filter(tensor):
    """Creates a layer which replace NaN's with zero's."""
    return tf.where(tf.is_nan(tensor), tf.zeros_like(tensor), tensor)


def create_negative_filter(tensor):
    """Creates a layer which replace negative value's with zero's."""
    return tf.maximum(tensor, tf.constant(0.0))


class ModelCreator:
    """
    The ModelCreator can create a neural network model from a configuration. The configuration is specified
    as follow:
    [  ['Layer_type', {'layer_param_x' : 0.025, 'layer_param_y' : 0.025, 'layer_param_z' : '__variable_param'}],
       ['Layer_type', {'layer_param_x' : 0.0, 'layer_param_y' : 1}],
       ['Layer_type', {'layer_param_y' : 0.2, 'layer_param_z' : 'str_parameter'}],
       ['Layer_type', {'layer_param_x' : 0.025, 'layer_param_y' : 8.05}]
    ]

    The Layer_type specifies the type of layer to be created. The parameters in the dictionary for each layer contain
    the layer settings.

    Note the last parameter value of the first layer. Every value with type string starting with 2 underscores, is
    considered a variable parameter value. The parameter value of the parameter with the '__*name*' will be replaced
    by the value of the parameter given as kwark parameter with the create_model() function call.

    This means that after executing of the following code, will the '__param' value be replaced with the value 10.01.
    The value of Layer1_type, parameter layer_param_z will be 10.01:

    config = [  ['Layer1_type', {'layer_param_x' : 0.025, 'layer_param_y' : 0.025, 'layer_param_z' : '__param'}],
                ['Layer2_type', {'layer_param_x' : 0.0, 'layer_param_y' : 1}]
             ]

    input = 'model input tensor'

    modelcreator.create_model(config, input, __param=10.01)

    The input tensor will be assigned to the first layer. All next layers will receive the output of the previous layer
    as input.

    #########
    # USAGE #
    #########

    m = ModelCreator()
    m.add_layer_type(conv_layer_creator,    'conv')
    m.add_layer_type(reshape_layer_creator, 'reshape')

    config = [['conv',    {'depth' : 3, 'stride' : 2}],
              ['reshape', {'output_shape': '__output_shape'}]]

    input = 'model input tensor'

    model = m.create_model(config, input, __output_shape=[-1, 2, 2, 3])

    """
    def __init__(self):
        self.layer_type_dict = {'conv': create_conv,
                                'leaky_relu': create_leaky_relu,
                                'pull': create_pull,
                                'flatten': create_flatten,
                                'fully_connected': create_fully_connected,
                                'reshape': create_reshape,
                                'relu': create_relu}

    def add_layer_type(self, layer_creator_func, name):
        """
        Add a layer type.
        :param layer_creator_func: contains a function, which will create a layer.
        :param name: contains the name of the layer type, which can be used in the model configuration.
        """
        self.layer_type_dict[name] = layer_creator_func

    def create_model(self, model_config, model_input, **kwargs):
        """
        Creates a model by the given model configuration. Variable parameters can be supplied via the **kwarks input
        of the model.

        A model configuration should have the following format:
        [  ['Layer_type', {'layer_param_x' : 0.025, 'layer_param_y' : 0.025, 'layer_param_z' : '__variable_param'}],
           ['Layer_type', {'layer_param_x' : 0.0, 'layer_param_y' : 1}],
           ['Layer_type', {'layer_param_y' : 0.2, 'layer_param_z' : 'str_parameter'}],
           ['Layer_type', {'layer_param_x' : 0.025, 'layer_param_y' : 8.05}]
        ]

        The parameter value's, starting with 2 underscores, will be replaces by the value of the kwark parameters with
        the same name.

        :param model_config: contains the model configuration.
        :param model_input: contains the model input. The input will be assigned to the first layer. All next layers
                                will receive the output of the previous layer as input.
        :param kwargs: contains variables which can be used in the model configuration. variable parameter names should
                        start with 2 underscores.
        :return: returns the model
        """
        process_value = model_input
        for key in kwargs:
            if not key.startswith('__'):
                raise NameError("Variable parameter does not start with \'__\'")

        for layer_type, layer_parameters in model_config:
            for key in list(layer_parameters):
                value = layer_parameters[key]
                if type(value) is str and value.startswith('__'):
                    layer_parameters[key] = kwargs[value]

            process_value = self.layer_type_dict[layer_type](process_value, **layer_parameters)
            print(process_value)
        return process_value
