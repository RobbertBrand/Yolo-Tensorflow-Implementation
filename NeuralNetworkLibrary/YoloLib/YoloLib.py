"""
Module contains all functions to create, train, test and use a yolo network.

#########
# USAGE #
#########

# set location of saved yolo model to use. A new model is created if string is empty.
# param_saved_model_file = "models\\trained_model_1.ckpt"
# NOTE: Only new models can be trained. A model loaded from disk can only be used for prediction. A new model is created
# when param_saved_model_file is None. A model will be loaded from disk (for prediction) when param_saved_model_file
# contains a model location as a string.
param_saved_model_file = None

# set input image size (width, height, channels)
param_input_img_shape = (350, 350, 3)

# set model learning_rate parameter, used during model training
param_learning_rate = 0.001

## set yolo specific parameters
# set the yolo model output grid size
param_yolo_outp_cell_grid_size = (10, 10)

# set amount of object classes to predict
param_n_classes = 2

# specify yolo model output data shape
# yolo output shape is the [amount of horizontal cells,
                            by amount of vertical cells,
                            by length of a single bbox prediction times the amount of predictions per cell]
param_model_outp_shape = [param_yolo_outp_cell_grid_size[0],
                          param_yolo_outp_cell_grid_size[1],
                          (5 + param_n_classes) * n_anchors]

yolo_cell_pix_size = [param_input_img_shape[0] / param_yolo_outp_cell_grid_size[0],
                      param_input_img_shape[1] / param_yolo_outp_cell_grid_size[1]]

# specify the model configuration. The parameter value '__yolo_output_shape', will be replaces by the required model
# data output shape. The parameter value '__output_neuron_count' will be replaces with the
# amount of neurons in the output shape as integer.
model_config = [  ['Layer_type', {'layer_param_y' : 0.025, 'layer_param_z' : -0.258}],
                  ['Layer_type', {'layer_param_x' : 0.0, 'layer_param_y' : 1}],
                  ['Layer_type', {'layer_param_y' : 0.2, 'layer_param_z' : 'str_parameter'}],
                  ['fully_connected', {'output_count': __output_neuron_count}]
                  ['reshape', {'output_shape': '__yolo_output_shape'}],
                ]

train_data_generator = "a python data generator returning each generation, a batch of images as a list and a batch
                        of yolo output labels as a list. The yolo output labels are equal to the preferred yolo model
                        output"     # yolo output data can be generated with the YoloToolsLib.YoloLabelCreator class.

test_data_generator = "a python data generator returning each generation, a batch of images as a list and a batch
                       of labels as a list matching the images. Each labels batch 'list item' containing 5 lists:
                       x1, y1, x2, y2 and category. The x1 and y1 list defines the left top positions of a bboxes for
                       an image, x2 and y2 define the right bottom positions. Category specifies the category as a
                       integer being 0 to (param_n_classes - 1).
                       labels batch = [ [x1, y1, x2, y2, cat], ...]
                       x1 and y1 and x2 and y2 = [pixel_pos, ...]
                       category = [cat_nr, ...]
                       "


## declare model
model = YoloObjectDetection(model_config, param_input_img_shape, param_model_outp_shape, param_n_classes,
                            yolo_cell_pix_size, pre_trained_model=param_saved_model_file)

# train model until data generator depletes. model.train can be run multiple times to continue training.
# NOTE: Only new models can be trained. A new model is created when the pre_trained_model parameter during
# initialization is None.
model.train(train_data_generator, param_learning_rate)

test_precision, test_recall = model.test(test_data_generator)

# all trained variables will be saves. (Variables like weights and biases.)
model.save(save_path)

# use model to make predictions on images.
prediction = model.predict([image])

# prediction returns 6 lists: prediction_certainty, x1, y1, x2, y2, category_numeric. All in format:
#              [[X1_bbox value's for first data point], [X1_bbox value's for second data point], ...]
# Note that this slightly differs from the test data generator format! For the test data generator is the labels
# list as long as the the batch size and contains each list item (data point) 5 lists with bbox specification. The
# prediction returns 6 lists, all containing the x1 OR y1 OR X1... values. Each of these 6 lists contains as many items
# as the batch size.


###############
# DATA FORMAT #
###############

Test data format:
          first data point:    second data point:   ...   n data point:
labels =  [[x1,y1,x2,y2,cat],   [x1,y1,x2,y2,cat],   ...,  [x1,y1,x2,y2,cat]]
          each x1,y1,x2,y2,cat being a list with an item for each bbox for that data point (image)

Prediction data format:

prediction = [cert, x1, y1, x2, y2, cat]

             cert = [[cert_bbox value's for first data point], [cert_bbox value's for second data point], ...]
             x1 = [[X1_bbox value's for first data point], [X1_bbox value's for second data point], ...]
             ...
             cat = [[cat_bbox value's for first data point], [cat_bbox value's for second data point], ...]

             each "*_bbox value's for * data point" being a list with as much value's as found bboxes for that data
             point (image).

cert =              being the prediction certainty for a single bbox. higher is more certain.
x1, y1, x2, y2 =    The x1 and y1 value's define the left top positions of a bboxes, x2 and y2 define the right bottom
                        bbox positions.
cat =               Category specifies the category as a integer being 0 to (param_n_classes - 1).

"""
from NeuralNetworkLibrary.NeuralNetworkLib import *


######################################
# Yolo output data processing layers #
######################################


def yolo_unpack_output(data, outp_cat_count, enable_activation):
    """
    Add's a yolo model output data preparation layer to the yolo tensorflow pipeline. Unpacks yolo data in to separate
    parts. This separated data is much easier to use and process than the compact yolo model output data.

        yolo output data in format:
        [[x_cell_1 [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]],
        ...
        [x_cell_n [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]]]

        with anchor_n in format:
        [object_detected 1 or 0, bbox_center_x, bbox_center_y, bbox_width, bbox_height, category_0, category_n]

    :param data: yolo formatted data
    :param outp_cat_count: amount of classes (detection categories) as integer
    :param enable_activation: enable the output activation
                                sigmoid in iou, x and y
                                exp on w and h
                                softmax on classes
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
        classes = tf.nn.softmax(classes, axis=3)

    return iou, x, y, w, h, classes


def yolo_pred_to_bbox(logits, n_classes, cell_pixel_width, cell_pixel_height):
    """
    Add's a yolo model output data preparation layer to the yolo Tensorflow pipeline. Data preparation is necessary to
    create a more standardized and a easier to handle data format. This pre-processed data requires less processing in
    the final application than the original data format does.

    This output layer add's:
     - translation from yolo bbox positions and sizes to image pixel bbox positions and sizes.
     - bbox coordinates in x1 y1 and x2 y2 coordinates, instead of Xcenter Ycenter Width Height.
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
    Add's a yolo loss function to the yolo tensorflow pipeline. The loss function is required to determine the "error"
    in the yolo model output. The model can be adjusted via back-propagation (learn), based on the loss function and its
    determined error.

    Returns position_loss, size_loss, category_loss,
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
        confidence_score = create_iou(label_x, label_y, label_w, label_h, predicted_x, predicted_y, predicted_w,
                                      predicted_h)

        confidence_loss = tf.reduce_sum(object_in_anchor * tf.pow(predicted_iou - confidence_score, 2)) + (
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


#################
# Yolo frontend #
#################


class YoloObjectDetection:
    """
    This class forms an easy to use front-end between the yolo model library and the implementation. This class can be
    used to easily create, train, test, store, load and use a yolo model.
    """

    def __init__(self, model_config, param_input_img_shape, param_model_outp_shape, param_n_classes,
                 yolo_cell_pix_size, pre_trained_model=None):
        """
        Initializes the yolo front-end.
        :param model_config: contains the model configuration in format:
                                [  ['Layer_type', {'layer_param_y' : 0.025, 'layer_param_z' : -0.258}],
                                   ['Layer_type', {'layer_param_x' : 0.0, 'layer_param_y' : 1}],
                                   ['Layer_type', {'layer_param_y' : 0.2, 'layer_param_z' : 'str_parameter'}],
                                   ['fully_connected', {'output_count': __output_neuron_count}]
                                   ['reshape', {'output_shape': '__yolo_output_shape'}]
                                ]
                                The parameter value '__yolo_output_shape', will be replaces by the required model data
                                output shape. The parameter value '__output_neuron_count' will be replaces with the
                                amount of neurons in the output shape as integer.
        :param param_input_img_shape: defines the shape of the input images in format (width, height) in pixels
        :param param_model_outp_shape: contains a list with the yolo output dimensions of a prediction for a single data
            point. This should be a yolo compatible shape:
            [x, y, anchor_depth]
            NOTE: the batch dimension will be added by the function

            output data in format:
            [[x_cell_1 [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]],
            ...
            [x_cell_n [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]], ..., batch_n]

            with anchor in format:
            [object_detected 1 or 0, bbox_center_x, bbox_center_y, bbox_width, bbox_height, category_0, category_n]
        :param param_n_classes: sets amount of classes (categories) to predict
        :param yolo_cell_pix_size: sets the size of the yolo output cells in pixels
        :param pre_trained_model: When None will the train and prediction pipeline be initiated and a new model be
                                    trained.
                                    When instead of None, a model file location is given as string is only the
                                    prediction pipeline initiated.
                                    Training of a currently saved model is not possible because training variables
                                    are not saved when saving the model.
        """
        self.train_and_predict = pre_trained_model is None

        # reset tensor flow graph
        tf.reset_default_graph()

        # declare place holders
        with tf.name_scope("PlaceHolders"):
            # set place holders                  shape = [batch, in_height, in_width, in_channels]
            self.features = tf.placeholder("float32", shape=[None, param_input_img_shape[1], param_input_img_shape[0],
                                                             param_input_img_shape[2]], name="features")

        # get model
        output_neuron_count = 1
        for i in param_model_outp_shape:
            output_neuron_count *= i

        model_creator = ModelCreator()
        self.logits = model_creator.create_model(model_config, self.features,
                                                 __output_neuron_count=output_neuron_count,
                                                 __yolo_output_shape=[-1] + param_model_outp_shape)

        # build prediction pipeline
        with tf.name_scope("PredictionPipeLine"):
            # self.iou_op, self.x1_op, self.y1_op, self.x2_op, self.y2_op, self.cat_op = self.prediction_ops
            self.prediction_ops = yolo_pred_to_bbox(self.logits, param_n_classes,
                                                    yolo_cell_pix_size[0], yolo_cell_pix_size[1])

        if self.train_and_predict:
            # declare training place holders
            with tf.name_scope("TrainPlaceHolders"):
                # set place holders                  shape = [batch, in_height, in_width, in_channels]
                self.labels = tf.placeholder("float32", shape=[None, *param_model_outp_shape], name="labels")
                self.rate = tf.placeholder("float32", shape=[], name="learning_rate")

            # build training pipeline
            with tf.name_scope("TrainPipeLine"):
                self.loss_ops = yolo_loss(self.logits, self.labels, param_n_classes)
                self.pos_loss_op, self.size_loss_op, self.cat_loss_op, self.conf_loss_op = self.loss_ops
                self.total_loss_op = self.pos_loss_op + self.size_loss_op + self.cat_loss_op + self.conf_loss_op
                self.training_op = tf.train.AdamOptimizer(learning_rate=self.rate, name="ADAM").minimize(
                    self.total_loss_op)

        self.model_save_op = tf.train.Saver(var_list=tf.trainable_variables())

        self.sess = tf.Session()

        if self.train_and_predict:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.model_save_op.restore(self.sess, pre_trained_model)

    def train(self, train_data_generator, learn_rate=0.001):
        """
        Train the model on the data provided by the data generator. This function runs until the data generator
        depletes. The data generator should return a x_batch containing a list of images and y_batch containing a list
        of yolo model output labels. Yolo label in format:
                [[x_cell_1 [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]],
                ...
                [x_cell_n [y_cell_1 [anchor_1], [anchor_ndepth]], [y_cell_n [anchor_1], [anchor_ndepth]]]]

                with anchor in format:
                [object_detected 1 or 0, bbox_center_x, bbox_center_y, bbox_width, bbox_height, category_0, category_n]
        :param train_data_generator: contains data generator
        :param learn_rate: sets learning rate
        """
        if not self.train_and_predict:
            raise EnvironmentError("A model loaded from a saved location can not be trained.")

        for batch_x, batch_y in train_data_generator:
            # perform forward and backward propagation and gather data
            res_total_loss, res_position_loss, res_size_loss, res_category_loss, res_confidence_loss, _ = self.sess.run(
                [self.total_loss_op, *self.loss_ops, self.training_op],
                feed_dict={self.features: batch_x, self.labels: batch_y, self.rate: learn_rate})

            # display epoch details
            print("LOSS: pos {:8.3f} size {:8.3f} cat {:8.3f} conf {:8.3f} tot {:8.3f}".format(
                res_position_loss, res_size_loss, res_category_loss, res_confidence_loss, res_total_loss))

    def test(self, test_data_generator, certainty_thresholds=(0.1, 0.2, 0.3, 0.4, 0.6)):
        """
        Test the model on the data provided by the data generator. This function runs until the data generator
        depletes. The data generator should return a x_batch containing a list om images and y_batch containing a list
        of labels. Test labels format:
                  first data point:    second data point:   ...   n data point:
                  [[x1,y1,x2,y2,cat],   [x1,y1,x2,y2,cat],   ...,  [x1,y1,x2,y2,cat]]
                  each x1,y1,x2,y2,cat being a list with an item for each bbox for that data point (image)

        The function returns the precision and recall found.

        :param test_data_generator: contains data generator
        :param certainty_thresholds: contains a list of certainty thresholds to determine precision and recall against.
        :return: precision and recall, per certainty threshold
        """
        test_recall_list = []
        test_precision_list = []
        for test_batch_x, test_batch_y in test_data_generator:

            predd_batch = self.predict(test_batch_x)

            # compute the performance for each batch
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
                                                                       certainty_thresholds)
                test_precision_list.append(pred_precision)
                test_recall_list.append(pred_recall)

        test_precision = np.mean(test_precision_list, axis=0)
        test_recall = np.mean(test_recall_list, axis=0)
        return test_precision, test_recall

    def predict(self, input_image_batch):
        """
        Predict bboxes for batch of input images using yolo model. Prediction is returned in 6 lists:
        predd_certainty, predd_x1, predd_y1, predd_x2, predd_y2, predd_cat. Each list is built as following:
            [[batch_first_image bbox value, batch_first_image bbox value, ...], [batch_second_image bbox value, ...]]
        :param input_image_batch: list of images
        :return: lists: predd_certainty, predd_x1, predd_y1, predd_x2, predd_y2, predd_cat
        """
        return self.sess.run([*self.prediction_ops], feed_dict={self.features: input_image_batch})

    def save(self, save_path):
        """
        save model to given file location.
        :param save_path: file name and location to store the trained model
        """
        self.model_save_op.save(self.sess, save_path)
