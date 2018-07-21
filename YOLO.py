"""
Total implementation.
"""

from NeuralNetworkLibrary.YoloLib.YoloLib import *
from NeuralNetworkLibrary.YoloLib.YoloToolsLib import *
from DataLibrary.DataGenerationLib import *
from BasicLib.BasicFunctions import *
from DataLibrary import COCO
import sklearn.model_selection

# param_ = parameters
# img   = image
# outp  = output
# coco  = Microsoft coco dataset interface
# loc   = location
# dir   = directory
# _x    = input data
# _y    = output data (label)
# dict  = dictionary
# cat   = category
# lbl   = label
# _n    = count
# pix   = pixel
# pos   = position
# conf  = confidence
# op    = tensorflow operation
# pred  = predict
# predd = predicted (prediction result)


###############
# ENVIRONMENT #
###############

print_tf_stats()

##############
# Parameters #
##############

# Model parameters
##################

# set network model description
param_model_name = "YOLO"

# set location model to use. New model is created is string is empty
# param_saved_model_file = "models\\.ckpt"
param_saved_model_file = ""

# set input image size (width, height, channels)
param_input_img_shape = (350, 350, 3)

# set learning_rate parameters
# A training run for each learning_rate will with the corresponding learning_rate_dacay_factor will be executes
param_learning_rate = 0.001
param_learning_rate_dacay_factor = 0.1

# set batch_size parameters
# A training run for each batch size will be executes. Each batch size will be run for all learning rates.
param_batch_size = 32

# set number of epochs
param_epochs = 100

# compute model accuracy average over n batches
param_accuracy_over_n_batches = 50

# compute model accuracy every n training batches
param_accuracy_each_n_batches = 500

# set yolo specific parameters
param_yolo_outp_cell_grid_size = (10, 10)
param_yolo_anchors_definition = [(70, 225), (150, 40), (40, 150), (40, 100), (50, 50), (100, 300)]

# data parameters
#################

# set category filters filters
param_coco_cat_filters = [['person'], ['car'], ['bus'], ['truck']]

# set coco dataset locations
param_coco_annotation_file = '..\\COCO\\annotations_trainval2017\\annotations\\instances_train2017.json'
param_coco_img_dir = '..\\COCO\\annotations_trainval2017\\images\\train2017\\'

# directories
#############

# set directory's
param_model_store_dir = "models/"


################
# Prepare data #
################

# load data set
coco = COCO.CocoDatasetInterface(param_coco_annotation_file, param_coco_img_dir)
data_x, data_y, data_dict_cat = coco.get_category_labeled_images(param_coco_cat_filters)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data_x,
                                                                            data_y,
                                                                            test_size=0.05,
                                                                            random_state=85)

yolo_lbl_creator = YoloLabelCreator(cat_count=len(data_dict_cat),
                                    img_size=(param_input_img_shape[0],
                                              param_input_img_shape[1]),
                                    cell_grid=param_yolo_outp_cell_grid_size,
                                    anchor_descriptors_list=param_yolo_anchors_definition,
                                    flatten=False)

yolo_data_augmenter = DataAugmenter(outp_img_size=(param_input_img_shape[0],
                                                   param_input_img_shape[1]))

train_data_generator_creator = DataGenerator(x_train,
                                             y_train,
                                             yolo_data_augmenter.augment_yolo_data,
                                             yolo_lbl_creator.bbox_to_yolo_label,
                                             batch_size=param_batch_size)

test_on_train_data_generator_creator = DataGenerator(x_train,
                                                     y_train,
                                                     yolo_data_augmenter.augment_yolo_data,
                                                     bbox_splitter,
                                                     batch_size=param_batch_size)

test_data_generator_creator = DataGenerator(x_test,
                                            y_test,
                                            yolo_data_augmenter.augment_yolo_data,
                                            bbox_splitter,
                                            batch_size=param_batch_size)

param_model_outp_shape = yolo_lbl_creator.get_shape()
param_n_classes = yolo_lbl_creator.get_category_count()
yolo_cell_pix_size = [param_input_img_shape[0] / param_yolo_outp_cell_grid_size[0],
                      param_input_img_shape[1] / param_yolo_outp_cell_grid_size[1]]
print("class count: ", param_n_classes)
print("model set output shape: ", param_model_outp_shape)
print("yolo cell size", yolo_cell_pix_size)

##############
# LOAD MODEL #
##############
model = YoloObjectDetection(param_input_img_shape, param_model_outp_shape, param_n_classes, yolo_cell_pix_size)
model.init(pre_trained_model=param_saved_model_file)

###############
# TRAIN MODEL #
###############

# define training session name
sess_name = get_time() + " " + param_model_name + " lr_ {} lrdcay_ {} ep_ {} btch_sz_ {}".format(
    param_learning_rate, param_learning_rate_dacay_factor, param_epochs, param_batch_size)

# define storage information trained models
model_store_dir = param_model_store_dir + sess_name

# initialize variables
learn_rate_value_with_decay = param_learning_rate
epoch = 0
last_epoch = 0
test_precision = 0.0
test_recall = 0.0

# start training epochs
# print("start training model \"" + sess_name + "\"")
while epoch < param_epochs and param_saved_model_file == "":
    # train model
    train_data_generator = train_data_generator_creator.get_data_generator(param_accuracy_each_n_batches)
    train_result = model.train(train_data_generator, learn_rate_value_with_decay)

    # compute accuracy in batches
    test_data_generator = test_data_generator_creator.get_data_generator(param_accuracy_over_n_batches, verbose=False)
    test_precision, test_recall = model.test(test_data_generator)

    print()
    print("precision on test data: ", np.array(test_precision))
    print("recall on test data:    ", np.array(test_recall))
    print()

    test_on_train_data_generator = test_on_train_data_generator_creator.get_data_generator(param_accuracy_over_n_batches, verbose=False)
    train_precision, train_recall = model.test(test_on_train_data_generator)

    print()
    print("precision on train data: ", np.array(train_precision))
    print("recall on train data:    ", np.array(train_recall))
    print()

    # decrease the learning rate over time
    epoch, batch, batch_per_epoch = train_data_generator_creator.get_status()
    if epoch > last_epoch or batch >= batch_per_epoch:
        learn_rate_value_with_decay = learn_rate_value_with_decay * (1 - param_learning_rate_dacay_factor)
        last_epoch += 1
        save_path = "{} prc_{:.4f} rcl_{:.4f} ep_{}.ckpt".format(model_store_dir, np.max(test_precision),
                                                                 np.max(test_recall), epoch)
        model.save(save_path)
        print("model stored as \"" + save_path + "\"")
        print("learning rate: ", learn_rate_value_with_decay)


# test model
test_data_generator = test_data_generator_creator.get_data_generator(10)
image, label = next(test_data_generator)
prediction = model.predict(image)

# display images in predicted batch
for image, certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list in zip(image, *prediction):
    show_yolo_result(image, certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list, 0.1, param_n_classes)

# display images with labels from dataset
# for image, pred in zip(image, label):
#     x1_list, y1_list, x2_list, y2_list, cat_list = pred
#     certainty_list = []
#     for i in x1_list:
#         certainty_list.append(1)
#     show_yolo_result(image, certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list, 0.1, param_n_classes)
