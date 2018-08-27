"""
Train new Yolo Model.
"""

from NeuralNetworkLibrary.YoloLib.YoloLib import *
from NeuralNetworkLibrary.YoloLib.YoloToolsLib import *
from DataLibrary.DataGenerationLib import *
from BasicLib.BasicFunctions import *
from DataLibrary import COCO
from BasicLib.ConfgurationManger import ConfigurationManager
import sklearn.model_selection
import logging as log
import argparse
import os

##############
# DICTIONARY #
##############

################
# VARIABLE NAMES

# param_ = parameters
# img   = image
# outp  = output
# coco  = Microsoft coco data set interface
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
# op    = Tensorflow operation
# pred  = predict
# predd = predicted (prediction result)

#######
# TERMS

# data point = is a single data point, like one image or one list of bboxes matching this one image.

##############
# Parameters #
##############

# commandline parameter
command_line_parser = argparse.ArgumentParser(description='Yolo model training module.')
command_line_parser.add_argument('-n', '--model_name', help='Name of the Yolo model.', type=str, default='New Model')
command_line_parser.add_argument('-c', '--config-file', help='Configuration file location.', type=str,
                                 default='config.ini')
command_line_args = command_line_parser.parse_args()

# configuration file parameters
config = ConfigurationManager(command_line_args.config_file)

config['Model.meta']['model_name'] = command_line_args.model_name
config['Model.meta']['model_start_date'] = get_time()

# define storage information trained models
param_model_store_dir = os.path.join(config['System']['model_store_dir'], config['Model.meta']['model_name'] + " " +
                                     config['Model.meta']['model_start_date'] + os.sep)

##################
# CREATE PROJECT #
##################

if not os.path.exists(param_model_store_dir):
    print("Project created")
    os.makedirs(param_model_store_dir)
    param_saved_model_file = ""

###########
# LOGGING #
###########

log.basicConfig(filename=param_model_store_dir + 'model.log',
                level=log.DEBUG,
                format='%(asctime)s %(levelname)s:   %(message)s',
                datefmt='%m-%d-%Y %H:%M:%S')

# Duplicate logging stream to console. logs will be printed and logged to a file.
log_stream = log.StreamHandler()
log_stream.setLevel(log.INFO)
log_stream_format = log.Formatter('%(message)s')
log_stream.setFormatter(log_stream_format)
log.getLogger('').addHandler(log_stream)

log.info("Training Model {}".format(config['Model.meta']['model_name']))
log.info("Model storage dir: {}".format(param_model_store_dir))
# log.info("Configuration:\n{}".format(config.get_parameter_set_as_pretty_string()))

###############
# ENVIRONMENT #
###############

log.info("\n" + get_tf_stats_as_pretty_string())

################
# Prepare data #
################

# load data
coco = COCO.CocoDatasetInterface(config['DataSet.coco']['annotation_file'], config['DataSet.coco']['img_dir'])
img_size_list = coco.get_image_sizes(config['DataSet.coco']['cat_filters'])
data_x, data_y, data_dict_cat = coco.get_category_labeled_images(config['DataSet.coco']['cat_filters'],
                                                                 print_func=log.info)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data_x,
                                                                            data_y,
                                                                            test_size=config['DataSet']['test_size'],
                                                                            random_state=85)

# determine optimized anchor definition
print("determining optimized anchor box definition...")
opti_anchor_def, anchor_coco_iou = compute_anchor_bboxes(img_size_list,
                                                         data_y,
                                                         config['Model.input']['input_img_shape'],
                                                         quit_n_anchors=config['Model.tune']['anchors_max'],
                                                         quit_min_iou_gain=config['Model.tune']['anchor_min_iou_gain'],
                                                         quit_min_iou=config['Model.tune']['anchor_min_iou'])

config['Model.output']['yolo_anchors_definition'] = opti_anchor_def

# create data generators
yolo_lbl_creator = YoloLabelCreator(cat_count=len(data_dict_cat),
                                    img_size=(config['Model.input']['input_img_shape'][0],
                                              config['Model.input']['input_img_shape'][1]),
                                    cell_grid=config['Model.output']['yolo_outp_cell_grid_size'],
                                    anchor_descriptors_list=config['Model.output']['yolo_anchors_definition'],
                                    flatten=False)

yolo_train_data_proc = YoloDataFormatter(label_processor_func=yolo_lbl_creator.bbox_to_yolo_label,
                                         outp_img_size=(
                                             config['Model.input']['input_img_shape'][0],
                                             config['Model.input']['input_img_shape'][1]))

yolo_test_data_proc = YoloDataFormatter(label_processor_func=bbox_splitter,
                                        outp_img_size=(
                                            config['Model.input']['input_img_shape'][0],
                                            config['Model.input']['input_img_shape'][1]))

train_data_generator_creator = DataGenerator(x_train,
                                             y_train,
                                             yolo_train_data_proc.process_data_to_yolo_format,
                                             batch_size=config['Train.learn']['batch_size'])

test_on_train_data_generator_creator = DataGenerator(x_train,
                                                     y_train,
                                                     yolo_test_data_proc.process_data_to_yolo_format,
                                                     batch_size=config['Train.evaluate']['batch_size'])

test_data_generator_creator = DataGenerator(x_test,
                                            y_test,
                                            yolo_test_data_proc.process_data_to_yolo_format,
                                            batch_size=config['Train.evaluate']['batch_size'])

# update configuration
config['Model.output']['yolo_outp_shape'] = yolo_lbl_creator.get_shape()
config['Model.output']['yolo_n_classes'] = yolo_lbl_creator.get_category_count()
config['Model.output']['yolo_cell_pix_size'] = [
    config['Model.input']['input_img_shape'][0] / config['Model.output']['yolo_outp_cell_grid_size'][0],
    config['Model.input']['input_img_shape'][1] / config['Model.output']['yolo_outp_cell_grid_size'][1]]

# return statistics
log.info("anchor boxes defined: {}".format(len(config['Model.output']['yolo_anchors_definition'])))
log.info("bbox over anchor iou: {}".format(anchor_coco_iou))
log.info("anchor definition: {}".format(config['Model.output']['yolo_anchors_definition']))

log.info("class count: {}".format(config['Model.output']['yolo_n_classes']))
log.info("model set output shape: {}".format(config['Model.output']['yolo_outp_shape']))
log.info("yolo cell size{}".format(config['Model.output']['yolo_cell_pix_size']))

##############
# LOAD MODEL #
##############

model = YoloObjectDetection(config['Model']['definition'],
                            config['Model.input']['input_img_shape'],
                            config['Model.output']['yolo_outp_shape'],
                            config['Model.output']['yolo_n_classes'],
                            config['Model.output']['yolo_cell_pix_size'],
                            pre_trained_model=None)

#################
# SAVE SETTINGS #
#################

config.save(param_model_store_dir + 'config.ini')

###############
# TRAIN MODEL #
###############

learn_rate_value_with_decay = config['Train.learn']['learn_rate']
epoch = 0
last_epoch = 0
test_precision = 0.0
test_recall = 0.0

# start training epochs
while epoch < config['Train.learn']['epochs']:
    # train model
    train_data_generator = train_data_generator_creator.get_data_generator(
        config['Train.evaluate']['accuracy_each_n_batches'])
    model.train(train_data_generator, learn_rate_value_with_decay)

    # compute accuracy in batches
    test_data_generator = test_data_generator_creator.get_data_generator(
        config['Train.evaluate']['accuracy_over_n_batches'], verbose=False)
    test_precision, test_recall = model.test(test_data_generator)

    log.info("On test data,  precision: {} recall: {}".format(format_list_of_floats(test_precision),
                                                              format_list_of_floats(test_recall)))

    # compute accuracy in batches
    test_on_train_data_generator = test_on_train_data_generator_creator.get_data_generator(
        config['Train.evaluate']['accuracy_over_n_batches'], verbose=False)
    train_precision, train_recall = model.test(test_on_train_data_generator)

    log.info("On train data, precision: {} recall: {}".format(format_list_of_floats(train_precision),
                                                              format_list_of_floats(train_recall)))

    # decrease the learning rate over time
    epoch, batch, batch_per_epoch = train_data_generator_creator.get_status()
    if epoch > last_epoch or batch >= batch_per_epoch:
        learn_rate_value_with_decay = learn_rate_value_with_decay * (
                1 - config['Train.learn']['learn_rate_dacay_factor'])
        last_epoch += 1

        save_file_name = "{} prc_{:.4f} rcl_{:.4f} ep_{}.ckpt".format(get_time(), np.max(test_precision),
                                                                      np.max(test_recall), epoch)

        model.save(param_model_store_dir + save_file_name)
        log.info("model stored as \" {}\"".format(save_file_name))
        log.info("epoch: {} precision: {:.4f} recall: {:.4f} learning rate: {}".format(epoch, np.max(test_precision),
                                                                                       np.max(test_recall),
                                                                                       learn_rate_value_with_decay))

        config.save(param_model_store_dir + 'config.ini')
