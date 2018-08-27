"""
Predict with trained yolo model
"""

import os
# GPU usage is turned off because models trained in the cloud are likely to be to large for the local GPU.
# COMMENT OR REMOVE THE NEXT LINE TO ENABLE GPU PROCESSING
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from NeuralNetworkLibrary.YoloLib.YoloLib import *
from NeuralNetworkLibrary.YoloLib.YoloToolsLib import *
from BasicLib.ConfgurationManger import ConfigurationManager
from DataLibrary.DataGenerationLib import DataGenerator
import argparse
import cv2

##############
# Parameters #
##############

command_line_parser = argparse.ArgumentParser(description='Yolo model prediction module.')
command_line_parser.add_argument('model_location', help='Yolo model file location.', type=str)
command_line_parser_inp_src = command_line_parser.add_mutually_exclusive_group()
command_line_parser_inp_src.add_argument('--image', help='Sets image folder to feed images to the yolo model.',
                                         type=str)

command_line_args = command_line_parser.parse_args()

param_model_store_dir = command_line_args.model_location

config_file = os.path.join(os.path.split(param_model_store_dir)[0], 'config.ini')
config = ConfigurationManager(config_file)
config.print_parameter_set()

##############
# LOAD MODEL #
##############

model = YoloObjectDetection(config['Model']['definition'],
                            config['Model.input']['input_img_shape'],
                            config['Model.output']['yolo_outp_shape'],
                            config['Model.output']['yolo_n_classes'],
                            config['Model.output']['yolo_cell_pix_size'],
                            pre_trained_model=param_model_store_dir)

###########
# PREDICT #
###########
cat_colors = [get_random_color() for _ in range(config['Model.output']['yolo_n_classes'])]
if command_line_args.image is None:
    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        ret, image = capture.read()
        if cv2.waitKey(1) == 27 or not ret:
            break

        image, image_norm = yolo_prepare_image(image, (config['Model.input']['input_img_shape'][0],
                                                       config['Model.input']['input_img_shape'][1]))

        certainty_list, x1_list, y1_list, x2_list, y2_list, cat_list = model.predict([image_norm])

        image = augment_image_with_yolo_prediction(image, certainty_list[0], x1_list[0], y1_list[0], x2_list[0],
                                                   y2_list[0], cat_list[0], 0.1, cat_colors)

        cv2.imshow('Press ESC to close', image)
    capture.release()
    cv2.destroyAllWindows()

else:
    image_list = get_filenames_by_wildcard(os.path.join(command_line_args.image, "*.jpg"))


    def data_gen_load_image(data_x, _):
        """Temp function to load and prepare data in data generator"""
        img, img_norm = yolo_prepare_image(load_image(data_x),
                                           (config['Model.input']['input_img_shape'][0],
                                            config['Model.input']['input_img_shape'][1]))
        return img, img_norm


    data_generator_creator = DataGenerator(image_list, np.zeros_like(image_list), data_gen_load_image,
                                           config['Train.evaluate']['batch_size'])
    data_gen = data_generator_creator.get_data_generator(len(image_list))

    for image_list, image_norm_list in data_gen:
        prediction = model.predict(image_norm_list)
        show_yolo_results_batch(image_list, prediction, 0.1, cat_colors)
