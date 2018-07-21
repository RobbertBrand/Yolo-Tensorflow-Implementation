# Yolo-Tensorflow-Implementation
A Yolo object detection implementation in Tensorflow, trainable using Tensorflow optimizers like ADAM. 

# Description 
This Python project is a, from scratch, implementation of a Yolo object detection neural network model in Tensorflow. This implementation consists out of functioning Yolo model, trainable using the Tensorflow ADAM optimizer on data like the Microsoft COCO dataset. 

# Installation 
Install the following Python packages in your environment: 
Matplotlib v2.2.2 
Numpy v1.14.4 
Opencv v3.3.1 
Pandas v0.22.0 
Pycocotools v2.0 
Scikit-image v0.13.1 
Scikit-learn v0.19.1 
Scipy v1.0.1 
Tensorboard v1.8.0 
Tensorflow (-gpu) v1.8.0 

Download the coco dataset files from http://cocodataset.org/#download 

Clone this project and changes the file locations of the COCO dataset in the YOLO.py file:
param_coco_annotation_file = '..\\COCO\\annotations_trainval2017\\annotations\\instances_train2017.json'
param_coco_img_dir = '..\\COCO\\annotations_trainval2017\\images\\train2017\\'

You are ready to roll! 

# Usage 
Run the YOLO.py file to start training your YOLO model. Train and test results will be displayed while training. The model is stored every epoch. 

A stored model can be loaded and tested by filling in the stored model file location in the “param_saved_model_file” variable in the YoloLib.py module. Now the the loaded model will be used to predict objects in one batch of images. Each image in this batch will be shown with the predicted objects marked. 

These are prediction results of a yolo model trained with this project for 12 epochs:
![alt text](https://github.com/RobbertBrand/Yolo-Tensorflow-Implementation/blob/master/Figure_1-1.png "")
![alt text](https://github.com/RobbertBrand/Yolo-Tensorflow-Implementation/blob/master/Figure_1-2.png "")

# performance
This Python project was used to train the Yolo model defined in YoloLib.py, on an Intel i7 set-up with a Nvidia Geforce 750M graphic card. The model was trained on about 65000 Coco data set images for between 10 and 20 epochs. Results are acceptable but far from perfect because of two reasons. A larger model with more yolo output cells couldn’t be trained on my set-up because of the limited amount of available working memory. The second reason is that filtering of the predictions has to be further implemented like a non-max suppression algorithm.

To boost model precision and recall in a relative simple way, should the following things be done: 
In the YoloLib.py module under the function “create_yolo_model” should the model be sized up by adding layers, increasing filter depths and increasing the size of the fully connected layer.

In YOLO.py should: 
The input image size be increased in parameter param_input_img_shape. (The project will resize the images automatically) 

The Yolo output cell grid size be increase in parameter param_yolo_outp_cell_grid_size. The set size should match the input image size by making sure the image size is dividable by the output cell grid size. 

The anchor box specification (the to be predicted object widths and heights) should match the object sizes of the data set as close as possible. This is done in parameter param_yolo_anchors_definition. For example; if the model should predict cars, does it make more sense to make wide and less high anchors than high and less wide ones. 

At last should the learning rate and learning rate decay in parameter param_learning_rate and param_learning_rate_dacay_factor be fine-tuned. Note that a to large learning rate can result in NaN loss values and corrupt the model. 
