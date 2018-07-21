"""
module contains tools for data generation.
"""

from BasicLib.BasicFunctions import *
import sklearn.utils


class DataGenerator:
    """
    Data Generator class, creates a data generator to generate a given amound of batches.
    """
    def __init__(self, image_file_list, labels_list, augmentation_func, label_creator_func, batch_size=128):
        """
        Initialize class.

        :param image_file_list: contains a list of image file locations
        :param labels_list: contains the labels for the images in format [[[x,y,width,height], bbox_category_id], ...]
        :param augmentation_func: specifies a function which applies data augmentation. the function input shall be a
                        single data point (one image + labels): augmentation_func(image, labels)
        :param label_creator_func: specifies a function which creates output labels from the labels of a single
                        data point: label_creator_func(resized_labels)
        :param batch_size: sets the output batch size
        """
        # generate error when the amount of labes and images does not match
        assert len(image_file_list) == len(labels_list)

        self.param_image_file_list = image_file_list
        self.param_label_list = labels_list
        self.param_augmentation_func = augmentation_func
        self.param_label_creator_func = label_creator_func
        self.param_batch_size = batch_size
        self.param_n_data_points = len(self.param_image_file_list)
        self.epoch_count = 0
        self.data_point_count = 0

        for i in range(25):
            self.shuffle_data()

    def get_status(self, verbose=False):
        """
        Returns status of generated data in epochs, batches, total batches per epoch. If verbose is True, will the data
        also be printed.
        :param verbose:
        :return:
        """
        epoch_count = self.epoch_count
        batch_count = self.data_point_count / self.param_batch_size
        batches_per_epoch = self.param_n_data_points / self.param_batch_size - 1
        if verbose:
            print("Epoch {:2}, Batch {:4.1f}/{:4.1f}".format(epoch_count, batch_count, batches_per_epoch), end=' ')
        return epoch_count, batch_count, batches_per_epoch

    def shuffle_data(self):
        """
        Shuffle data in generator.
        """
        self.param_image_file_list, self.param_label_list = sklearn.utils.shuffle(self.param_image_file_list,
                                                                                  self.param_label_list)

    def get_data_generator(self, generate_n_batches, verbose=True):
        """
        Returns a data generator that generates a given amount of data batches. The generator depletes when the given
        amount of batches is generated, or when the epoch is finished.
        :param generate_n_batches: specifies the amount of to be generated batches.
        :param verbose: print epoch and batch information if true
        :return: returns two lists, a list of images, a list of labels and a list with status info. Labels were
                created by label_creator_func.
        """
        if self.data_point_count >= self.param_n_data_points:
            self.epoch_count += 1
            self.data_point_count = 0
            self.shuffle_data()

        # create batches
        start_point = self.data_point_count
        stop_point = start_point + (generate_n_batches * self.param_batch_size)
        stop_point = np.minimum(stop_point, self.param_n_data_points)
        for batch_start_pos in range(start_point, stop_point, self.param_batch_size):
            self.get_status(verbose)

            batch_stop_pos = batch_start_pos + self.param_batch_size

            # extract batch
            image_file_batch = self.param_image_file_list[batch_start_pos: batch_stop_pos]
            label_batch = self.param_label_list[batch_start_pos: batch_stop_pos]

            output_label_batch = []
            image_batch = []
            # create datapoints in batch
            for labels, image_file in zip(label_batch, image_file_batch):
                # load image
                image = load_image(image_file, normalize=True)

                # augment data
                resized_image, resized_labels = self.param_augmentation_func(image, labels)

                # store loaded data
                output_label_batch.append(self.param_label_creator_func(resized_labels))
                image_batch.append(resized_image)

                self.data_point_count += 1

            yield image_batch, np.array(output_label_batch)
