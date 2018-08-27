"""
Module contains tools for data generation. This library is designed to provide a data generator, which processes data
and returns it in batches.

#########
# USAGE #
#########

def image_loader(img_file_loc, image_label)
    img = load_image(img_file_loc)
    return img, image_label

x_data = [list of image file locations]
y_data = [list of labels for images]
data_batch_size = 32        # specifies number of data points (images and labels) in a single batch.
amount_of_batches = 200     # amount of batches to be generated before depletion. The generator also depletes when al
                            # available data form x_data and y_data was generated in batches.

data_generator_creator = DataGenerator(x_data,
                                       y_data,
                                       data_processing_func=image_loader,
                                       batch_size=32)

data_generator = data_generator_creator.get_data_generator(amount_of_batches_to_be_generates)


x_data_batch, y_data_batch = next(data_generator)

"""

from BasicLib.BasicFunctions import *
import sklearn.utils


class DataGenerator:
    """
    DataGenerator object, creates a data generator to generate a given amount of batches. A data processor function has
    to be defined during the DataGenerator object declaration to process the source data with, while generating a batch.
    Because the data is processed on the fly, every batch again, for every data point, is there no need to store the
    processed data in memory after the batch was used. This can be very useful when for example, the data will expand
    after processing, like when the data processor function loads images in to memory.

    The data generator created with the DataGenerator object, will be generating a specified amount of data batches
    after which it will deplete. The data generator will also deplete after generating all data from the source
    (data_x_list and data_y_list).

    When creating a new data generator with the DataGenerator object after one depletes, will the new data generator
    continue generating data where the previous generator stopped. If the previous data generator(s), generated all
    available data, shall the source data be shuffled and does the new data generator start again, generating the
    available data.
    """
    def __init__(self, data_x_list, data_y_list, data_processing_func, batch_size=128):
        """
        Initialize class.

        :param data_x_list: contains a list of data points
        :param data_y_list: contains a list of data points (labels)
        :param data_processing_func: specifies a function. The data generator will feed the function a single data point
                                        of data_x and data_y like (data_x_list[0], data_y_list[0]). The function should
                                        return the processed data_x and data_y data points. The data_x and data_y
                                        data point, processed by the function are than added to the output batch of the
                                        generator. EXAMPLE of data_processing_func use:

                                            def data_processing_func(data_x, data_y):
                                                Return data_x + 1, data_y - 1

        :param batch_size: sets the output batch size (amount of data points)
        """
        # generate an error when the amount of data_x and data_y does not match because each data_x_list[x] and
        # data_y_list[x], should be a matching set.
        assert len(data_x_list) == len(data_y_list)

        self.param_data_x_list = data_x_list
        self.param_data_y_list = data_y_list
        self.param_data_processing_func = data_processing_func
        self.param_batch_size = batch_size
        self.param_n_data_points = len(self.param_data_x_list)
        self.epoch_count = 0
        self.data_point_count = 0

        for i in range(25):
            self.shuffle_data()

    def get_status(self, verbose=False):
        """
        Returns status of generated data in epochs, batches, total batches per epoch. If verbose is True, will the data
        also be printed.
        :param verbose: prints status when true
        """
        epoch_count = self.epoch_count
        batch_count = self.data_point_count / self.param_batch_size
        batches_per_epoch = self.param_n_data_points / self.param_batch_size - 1
        if verbose:
            print("Epoch {:2}, Batch {:4.1f}/{:4.1f}".format(epoch_count, batch_count, batches_per_epoch), end=' ')
        return epoch_count, batch_count, batches_per_epoch

    def shuffle_data(self):
        """Shuffle data in generator."""
        self.param_data_x_list, self.param_data_y_list = sklearn.utils.shuffle(self.param_data_x_list,
                                                                               self.param_data_y_list)

    def get_data_generator(self, generate_n_batches, verbose=True):
        """
        Returns a data generator that generates a given amount of data batches. The generator depletes when the given
        amount of batches is generated, or when the epoch is finished.
        :param generate_n_batches: specifies the amount of to be generated batches.
        :param verbose: print epoch and batch information if true
        :return: returns two lists, a list of data_x data points, a list of data_y data points and a list with status
                info.
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
            # print status to inform user about data generation status iver verbose is True
            self.get_status(verbose)

            batch_stop_pos = batch_start_pos + self.param_batch_size

            # extract batch of data from the source
            data_x_batch = self.param_data_x_list[batch_start_pos: batch_stop_pos]
            data_y_batch = self.param_data_y_list[batch_start_pos: batch_stop_pos]

            output_data_y_batch = []
            output_data_x_batch = []
            # process the data points from the selected batch and add them to a output batch.
            for data_y, data_x in zip(data_y_batch, data_x_batch):
                processed_data_x, processed_data_y = self.param_data_processing_func(data_x, data_y)

                output_data_y_batch.append(processed_data_y)
                output_data_x_batch.append(processed_data_x)

                self.data_point_count += 1

            yield output_data_x_batch, np.array(output_data_y_batch)
