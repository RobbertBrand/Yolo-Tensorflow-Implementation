"""
COCO provides a simple way to use the coco data set thru a standardized interface. Implementing this
module can reduce complexity in the code for gathering and preparing "Coco data set" data. Besides that does the module
provide a standardized and simple interface which could be used with any data set containing image file locations and
bboxes.

#########
# USAGE #
#########

# set category filters filters
param_coco_cat_filters = [['person'], ['car'], ['bus'], ['truck']]

# set coco dataset locations
param_coco_annotation_file = '..\\COCO\\annotations_trainval2017\\annotations\\instances_train2017.json'
param_coco_img_dir = '..\\COCO\\annotations_trainval2017\\images\\train2017\\'

# load data set
coco = COCO.CocoDatasetInterface(param_coco_annotation_file, param_coco_img_dir)
data_x, data_y, data_dict_cat = coco.get_category_labeled_images(param_coco_cat_filters)


########################
# STANDARD DATA FORMAT #
########################

data_x is a list of image file locations [image_file_locations, ...]
data_y is a list with labels [[[bbox1_img1, bbox1_category_img1], [bbox2_img1, bbox2_category_img1], ...],
                              [[bbox1_img2, bbox1_category_img2], [bbox2_img2, bbox2_category_img2], ...],
                              ...]
The bboxN_imgN variables specify the actual bboxes in format [x,y,width,height] where x and y are the left top corner
position of the bbox.

"""

from pycocotools.coco import COCO
from BasicLib.BasicFunctions import *


def show_coco_data_point(img, label_list, load_image_from_file=False):
    """
    Display coco data set image and labels.
    :param img: loaded image of image file location
    :param label_list: labels
    :param load_image_from_file: interprets 'img' as file location when True.
    """
    image = img
    if load_image_from_file:
        image = load_image(img)

    for bbox, category in label_list:
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(pt1[0] + bbox[2]), int(pt1[1] + bbox[3]))

        color = (np.random.random((1, 3)) * 255).tolist()[0]
        cv2.rectangle(image, pt1, pt2, color, 2)
    show_image(image)


class CocoDatasetInterface:
    """
    This class forms a easy to use interface, meant to serve the data to a machine learning algorithm. Implementing this
    class can reduce complexity in the code for gathering and preparing data. Besides that does the class provide a
    standardized and simple interface which could be used with any data set containing image file locations and bboxes.

    EXAMPLE:
    from DataLibrary.COCO import *

    coco_annotation_file = '..\\COCO\\annotations_trainval2017\\annotations\\instances_train2017.json'
    coco_image_folder = '..\\COCO\\annotations_trainval2017\\images\\train2017\\'
    coco = CocoDatasetInterface(coco_annotation_file, coco_image_folder)
    images, labels, cat_dict = coco.get_category_labeled_images([['person'], ['car', 'bicycle', 'dog']])

    coco.print_available_categories()

    show_coco_data_point(images[0], labels[0], True)

    """

    def __init__(self, coco_ann_file, coco_img_dir):
        """
        Initialize class.
        :param coco_ann_file: file location of the COCO data set annotation file
        :param coco_img_dir:  file location of the COCO data set image files
        """
        # self.coco_annotation_file = coco_ann_file
        self.coco_image_folder = coco_img_dir
        self.coco = COCO(coco_ann_file)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.filtered_category_ids = None

    def print_available_categories(self):
        """Prints all the Coco data set categories."""
        print("ID:  Category:        Super Category:")
        for cat in self.categories:
            print("{:2}   {:15}  {}".format(cat['id'], cat['name'], cat['supercategory']))
        print()

    def get_images_ids(self, cat_nested_list):
        """
        Returns list of image id's of images which meet the given category filter. These id's can be used to load
        the image specifications.
        :param cat_nested_list: is a list of lists, each inner list describing the items which has to be in the image
                                    in the following format: [['car'], ['cat', 'horse']]
        :return: list of image specifications, list of category id's
        """
        img_id_list = []
        total_cat_list = []
        for cat_list in cat_nested_list:
            cat_id_list = self.coco.getCatIds(catNms=cat_list)
            total_cat_list += cat_id_list
            img_id_list += self.coco.getImgIds(catIds=cat_id_list)
        img_spec_list = self.coco.loadImgs(set(img_id_list))
        total_cat_list = list(set(total_cat_list))
        return img_spec_list, total_cat_list

    def build_category_dict(self, cat_list):
        """
        Creates two dictionaries linking the coco category id's to the normalized id's and the category names to their
        normalized id's. These Dictionaries can be used to make id normalization and id to name linking easy.
        Returns two dictionaries.:
            cat_dict[0 .. n_categories] => cat_name
            cat_translate_dict[coco_cat_id] => normalized_cat
        :param cat_list: list of coco category id's
        :return: cat_dict, cat_translate_dict
        """
        cat_spec_list = self.coco.loadCats(cat_list)
        cat_dict = {}
        cat_translate_dict = {}
        for cat_spec, normalized_id in zip(cat_spec_list, range(len(cat_spec_list))):
            cat_dict[normalized_id] = cat_spec['name']
            cat_translate_dict[cat_spec['id']] = normalized_id
        return cat_dict, cat_translate_dict

    def load_image_annotations(self, img_spec, cat_translate_dict, cat_list):
        """
        Returns annotations list bboxes in format [[x,y,width,height], bbox_category_id], ...] for the given image_spec,
        if bbox category is in cat_list.
        :param img_spec: coco image specification
        :param cat_translate_dict: cat_translate_dict[coco_cat_id] => normalized_cat
        :param cat_list: list of coco category id's
        :return: list bboxes in format [[x,y,width,height], bbox_category_id], ...]
        """
        img_bboxes = []
        ann_count_per_cat = [0] * len(cat_list)
        ann_spec_list = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_spec['id']))

        for ann_spec in ann_spec_list:
            if ann_spec['category_id'] in cat_list and ann_spec['iscrowd'] == 0:
                img_bboxes.append([ann_spec['bbox'], cat_translate_dict[ann_spec['category_id']]])
                ann_count_per_cat[cat_translate_dict[ann_spec['category_id']]] += 1
        return img_bboxes, ann_count_per_cat

    def get_image_file_location(self, img_spec):
        """
        Returns image file location
        :param img_spec: coco image specification
        :return: image file location
        """
        return self.coco_image_folder + img_spec['file_name']

    def get_category_labeled_images(self, cat_nested_list, verbose=True, print_func=print):
        """
        This function forms the actual interface and output of the class, providing the coco data via a standardized and
        simple format.

        Returns a list with [image_file_locations, ...] a list with labels [[bounding boxes, bbox category], ...] and a
        dictionary linking the category names to their id's. The images contain all the categories specified in the
        'cat_nested_list' parameter.

        :param cat_nested_list: is a list of lists, each inner list describing the items which has to be in the image.
        :param verbose: print when True, a description of the selected data.
        :param print_func: contains a function to print 'verbose' information with. Is the print function by default.
        :return: a list with image file locations, a list with corresponding labels in format
                    [[[x,y,width,height], bbox_category_id], ...], [[x,y,width,height], bbox_category_id], ...], ...]
                    and a dictionary linking the category names to their id's.

        example:
        get_category_labeled_images([['person'], ['car', 'bicycle', 'dog']] ,verbose=False)
        returns images with at least a person in it AND images with at least a car AND a bicycle AND a dog.
        labels for each category are added to each image, so a image images with at least a car AND a bicycle AND a dog
        might also contain labels of persons.
        """
        img_spec_list, cat_list = self.get_images_ids(cat_nested_list)
        cat_dict, cat_translate_dict = self.build_category_dict(cat_list)

        # load images and annotations
        x_data = []
        y_data = []
        total_ann_count = np.array([0] * len(cat_list))
        for img_spec in img_spec_list:
            image_file = self.get_image_file_location(img_spec)
            image_bboxes, img_ann_count = self.load_image_annotations(img_spec, cat_translate_dict, cat_list)
            total_ann_count += img_ann_count

            x_data.append(image_file)
            y_data.append(image_bboxes)

        # display data details
        if verbose:
            print_func("Categories selected: {}".format(cat_dict))
            print_func("Total images:        {}".format(len(img_spec_list)))
            for cat_id, cat_ann_count in zip(range(len(total_ann_count)), total_ann_count):
                print_func("Annotations in \"{}\": {}".format(cat_dict[cat_id], cat_ann_count))
        return x_data, y_data, cat_dict

    def get_image_sizes(self, cat_nested_list):
        """
        Returns a list of image sizes in pixels. If the same value for the 'cat_nested_list' parameter is used as with
        the 'get_category_labeled_images' method, will the returned sizes match the data_x and data_y result lists of
        the get_category_labeled_images method. So:
            img_size_list[i] belongs to data_x[i] and data_y[i]
        :param cat_nested_list: is a list of lists, each inner list describing the items which has to be in the image.
        :return: list of image sizes in format [[width, height], ...]
        """
        img_size_list = []
        img_spec_list, cat_list = self.get_images_ids(cat_nested_list)
        for img_spec in img_spec_list:
            img_size_list.append([img_spec['width'], img_spec['height']])
        return img_size_list
