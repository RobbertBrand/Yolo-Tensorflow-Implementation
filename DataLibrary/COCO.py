from pycocotools.coco import COCO
from BasicLib.BasicFunctions import *


def show_coco_data_point(img, label_list):
    """
    Display coco dataset image and labels.
    :param img: loaded images
    :param label_list: labels
    """
    # image = load_image(image)
    for bbox, category in label_list:
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(pt1[0] + bbox[2]), int(pt1[1] + bbox[3]))

        color = (np.random.random((1, 3)) * 255).tolist()[0]
        cv2.rectangle(img, pt1, pt2, color, 2)
    show_image(img)


class CocoDatasetInterface:
    """
    This class forms a easy to use interface, meant to serve the data to a machine learning algorithm.

    EXAMPLE:
    coco_annotation_file = '..\\COCO\\annotations_trainval2017\\annotations\\instances_train2017.json'
    coco_image_folder = '..\\COCO\\annotations_trainval2017\\images\\train2017\\'
    coco = CocoDatasetInterface(coco_annotation_file, coco_image_folder)
    images, cat_dict = coco.get_category_labeled_images([['person'], ['car', 'bicycle', 'dog']])

    coco.print_available_categories()

    show_coco_data_point(images[0][0])
    """

    def __init__(self, coco_ann_file, coco_img_dir):
        """
        Initialize class.

        :param coco_ann_file: file location of the COCO dataset annotation file
        :param coco_img_dir:  file location of the COCO dataset image files
        """
        # self.coco_annotation_file = coco_ann_file
        self.coco_image_folder = coco_img_dir
        self.coco = COCO(coco_ann_file)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.filtered_category_ids = None

    def print_available_categories(self):
        """
        Prints al the COCO dataset categories.

        :return: nothing
        """
        print("ID:  Category:        Super Category:")
        for cat in self.categories:
            print("{:2}   {:15}  {}".format(cat['id'], cat['name'], cat['supercategory']))
        print()

    def get_images_ids(self, cat_nested_list):
        """
        Returns list of image id's of images which meet the given category filter
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
        Returns image file locations
        :param img_spec: coco image specification
        :return: image file locations
        """
        return self.coco_image_folder + img_spec['file_name']

    def get_category_labeled_images(self, cat_nested_list, verbose=True):
        """
        Returns a list with [image_file_locations, ...] a list with labels [[bounding boxes, bbox category], ...] and a
        dictionary linking the category names to their id's. The images contain all the categories specified in the
        'cat_nested_list' parameter.

        :param cat_nested_list: is a list of lists, each inner list describing the items which has to be in the image
        :param verbose: print when True, a description of the selected data.
        :return: a list with image file locations, a list with corresponding labels in format
                    [[[x,y,width,height], bbox_category_id], ...], [[x,y,width,height], bbox_category_id], ...], ...]
                    and a dictionary linking the category names to their id's.

        example:
        get_get_category_labeled_images([['person'], ['car', 'bicycle', 'dog']] ,verbose=False)
        returns images with atleast a person in it AND images with atleast a car AND a bicyle AND a dog.
        labels for eachcategory are added to each image, so a image images with atleast a car AND a bicyle AND a dog
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
            print("Categories selected: ", cat_dict)
            print("Total images:        ", len(img_spec_list))
            print()
            for cat_id, cat_ann_count in zip(range(len(total_ann_count)), total_ann_count):
                print("Annotations in \"{}\": {}".format(cat_dict[cat_id], cat_ann_count))
        return x_data, y_data, cat_dict
