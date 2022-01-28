""" Simple script to run a pretrained SSD hand detector on some test images
"""

import os
import io
import argparse
import time
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2 

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from utils.inference_utils import run_inference_for_single_image

# Initiate argument parser
# parser = argparse.ArgumentParser(
#     description="Sample inference test from test images ")
# parser.add_argument("-tdir",
#                     "--model_dir",
#                     help="Path to the folder where the model is stored.",
#                     default='./inference_model_detection/inference_graph_1',
#                     type=str)
# parser.add_argument("-l",
#                     "--label_dir",
#                     help="Path to the label map file.",
#                     default='./inference_model_detection/labelmap.pbtxt',
#                     type=str)
# args = parser.parse_args()
 
if __name__ == '__main__':

    # Enable GPU dynamic memory allocation
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
            print(e)

    # score threshold for showing bounding boxes.
    _score_thresh = 0.2

    MODEL_NAME = './hand_inference_graph'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/saved_model'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

    # PATH_TO_TEST = './images/test_dataset_EGO/CARDS_COURTYARD_H_S'
    PATH_TO_TEST = './images/test_dataset_OHD/test_data/images'
    NUM_TEST_IMAGES = 10

    print('Loading the detection model...', end='')
    start_time = time.time()
    detection_model = tf.saved_model.load(PATH_TO_CKPT)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    images_list = glob.glob(PATH_TO_TEST + '/*.jpg')

    for idx in range(NUM_TEST_IMAGES):
        img = cv2.imread(images_list[idx])
        img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_dict = run_inference_for_single_image(detection_model, img_np)
        viz_utils.visualize_boxes_and_labels_on_image_array( 
            img_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            min_score_thresh=_score_thresh,
            line_thickness=3)

        image_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)    
        # cv2.imshow('image_{}'.format(idx),image_np)
        num_box_images = len([i for i in output_dict['detection_scores'] if i > _score_thresh])
        print('{} hands detected'.format(num_box_images))
        cv2.imwrite('./images/image_{}.jpg'.format(idx), image_np)

    # input("Enter to close windows ")
    # cv2.destroyAllWindows()