""" Simple script to run a pretrained SSD hand detector on some OR vide frames
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

parser = argparse.ArgumentParser(
    description="Sample inference test from test images ")
parser.add_argument("-tdir",
                    "--model_dir",
                    help="Path to the folder where the model is stored.",
                    default='./hand_inference_graph',
                    type=str)
parser.add_argument("-vdir",
                    "--vid_dir",
                    help="Path to the video file.",
                    default='./images/vids/2021-11-18-T08-50-01-axis2.mp4',
                    type=str)
parser.add_argument('-ds',
                    '--display',
                    type=int,
                    default=1,
                    help='Display the detected images using OpenCV')
parser.add_argument('-w',
                    '--write',
                    type=int,
                    default=0,
                    help='Write the processed images to the result directory')
args = parser.parse_args()
 
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

    MODEL_NAME = args.model_dir
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/saved_model'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')
    PATH_TO_VID = args.vid_dir

    print('Loading the detection model...', end='')
    start_time = time.time()
    detection_model = tf.saved_model.load(PATH_TO_CKPT)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    cap = cv2.VideoCapture(PATH_TO_VID)
    idx = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        img_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_np_resized = cv2.resize(img_np, (600, 600))
        output_dict = run_inference_for_single_image(detection_model, img_np_resized)
        viz_utils.visualize_boxes_and_labels_on_image_array( 
            img_np_resized,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            min_score_thresh=_score_thresh,
            line_thickness=3)
        image_np = cv2.cvtColor(img_np_resized, cv2.COLOR_BGR2RGB)    
        num_box_images = len([i for i in output_dict['detection_scores'] if i > _score_thresh])

        if (args.display > 0):
            cv2.imshow('Detection on frames', image_np)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        print('{} hands detected'.format(num_box_images))

        if(args.write > 0 ):
            cv2.imwrite('./images/result/image_{}.jpg'.format(idx), image_np)
        idx +=1
    cv2.destroyAllWindows()