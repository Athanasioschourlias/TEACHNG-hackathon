from tensorflow import keras

from base.node import TEACHINGNode
from .base_module import LearningModule

import numpy as np
import darknet
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16

class TSDModule(LearningModule):

    def __init__(self):

        self.model = None
        self._build()
    
    @TEACHINGNode(produce=False, consume=True)
    def __call__(self, input_fn):
        for msg in input_fn:
            frame = np.asarray(eval(msg.body["img"]), dtype='uint8').reshape(224, 224, 3)
            print("image received!")

            print("HELLOOO")

            # Load the YOLOv4-Tiny model configuration and weights
            config_file = 'cfg/yolov4-tiny.cfg'
            weight_file = 'weights/yolov4-tiny.weights'

            # Load the COCO class names
            class_names = []
            with open('data/coco.names', 'r') as f:
                class_names = [line.strip() for line in f.readlines()]

            # Load the YOLOv4-Tiny model
            network, _ = darknet.load_network(config_file, weight_file, batch_size=1)

            # Convert the frame to a darknet image
            darknet_image = darknet.make_image(frame.shape[1], frame.shape[0], 3)

            darknet.copy_image_from_bytes(darknet_image, frame.tobytes())

            # Perform object detection
            detections = darknet.detect_image(network, class_names, darknet_image)

            # Draw bounding boxes and labels on the frame
            darknet.print_detections(detections)
            darknet.draw_boxes(detections, frame)



            # Display the resulting frame
            # frame_resized = cv2.resize(frame_rgb, (frame.shape[1], frame.shape[0]))
            # cv2.imshow('Object Detection', frame_resized)

    def _build(self):
        self.model = keras.applications.MobileNetV2(weights='imagenet', include_top=True)
        self.model.trainable = False
        self.model.summary()
