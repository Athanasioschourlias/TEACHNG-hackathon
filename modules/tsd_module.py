from tensorflow import keras

from base.node import TEACHINGNode
from .base_module import LearningModule

import numpy as np
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16

class TSDModule(LearningModule):

    def __init__(self):

        self.model = None
        self._build()
    
    @TEACHINGNode(produce=False, consume=True)
    def __call__(self, input_fn):
        for msg in input_fn:
            np_img = np.asarray(eval(msg.body["img"]), dtype='uint8').reshape(224, 224, 3)
            print("image received!")

            print("HELLOOO")
            # Load the pre-trained VGG16 model
            model = VGG16(weights='imagenet')

            # Preprocess the frame for VGG16 model
            frame = preprocess_input(np_img)

            # Expand dimensions to match VGG16 input shape
            frame = np.expand_dims(frame, axis=0)

            # Perform classification
            preds = model.predict(frame)
            predictions = decode_predictions(preds, top=3)[0]

            print("predictions", predictions)

            # # Display the top predictions
            # for _, label, probability in predictions:
            #     cv2.putText(frame, f"{label}: {probability:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #     # Display the resulting frame
            #     cv2.imshow('Video Classification', frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break


            # img_batch = np.expand_dims(np_img, 0)
            # pred = self.model.predict(preprocess_input(img_batch))
            # print("Predictions: ", decode_predictions(pred))

    def _build(self):
        self.model = keras.applications.MobileNetV2(weights='imagenet', include_top=True)
        self.model.trainable = False
        self.model.summary()
