from resnet50 import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


import numpy as np
import argparse
import cv2
import os
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--target", required = True, help = "Target directory")
args = vars(ap.parse_args())


inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input


print("Loading ResNet50...")
model = ResNet50(weights='imagenet', include_top=False)


features = {}
c = 0
for images in os.listdir(args["target"]):
    filename = args["target"] + '/' + images
    image = load_img(filename, target_size = inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0) # image.shape = (1, 224, 224, 3)
    image = preprocess(image) # image.shape = (1, 224, 224, 3)
    pred = model.predict(image) # pred.shape = (1, 1, 1, 2048)
    image_id = images.split('.')[0]
    features[image_id] = pred
    print('>{}, count = {}'.format(images, c))
    c += 1


pickle.dump(features, open('features.pkl', 'wb'))
