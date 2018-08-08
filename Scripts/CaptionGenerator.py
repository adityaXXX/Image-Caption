from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from resnet50 import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model

import argparse
import numpy as np
import os
import pickle

from tkinter import *
from PIL import ImageTk, Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Target Image")
args = vars(ap.parse_args())


inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

print("Loading ResNet...")
resnet = ResNet50(include_top=False, weights='imagenet')

print("Extracting features.... ")
features = {}
# for images in os.listdir(args["image"]):
filename = args["image"]
image = load_img(filename, target_size = inputShape)
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)
image = preprocess(image)
pred = resnet.predict(image)
image_id = filename.split('.')[0]
features[image_id] = pred


def train_test_data(filename):
    DataFile = open(filename, 'r')
    Data = DataFile.read()
    DataFile.close()

    ImageID = []

    textDataFile = pickle.load(open('descriptions.pkl', 'rb'))

    for line in Data.split('\n'):
        if len(line) < 1:
            continue
        ImageID.append(line.split('.')[0])

    Data = {}

    for key in textDataFile:
        if key in ImageID:
            Data[key] = textDataFile[key]

    for ID in Data:
        for i in range(len(Data[ID])):
            l = Data[ID][i]
            l = "START " + " ".join(l) + " END"
            Data[ID][i] = l

    return Data


def extractDescriptionsInListOfStrings(dictionaryOfDescriptions):
    desc = []
    for key in dictionaryOfDescriptions:
        for l in dictionaryOfDescriptions[key]:
            s = "".join(l)
            desc.append(s)
    # print(desc)
    return desc


def createTokenizer(Data):
    desc = extractDescriptionsInListOfStrings(Data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc)
    return tokenizer


trainData = train_test_data('../Flickr8k/Flickr_8k.trainImages.txt')
# print(trainData)
tokenizerTrain = createTokenizer(trainData)
# print(tokenizerTrain.word_index)
# k = tokenizerTrain.word_index
# pickle.dump(k, open('word_index.pkl', 'wb'))


def word_for_id(integer, tokenizer):
    k = tokenizer.word_index
    # print(integer)
    for word in k.keys():
        if k[word] == integer:
            return word
    return None


def createCaptions(tokenizer, photoData, MaxLength, model):
    for key, feature in photoData.items():
        inSeq = "start"
        for i in range(MaxLength):
            sequence = tokenizer.texts_to_sequences([inSeq])[0]
            sequence = pad_sequences([sequence], maxlen = MaxLength)
            ID = model.predict([feature[0][0], sequence])
            # print("ID:- {}".format(ID))
            # print(ID.shape)
            ID = np.argmax(ID)
            # print("Argmax:- {}".format(ID))
            ID = word_for_id(ID, tokenizer)
            if ID is None:
                break
            inSeq += " " + ID
            if ID == "end":
                break
        return inSeq

print("Loading Model...")
model = load_model('../TrainedModels/Model_5.h5')
caption = createCaptions(tokenizerTrain, features, 36, model)
caption = caption[6:len(caption)-4]
print("Done!!")
# print(features['DSC05733'][0][0][0])


root = Tk()
root.title("Caption Generator")
p = Image.open(args["image"])
p = p.resize((250, 250), Image.ANTIALIAS)
text1 = Text(root, height=20, width=30)
photo=ImageTk.PhotoImage(p)
text1.insert(END,'\n')
text1.image_create(END, image=photo)

text1.pack(side=TOP)

text2 = Text(root, height=20, width=100)
text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
text2.tag_configure('big', font=('Verdana', 20, 'bold'))
text2.tag_configure('color', foreground='#476042',
						font=('Tempus Sans ITC', 12, 'bold'))
text2.insert(END,'\nCaption\n', 'big')
quote = caption
text2.insert(END, quote, 'color')
text2.pack(side=BOTTOM)

root.mainloop()
