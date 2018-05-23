from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
# from keras.utils import to_categorical
# from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import ModelCheckpoint
import numpy as np


from pickle import load


def train_test_data(filename):
    DataFile = open(filename, 'r')
    Data = DataFile.read()
    DataFile.close()

    ImageID = []

    textDataFile = load(open('descriptions.pkl', 'rb'))

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


def extractPhotoFeatures(filename):
    photoFeatureFile = load(open('features.pkl', 'rb'))
    photoData = {}
    ImageID = []

    DataFile = open(filename, 'r')
    Data = DataFile.read()
    DataFile.close()

    for line in Data.split('\n'):
        if len(line) < 1:
            continue
        ImageID.append(line.split('.')[0])

    for key in photoFeatureFile:
        if key in ImageID:
            photoData[key] = photoFeatureFile[key]

    return photoData


# trainData = train_test_data('../Flickr_8k.trainImages.txt')
# testData = train_test_data('../Flickr_8k.testImages.txt')
# trainPhoto = extractPhotoFeatures('../Flickr_8k.trainImages.txt')
# testPhoto = extractPhotoFeatures('../Flickr_8k.testImages.txt')

# print(testPhoto)
# print("Training Data size :- {}".format(len(trainData)))
# print("Test Data size :- {}".format(len(testPhoto)))


# print(photoData['2513260012_03d33305cf'][0])
# shape of each key of photoData = (1, 1, 1, 2048)


def extractDescriptionsInListOfStrings(dictionaryOfDescriptions):
    desc = []
    for key in dictionaryOfDescriptions:
        for l in dictionaryOfDescriptions[key]:
            s = "".join(l)
            desc.append(s)
    return desc


# print(trainData)
# desc = extractDescriptionsInListOfStrings(trainData)
# print(desc)


def createTokenizer(Data):
    desc = extractDescriptionsInListOfStrings(Data)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc)
    return tokenizer


def maxLength(textData):
	lines = extractDescriptionsInListOfStrings(textData)
	return max(len(d.split()) for d in lines)


def createData(tokenizer, photo, descriptions, maxLength):
    X1, X2, y = [], [], []
    vocab_size = len(tokenizer.word_index) + 1
    for description in descriptions:
        encoder = tokenizer.texts_to_sequences([description])[0]
        for i in range(1, len(encoder)):
            inSeq = encoder[:i]
            outSeq = encoder[i]
            inSeq = pad_sequences([inSeq], maxlen = maxLength, padding = 'pre')[0]
            outSeq = to_categorical(outSeq, num_classes = vocab_size)
            X1.append(photo)
            X2.append(inSeq)
            y.append(outSeq)
    return np.array(X1), np.array(X2), np.array(y)


def createModel(maxLength, vocab_size):
    in1 = Input(shape = (2048, ))
    layer1 = Dropout(0.5)(in1)
    layer2 = Dense(256, activation = 'relu')(layer1)

    in2 = Input(shape = (maxLength,))
    layer3 = Embedding(vocab_size, 256, mask_zero = True)(in2)
    layer4 = Dropout(0.5)(layer3)
    layer5 = LSTM(256)(layer4)

    decoder1 = add([layer2, layer5])
    decoder2 = Dense(256, activation = "relu")(decoder1)
    output = Dense(vocab_size, activation = "softmax")(decoder2)

    model = Model(inputs = [in1, in2], outputs = output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print(model.summary())
    # plot_model(model, to_file = 'model.jpg', show_shapes = True)
    return model


def DataGenerator(tokenizer, photoData, textData, maxLength):
    while True:
        for key, descriptions in textData.items():
            photo = photoData[key][0][0][0]
            input1, input2, out = createData(tokenizer, photo, descriptions, maxLength)
            yield([input1, input2], out)




trainData = train_test_data('../Flickr8k/Flickr_8k.trainImages.txt')
MaxLength = maxLength(trainData)
trainPhoto = extractPhotoFeatures('../Flickr8k/Flickr_8k.trainImages.txt')
# print(MaxLength)
tokenizerTrain = createTokenizer(trainData)
vocab_size = len(tokenizerTrain.word_index) + 1
print("Training descriptions size:- {}".format(len(trainData)))
print("Training photos size:- {}".format(len(trainPhoto)))
print("Vocabulary size:- {}".format(vocab_size))
print("Max Length:- {}".format(MaxLength))
# X1train, X2train, ytrain = createData(tokenizerTrain, trainPhoto, trainData, MaxLength)


# testData = train_test_data('../Flickr_8k.devImages.txt')
# testPhoto = extractPhotoFeatures('../Flickr_8k.devImages.txt')
# X1test, X2test, ytest = createData(tokenizerTrain, testPhoto, testData, MaxLength)


# generator = DataGenerator(tokenizerTrain, trainPhoto, trainData, MaxLength)
# ins, outs = next(generator)
# print(ins[0].shape)
# print(ins[1].shape)
# print(outs.shape)


model = createModel(MaxLength, vocab_size)
epochs = 20
steps = len(trainData)
for i in range(epochs):
    generator = DataGenerator(tokenizerTrain, trainPhoto, trainData, MaxLength)
    model.fit_generator(generator, epochs = 1, steps_per_epoch = steps)
    model.save('Model_' + str(i) + '.h5')
# filepath = "model.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')
# model.fit([X1train, X2train], ytrain, epochs=20, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
