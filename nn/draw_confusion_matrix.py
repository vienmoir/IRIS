# coding: utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flonet import FloNet
from cropim import CropIm


from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
#import argparse
import random
import pickle
import cv2
import os

# confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
#


print('[info] imported everything, yay')

args = {
    "dataset": "..\\dataset",
    "confm_train": "..\\confm_train.png",
    "confm_test": "..\\confm_test.png",
    "confm_all": "..\\confm_all.png",
    "confm_normal": "..\\confm_normal.png",
    "model": "..\\florec.model"
}



# the fun part

IMAGE_DIMS = (120, 120, 3)

#initialize data labels
data = []
labels = []

print("[info] loading paths...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

print('[info] loading images...')
for imagePath in imagePaths:
   # image = cv2.imread(imagePath, 3) 
    image = CropIm(imagePath)
   # print(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
    
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
print('[info] images collected')

data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)
print("[info] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.1, 
                                                  random_state = 42)

print("[info] load model...")
model = load_model(args["model"])

def plot_confusion_matrix(cm, classes, save_path, normalize = False, title = "Confusion matrix",
cmap = plt.get_cmap('Purples')):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting mormile = true
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]).round(2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix without normalization')

    print(cm)

    tresh =  cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                horizontalalignment = "center",
                color = "white" if cm[i, j] > tresh else "black")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path)


cm_plot_labels = lb.classes_

predY = model.predict(testX, verbose = 2)
cm = confusion_matrix(testY.argmax(axis=1), predY.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, save_path = args["confm_test"], title="Confusion Matrix Test")

predY = model.predict(trainX, verbose = 2)
cm = confusion_matrix(trainY.argmax(axis=1), predY.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, save_path = args["confm_train"], title="Confusion Matrix Train")


predY = model.predict(np.vstack((trainX, testX)), verbose = 2)
cm = confusion_matrix(np.vstack((trainY, testY)).argmax(axis=1), predY.argmax(axis=1))
plot_confusion_matrix(cm, cm_plot_labels, save_path = args["confm_all"], title="Confusion Matrix All")


