# coding: utf-8
# credits: https://www.pyimagesearch.com
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from cropim import CropIm
import numpy as np
import imutils
import pickle
import cv2
import os

args = {
    "model1": '..\\model\\3c98a.model',
    "model2": '..\\model\\5c85a.model',
    "model3": '..\\model\\5c33a200e.model',
    "model4": '..\\model\\6c79a200e.model',
    "lb1": '..\\model\\3c98a.pickle',
    "lb2": '..\\model\\5c85a.pickle',
    "lb3": '..\\model\\5c33a200e.pickle',
    "lb4": '..\\model\\6c79a200e.pickle'
}

# load the image
path = '..\\testim\\acer_platanoides__03.jpg'
image = CropIm(path)
output = cv2.imread(path,3)
 
# pre-process the image for classification
image = cv2.resize(image, (120, 120))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# load the trained convolutional neural network and the label
# binarizer
print("[info] loading network...")
model = load_model(args["model4"])
lb = pickle.loads(open(args["lb4"], "rb").read())
 
# classify the input image
print("[info] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

filename = path[path.rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"
 
# build the label and draw the label on the image
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)

# show the output image
print("[INFO] {}".format(label))
plt.axis("off")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(label)
plt.show()
plt.savefig("..\\results\\classify\\ap379a.png")