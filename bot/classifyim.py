# coding: utf-8
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from cropim import CropIm
import numpy as np
import imutils
import pickle
import cv2
import os

def Classify(image):
	print("[info] loading network...")
	model = load_model('..\\model\\10c93a200e.model')
	lb = pickle.loads(open('..\\model\\10c93a200e.pickle', "rb").read())
	image = cv2.resize(image, (120, 120))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	predict = model.predict(image)
	proba = predict[0]
	print(proba)
	idx = np.argmax(proba)
	print(idx)
	print("[info] classified")
	if proba[idx] < 0.5:
		print("none")
		first = second = third = "none"
	else:
	    ruix = np.partition(proba, -3)[-3:]
	    print(ruix)
	    print("ok")
	    if ruix[0] > ruix[1]:
	    	print("1")
	    	if ruix[0] > ruix[2]:
		    	print("2")
		    	first = ruix[0]
		    	if ruix[1] > ruix[2]:
		    		print("3")
		    		second = ruix[1]
		    		third = ruix[2]
		    	else:
		    		print("4")
		    		second = ruix[2]
		    		third = ruix[1]
	    	else:
		    	print("5")
		    	first = ruix[2]
		    	second = ruix[0]
		    	third = ruix[1]
	    elif ruix[0] > ruix[2]:
	    	print("6")
	    	first = ruix[1]
	    	second = ruix[0]
	    	third = ruix[2]
	    elif ruix[1] > ruix[2]:
	    	print("7")
	    	first = ruix[1]
	    	second = ruix[2]
	    	third = ruix[0]
	    else: 
	    	print("8")
	    	first = ruix[2]
	    	second = ruix[1]
	    	third = ruix[0]
	    prob = proba[idx] * 100
	    first = lb.classes_[np.where(proba == first)]
	    second = lb.classes_[np.where(proba == second)]
	    third = lb.classes_[np.where(proba == third)]
	    print(first, second, third)
	return prob, first, second, third