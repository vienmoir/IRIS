# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def CropIm(path):
	img = cv2.imread(path,3) #read img
	b,g,r = cv2.split(img)
	img = cv2.merge([r,g,b]) # revert to rgb
	edges = cv2.Canny(img, 200,400) # apply canny

	# cannied and rgb on input
	def square_it(grim, orim):
		y, x = grim.shape
		xc = x//2 # saving space
		yc = y//2
		if x > y: # horizontal
		    lw = np.sum(grim[0:y, 0:y] == 255) # left square 
		    cw = np.sum(grim[0:y, xc-yc:xc+yc] == 255) #centre
		    rw = np.sum(grim[0:y, x-y:x] == 255) # right
		else: # vertical
		    lw = np.sum(grim[0:x, 0:x] == 255) #  top
		    cw = np.sum(grim[yc-xc:yc+xc, 0:x] == 255)
		    rw = np.sum(grim[y-x:y, 0:x] == 255) # bottom 
		if lw > rw:
		    if lw > cw: #looking for the square with max(white pixels)
			    if x > y:
				    return grim[0:y, 0:y], orim[0:y, 0:y]
			    else:
	    			return grim[0:x, 0:x], orim[0:x, 0:x]
		    else:
			    if x > y:
				    return grim[0:y, xc-yc:xc+yc], orim[0:y, xc-yc:xc+yc]
			    else:
				    return grim[yc-xc:yc+xc, 0:x], orim[yc-xc:yc+xc, 0:x]
		elif rw > cw:
		    if x > y:
			    return grim[0:y, x-y:x], orim[0:y, x-y:x]
		    else:
	    		return grim[y-x:y, 0:x], orim[y-x:y, 0:x]
		else:
		    if x > y:
			    return grim[0:y, xc-yc:xc+yc], orim[0:y, xc-yc:xc+yc]
		    else:
		    	return grim[yc-xc:yc+xc, 0:x], orim[yc-xc:yc+xc, 0:x]

	edcr, orcr = square_it(edges, img) 

	# plt.subplot(221),plt.imshow(img,cmap = 'gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(222),plt.imshow(orcr)
	# plt.title('Cropped  Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(223),plt.imshow(edges,cmap = 'gray')
	# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(224),plt.imshow(edcr, cmap = 'gray')
	# plt.title('Cropped Edge Image'), plt.xticks([]), plt.yticks([])

	#plt.show()
	return orcr