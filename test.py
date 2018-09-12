#from loadmodel import LoadModel
from cropim import CropIm
from classifyim import Classify
import numpy as np

print("I'm here")
#model, lb = LoadModel()
def get_image():
   # global model, lb
    img = CropIm("..\\testim\\acer_platanoides__03.jpg")
    first, second, third = Classify(img)
    print("success")
    if first == "none":
        print("I'm not sure what is it. Please, try another image1")
    else:
        first = np.array2string(first)
        first = first.replace("_", " ")
        first = first.replace("['", "")
        first = first.replace("']", "")
        first = first.capitalize()
        second = np.array2string(second)
        second = second.replace("_", " ")
        second = second.replace("['", "")
        second = second.replace("']", "")
        second = second.capitalize()
        third = np.array2string(third)
        third = third.replace("_", " ")
        third = third.replace("['", "")
        third = third.replace("']", "")
        third = third.capitalize()
        print("This is most probably %s. It might also be %s or %s, though!" % (first, second, third))
get_image()

