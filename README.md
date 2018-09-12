# IRIS 
Inflorescence Recognition Intelligence System
## What was that for
The project was made for the Deep Learning class in TU Darmstadt. It is focused on using deep learning for image recognition.
## What is it about
Tree recognition is a popular topic in machine learning. We've collected 2000 images of 10 species of blooming trees (both from the internet and our own) in order to make a recognition system wrapped as a chat bot interface.
## What's inside
A not very well made repository that contains code for training a neural network based on VGG-16 for inflorescence image classification.
The collected dataset (not published due to its size) contains 2000 images of inflorescences of 10 tree species growing in St. Petersburg, Russia. The accuracy obtained on the testing set is ~93% after 200 epochs of training. Based on this model, a Telegram bot was created for instant tree recognition (the token is omitted). The code for the network is based on the tutorial by Adrian Rosebrock published on https://www.pyimagesearch.com. Finally, there is code for cropping an image based on the object edges found on an image; it serves for faster training and smaller overfitting gap.
