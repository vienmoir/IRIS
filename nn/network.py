# -*- coding: utf-8 -*-
# code partly based on the tutotial by https://www.pyimagesearch.com
# a modified VGG-16 with PReLU and 2 dense layers
from keras import backend as K # should be TensorFlow
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.advanced_activations import PReLU


class FloNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        
        # for TensorFlow <3
        inputShape = (height, width, depth)
        chanDim = -1
        
        # for other guys
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        # Conv -> PReLU -> Pool
        # 32 filters with 3 x 3 kernel
        model.add(Conv2D(32, (3, 3), padding = "same", 
                         input_shape = inputShape))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        # 3 x 3 pool size
        model.add(MaxPooling2D(pool_size = (3, 3)))
        model.add(Dropout(0.25))
        
        # (Conv -> PReLU) x 2 -> Pool
        # MORE FILTERS FOR THE GOD OF FILTERS
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(64, (3, 3), padding = "same"))
        # 'twas good, let's repeat
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        # 2 x 2 pool size
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        
        # (Conv -> PReLU) x 2 -> Pool
        # more filters?
        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # (Conv -> PReLU) x 2 -> Pool
        # more filters?
        model.add(Conv2D(256, (3, 3), padding = "same"))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(256, (3, 3), padding = "same"))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(256, (3, 3), padding = "same"))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # (Conv -> PReLU) x 2 -> Pool
        # more filters?
        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(512, (3, 3), padding = "same"))
        model.add(PReLU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))
        
        # FC -> PReLU -> Softmax

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model