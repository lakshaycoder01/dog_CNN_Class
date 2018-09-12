from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten,Embedding
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import keras.backend as K



class nnnet:
    @staticmethod
    def model(img_rows,img_cols,num_channels,no_classes):
        model=Sequential()
        input_shape=(img_rows,img_cols,num_channels)
        if(K.image_data_format()=="channels_first"):
            input_shape=(num_channels,img_rows,img_cols)
        model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu',input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(500,activation='relu'))
        model.add(Dense(no_classes,activation='sigmoid'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.summary()
        return model