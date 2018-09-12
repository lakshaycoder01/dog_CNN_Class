import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas import Series
from .neural_nets_cnn import nnnet
from keras.preprocessing.image import ImageDataGenerator


num_channels=3
img_rows=128
img_cols=128

#reading labels file
labels=pd.read_csv('./dog/labels.csv')
# unique_Class for Dense layer output dynamic
unique_no_classes=len(set(labels['breed']))
print(unique_no_classes)

breed_count=labels['breed'].value_counts(dropna=True)
print(breed_count.shape)

#making_matrix_output_variables(In keras you can  also use labelEncoder,to_categorical,one_hot_encoder)
target=pd.Series(labels['breed'])
one_hot=pd.get_dummies(target,sparse=True)
one_hotted_labels=np.asarray(one_hot)

#reading and plotting single image

img_1=cv2.imread('./dog/train/000bec180eb18c7604dcecc8fe0dba07.jpg',0)
plt.title('Original Image')
plt.imshow(img_1)

#plotting same image after doing resize
img_1_resize= cv2.resize(img_1, (img_rows, img_cols)) 
print (img_1_resize.shape)
plt.title('Resized Image')
plt.imshow(img_1_resize)

#converting_images and labels data into array list using tqdm which provide better support for looping

x_feature=[]
y_labels=[]

i=0
for f,img in tqdm(labels.values):
	
    train_img=cv2.imread('./dog/train/{}.jpg'.format(f))
    label=one_hotted_labels[i]
    train_img_resize=cv2.resize(train_img,(img_rows,img_cols))
    x_feature.append(train_img_resize)
    y_labels.append(label)
    i+=1
	
print(len(x_feature))
print(len(y_labels))

#normalization of image_data
x_train_data=np.array(x_feature,np.float32)/255.
print(x_train_data.shape)

#labels_
y_train=np.array(y_labels,np.uint8)
print(y_train.shape)

x_train,x_val,y_train,y_val=train_test_split(x_train_data,y_train,test_size=0.2,random_state=42) #spliiting of training data into train and validation

#getting test_data ids
test_ids=pd.read_csv('./dog/sample_submission.csv')
test_img_ids=test_ids['id']
print(test_img_ids.head())

#making list out of test_data and labels using tqdm

x_test_features=[]
for f in tqdm(test_img_ids.values):
    img=cv2.imread('./dog/test/{}.jpg'.format(f))
    img_resize=cv2.resize(img,(img_rows,img_cols))
    x_test_features.append(img_resize)


x_test_data=np.array(x_test_features,np.float32)/255.
print(x_test_data.shape)


model= nnnet.model(img_rows,img_cols,num_channels,unique_no_classes)

data_argumenter=ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

history=model.fit_generator(data_argumenter.flow(x_train,y_train,batch_size=64),validation_data=(x_val,y_val),steps_per_epoch=len(x_train)//64,epochs=28,verbose=1)

#after this we can plot the model loss and accuracy

#also we can save the model for future use

results = model.predict(x_test_data)
prediction = pd.DataFrame(results)






