#from numpy.random import seed
#seed(101)
#from tensorflow import set_random_seed
#set_random_seed(101)
#import tensorflow
from flask import Flask, render_template, request, send_from_directory

import pandas as pd
import numpy as np
import tensorflow 

tensorflow.random.set_seed(101)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import binary_accuracy

import os
import cv2

import imageio
import skimage
import skimage.io
import skimage.transform
 #predict the model accuracy in real timme data
#give outside data to predict the model
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
# Source: https://www.kaggle.com/fmarazzi/baseline-keras-cnn-roc-fast-5min-0-8253-lb

datagen = ImageDataGenerator(rescale=1.0/255)

#train_gen = datagen.flow_from_directory(train_path,
#                                        target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
#                                        batch_size=train_batch_size,
#                                        class_mode='categorical')

#val_gen = datagen.flow_from_directory(valid_path,
#                                        target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
#                                        batch_size=val_batch_size,
#                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
valid_path = 'C:/Users/User/DEEP LEARNING/deep learning/base_directories/val_dir'
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(224,224),
                                        batch_size=10,
                                        class_mode='categorical',
                                        shuffle=False)

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
IMAGE_SIZE = [224, 224]
conv_base = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# don't train existing weights
#freeze the weight layers from all the layers in the model
#Now we will be training only the classifiers (FC layers)
for layer in conv_base.layers:
    layer.trainable = False


from glob import glob
# useful for getting number of output classes
folders = glob('C:/Users/User/DEEP LEARNING/TB_directoriesVGG19_feature_extract(try)/train_dir/*')
folders

dropout_dense = 0.3


model = Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()
#from keras.applications.resnet import ResNet50
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')


model.compile(Adam(lr=0.0001), loss='binary_crossentropy', 
              metrics=['accuracy'])
model.load_weights('static/TB_vgg19model_100epoch_VGG19_feature_extract.h5')



COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    import numpy as np
    from keras.preprocessing import image
    # to predict outside data
 #   img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
 #   img_arr = cv2.resize(img_arr, (224,224))
 #   img_arr = img_arr / 255.0
 #   img_arr =  img_arr.reshape(1,224,224,3)
 #   prediction = model.predict(img_arr)
    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    img_arr = cv2.resize(img_arr, (224,224))
    #TB_img=image.load_img(img_arr, target_size=(224,224))
    TB_img=image.img_to_array(img_arr)
    TB_img=np.expand_dims(TB_img,axis=0)
    TB_img=TB_img/255
    prediction = model.predict(TB_img)
    
    test_gen.class_indices
    print(f"Probability the image is Normal is :{prediction}")
    #img_arr = cv2.imread('static/{}.jpg'.format(COUNT))
    #img_arr = cv2.resize(img_arr, (224,224))
    #x=image.img_to_array(img_arr)
    #x=np.expand_dims(x,axis=0)
    #img_data=preprocess_input(x)
    #classes=model.predict(img_data)
    #print(classes)
    #TB_img=image.load_img(img_arr, target_size=(96,96))
    #TB_img=image.img_to_array(TB_img)
    #TB_img=np.expand_dims(img_arr,axis=0)
    #TB_img=TB_img/255
    #TB_img = TB_img.reshape(1, 96,96,3)
    #prediction_probabilities=model.predict(TB_img)
    #test_gen.class_indices
    #output prediction #to predict outside data
    #print(f"Probability the image is Normal is :{prediction_probabilities}")
    x = round(prediction[0,0], 2)
    y = round(prediction[0,1], 2)
    predict = np.array([x,y])
    COUNT += 1
    return render_template('prediction.html', data=predict)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(debug=True, host="localhost", port=5000)

