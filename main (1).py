
import cv2
import glob
import random
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import csv
lines = []
images = []
measurements = []



with open(  './data/driving_log.csv') as csvfile :
     reader = csv.reader(csvfile)
     next(reader) #skips the first line

     for line in reader:
          lines.append(line)



for line in lines:
     source_path = line[0]
     filename = source_path.split('/')[-1]
     current_path =  './data/IMG/' + filename
     image = cv2.imread(current_path)
     images.append(image)

     measurements.append(float(line[3]))

x_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Cropping2D(cropping=((65, 20), (0, 0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
model.fit_generator(x_train, y_train,validation_steps = 0.2 , shuffle = True,nb_epoch = 6)

#model.fit_generator(x_train, y_train, epochs=6, shuffle=True)
#, validation_steps=0.2
model.save('model102.h5')
