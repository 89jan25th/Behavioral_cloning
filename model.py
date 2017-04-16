import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from preprocess import preprocess # resize image module

lines = []

# Left and right camera input function 
def multicamera(line, lines, correction):
    line_l, line_r = [], []
    line_l = line
    line_r = line
    line_l[3] = float(line[3]) + correction
    line_r[3] = float(line[3]) - correction
    line_l[0] = line_l[1]
    line_r[0] = line_r[2]
    lines.append(line_l)
    lines.append(line_r)
    return lines

# open and multiple camera utilization
correction = 0.2

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#        lines = multicamera(line, lines, correction)
        
#with open('./data/driving_log2.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)
#        lines = multicamera(line, lines, correction)
               
with open('./data/driving_log3.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#        lines = multicamera(line, lines, correction)
              
# Split the samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# generator function
batch_size = 64
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('\\')[-1]
                image = cv2.imread(name)           
                resize = preprocess(image) # resize image
                angle = float(batch_sample[3])
                images.append(resize)
                angles.append(angle) 
                images.append(cv2.flip(resize,1)) # augment a fillped image
                angles.append(angle * -1.0)           
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# The Nvidia CNN model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape = (70, 204, 3)))
# model.add(Cropping2D(cropping=((70,25), (0,0)))) - I Didn't use it.

model.add(Convolution2D(24,5,5)) #Conv 1
model.add(MaxPooling2D())
model.add(Activation('relu'))

model.add(Convolution2D(36,5,5)) #Conv 2
model.add(MaxPooling2D())
model.add(Activation('relu'))

model.add(Convolution2D(48,5,5)) #Conv 3
model.add(MaxPooling2D())
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3)) #Conv 4
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3)) #Conv 5
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, batch_size=64) - No need after introducing generator.
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*2, validation_data=validation_generator, \
    nb_val_samples=len(validation_samples)*2, nb_epoch=10) #train_sample and validation_sample multipled by 2 because of flipping augmentation

model.save('model.h5')
print(model.summary())