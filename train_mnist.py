## First build an mnsit model

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout , Flatten, Activation,Input
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler


## about data
batch_size = 128
num_classes = 26
epochs = 25

df = pd.read_csv("handwritten.csv").astype('float32')
df.rename(columns={'0':'label'}, inplace=True)

print('Loading done')
# Splite data the X - Our data , and y - the prdict label
X = df.drop('label',axis = 1)
y = df['label']

 #input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y)

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

input_shape = (28,28,1)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', name='preds'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

model.save('handwritten.h5')