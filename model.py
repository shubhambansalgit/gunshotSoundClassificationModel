import numpy as np 
# import pandas as pd

# import os
# import copy
import librosa
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import LSTM, Dense,Dropout,Flatten
from tensorflow import keras


from constants import SAVED_MODELS_FOLDER, MODEL_NAME, EPOCHS, PROCESSED_TRAINING_DATA, X_COPY_TXT, Y_COPY_NEW_TXT



x_copy = np.loadtxt(PROCESSED_TRAINING_DATA + X_COPY_TXT, dtype=int)
y_copy_new = np.loadtxt(PROCESSED_TRAINING_DATA + Y_COPY_NEW_TXT, dtype=int)



X_train, X_test, y_train, y_test = train_test_split(x_copy, y_copy_new, test_size=0.25, random_state=123, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)



input_shape = x_copy[0].shape

model = keras.Sequential()
model.add(LSTM(128,input_shape=input_shape,return_sequences=True))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(9, activation='softmax'))
model.summary()
model.compile(loss="SparseCategoricalCrossentropy", optimizer="adam", metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=72, validation_data=(X_val, y_val), shuffle=False)


model.save(SAVED_MODELS_FOLDER + MODEL_NAME)