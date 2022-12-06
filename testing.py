import os
import copy
import numpy as np 
import librosa
from tensorflow import keras

from constants import SAVED_MODELS_FOLDER,TESTING_DATA_FOLDER, MODEL_NAME, HOP_LENGTH, N_FFT


loaded_model = keras.models.load_model(SAVED_MODELS_FOLDER + MODEL_NAME)

raw_testing_data_list = []
testing_data_folder = os.listdir(TESTING_DATA_FOLDER)

for testing_file in testing_data_folder:
    test_file = TESTING_DATA_FOLDER + '/' + testing_file
    data, sr = librosa.load(test_file,sr=22050)
    raw_testing_data_list.append(data) 


testing_data_list = []
for raw_testing_data in range(len(raw_testing_data_list)):
    data = raw_testing_data_list[raw_testing_data]
    if len(data) == 44100:
        mfcc_data = np.array(librosa.feature.mfcc(data, n_fft=N_FFT,hop_length=HOP_LENGTH,n_mfcc=128))
        testing_data_list.append(mfcc_data)


x_copy = copy.copy(testing_data_list)

len_x_copy = []
for i in range(len(x_copy)):
    temp1 = x_copy[i].shape
    len_x_copy.append(temp1)
    
value, count = np.unique(len_x_copy, return_counts=True)

x_copy = np.array(x_copy)
print(x_copy)

y_pred=loaded_model.predict(x_copy)


print(np.argmax(y_pred,axis=1))