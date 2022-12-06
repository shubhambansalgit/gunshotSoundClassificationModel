import numpy as np 
import pandas as pd

import os
import copy
import librosa
from sklearn import preprocessing


from constants import TRAINING_DATA_FOLDER, PROCESSED_TRAINING_DATA, X_COPY_TXT, Y_COPY_NEW_TXT, HOP_LENGTH, N_FFT


list_folder = os.listdir(TRAINING_DATA_FOLDER)


raw_x = []
raw_y = []
for i in list_folder:
    list_file_wave = os.listdir(TRAINING_DATA_FOLDER + i)
    for j in list_file_wave:
        filename = TRAINING_DATA_FOLDER + i + '/' + j
        if j == 2:
            break
        data, sr = librosa.load(filename,sr=22050)
        
        raw_x.append(data)
        raw_y.append(i)


x = []
y = []
for i in range(len(raw_x)):
    data = raw_x[i]
    if len(data) == 44100:
        mfcc_data = np.array(librosa.feature.mfcc(data, n_fft=N_FFT,hop_length=HOP_LENGTH,n_mfcc=128))
        x.append(mfcc_data)
        y.append(raw_y[i])



x_copy = copy.copy(x)
y_copy = copy.copy(y)


le = preprocessing.LabelEncoder()
le.fit(y_copy)
y_copy_new = le.transform(y_copy)


len_x_copy = []
for i in range(len(x_copy)):
    temp = x_copy[i].shape
    len_x_copy.append(temp)


value, count = np.unique(len_x_copy, return_counts=True)
print(value,count)


x_copy = np.array(x_copy)

print(type(x_copy))
print(type(y_copy_new))


# np.savetxt(PROCESSED_TRAINING_DATA + X_COPY_TXT, x_copy, fmt='%d')
# np.savetxt(PROCESSED_TRAINING_DATA + Y_COPY_NEW_TXT, y_copy_new, fmt='%d')

# a2 = np.loadtxt('test1.txt', dtype=int)

score = y#['1', '2', '3']

with open("file.txt", "w") as f:
    for s in score:
        f.write(str(s) +"\n")
scor = []
with open("file.txt", "r") as f:
  for line in f:
    scor.append(line.strip())

if score == scor:
    print("hjk")
print(scor)
print(score)

f = open('foo', 'wb')

np.save(f, y)
data = np.load(open('foo'))

print(y)
print(data)


# # X_train, X_test, y_train, y_test = train_test_split(x_copy, y_copy_new, test_size=0.25, random_state=123, stratify=y)
# # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)
