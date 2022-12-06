import numpy as np 
import pandas as pd

import os
import copy
import librosa
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import LSTM, Dense,Dropout,Flatten
from tensorflow import keras


from constants import TRAINING_DATA_FOLDER, SAVED_MODELS_FOLDER, MODEL_NAME, HOP_LENGTH, N_FFT, EPOCHS


history_dict=history.history
loss_values=history_dict['loss']
acc_values=history_dict['accuracy']
val_loss_values = history_dict['val_loss']
val_acc_values=history_dict['val_accuracy']
epochs=range(1,51)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
ax1.plot(epochs,loss_values,'co',label='Training Loss')
ax1.plot(epochs,val_loss_values,'m', label='Validation Loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epochs,acc_values,'co', label='Training accuracy')
ax2.plot(epochs,val_acc_values,'m',label='Validation accuracy')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()



# from tensorflow.math import confusion_matrix

# TrainLoss, Trainacc = model.evaluate(X_train,y_train)
# TestLoss, Testacc = model.evaluate(X_test, y_test)
# y_pred=model.predict(X_test)
# print('Confusion_matrix: ',confusion_matrix(y_test, np.argmax(y_pred,axis=1)))


# import librosa.display
# fig, ax = plt.subplots(figsize=(20,7))
# librosa.display.specshow(X_train[0],sr=sr, cmap='cool',hop_length=hop_length)
# ax.set_xlabel('Time', fontsize=15)
# ax.set_title('MFCC', size=20)
# plt.colorbar()
# plt.show()