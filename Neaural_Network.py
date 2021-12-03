from keras.models import Sequential
from keras import layers

from lib.utils.common import calculate_distribution
from lib.loader import get_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

x, y, names = get_data()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

input_dim = x_train.shape[1]

nn_model = Sequential()
nn_model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
nn_model.add(layers.Dense(1, activation='sigmoid'))
nn_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
nn_model.summary()

nn_model.fit(x_train, y_train,
             epochs=20,
             verbose=False,
             validation_data=(x_test, y_test),
             batch_size=10)
loss, accuracy = nn_model.evaluate(x_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))
print("Training Loss: {:.4f}".format(loss))
loss, accuracy = nn_model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Testing Loss:  {:.4f}".format(loss))



'''
###   OUTPUTS

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 10)                107780    
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11        
=================================================================
Total params: 107,791
Trainable params: 107,791
Non-trainable params: 0


Training Accuracy: 0.0000
Training Loss: -164574.0938
Testing Accuracy:  0.0000
Testing Loss:  -164816.6875

'''