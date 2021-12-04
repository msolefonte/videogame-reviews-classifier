from keras.models import Sequential
from keras import layers

from lib.utils.common import calculate_distribution
from lib.loader import get_data, get_train_test_split_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Formatting

plt.rc('font', size=14)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=12)
plt.rcParams['figure.constrained_layout.use'] = True

x, y, names = get_data()

plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['mse']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.xlabel('Input')
    plt.ylabel('Mean Absolute Error')
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('neural-net_vs_MAE.jpg')
    plt.clf()

def run():
    x_train, x_test, y_train, y_test = get_train_test_split_data()

    input_dim = x_train.shape[1]

    nn_model = Sequential()
    nn_model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    nn_model.add(layers.Dense(1, activation='relu'))
    nn_model.compile(optimizer='adam',
                     loss='mse',
                     metrics=['mse'])
    nn_model.summary()

    model = nn_model.fit(x_train, y_train,
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
    plot_history(model)


def main():
    run()


if __name__ == "__main__":
    main()




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
_________________________________________________________________

Training Accuracy: 78.2631
Training Loss: 78.2631
Testing Accuracy:  136.1775
Testing Loss:  136.1775

'''
