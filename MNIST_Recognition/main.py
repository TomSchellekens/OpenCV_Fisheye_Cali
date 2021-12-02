import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import mnist

np.random.seed(0)
import neural_network

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nn = neural_network.neural_network_mnist()
    nn.show_possible_labels()
    nn.prepare_data()
    nn.create_network()
    nn.train_model(1024,50)
    nn.evaluate_model()
    for i in range(10):
        nn.predict_random_value()
    nn.confusion_matrix()





















