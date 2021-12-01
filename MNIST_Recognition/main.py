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





#load data from mnist database
def load_data():
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    return x_train, x_test, y_train, y_test

def show_labels(x_train,y_train, num_classes):
    f, ax = plt.subplots(1, num_classes, figsize=(20, 20))

    for i in range(0, num_classes):
        sample = x_train[y_train == i][0]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title("Label: {}".format(i), fontsize=16)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    num_classes = 10
    x_train, x_test, y_train, y_test = load_data()
    #show_labels(x_train,y_train, num_classes)

    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    #Prepare Data
    #Normalize Data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    #Reshape data
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    print(x_train.shape)

    #Make Neural Network
    model = Sequential()

    model.add(Dense(units=128, input_shape=(784,), activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    #Train model
    batch_size = 512
    epochs = 10
    model.fit(x_train, y_train, batch_size, epochs)

    #Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test loss: {}\nTest Accuracy: {}".format(test_loss, test_acc))

    #predicting with model
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(y_pred)
    print(y_pred_classes)

    #Single test
    random_idx = np.random.choice(len(x_test))
    x_sample = x_test[random_idx]
    y_true = np.argmax(y_test, axis=1)
    y_sample_true = y_true[random_idx]
    y_sample_pred_class = y_pred_classes[random_idx]

    plt.title("Predicted: {}\nTrue value: {}".format(y_sample_pred_class, y_sample_true), fontsize=16)
    plt.imshow(x_sample.reshape(28, 28), cmap='gray')
    plt.show()

    #Confusion Matrix
    confusion_mtx = confusion_matrix(y_true, y_pred_classes)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Preditect label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    plt.show()

    #Investigate error in the model
    errors = (y_pred_classes - y_true != 0)
    y_pred_classes_errors = y_pred_classes[errors]
    y_pred_erros = y_pred[errors]
    y_true_erros = y_true[errors]
    x_test_erros = x_test[errors]







