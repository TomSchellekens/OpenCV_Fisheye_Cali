import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import mnist


class neural_network_mnist:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        np.random.seed(0)
        self.num_classes = 10
        self.model = Sequential()
        self.test_loss = 0
        self.test_acc = 0
        self.y_pred = 0
        self.y_pred_classes = 0

    def show_possible_labels(self):
        f, ax = plt.subplots(1, self.num_classes, figsize=(20, 20))

        for i in range(0, self.num_classes):
            sample = self.x_train[self.y_train == i][0]
            ax[i].imshow(sample, cmap='gray')
            ax[i].set_title("Label: {}".format(i), fontsize=16)
        plt.show()

    def prepare_data(self):
        #
        self.y_train = keras.utils.np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.np_utils.to_categorical(self.y_test, self.num_classes)

        #
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # Reshape data
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)

    def create_network(self):
        self.model.add(Dense(units=128, input_shape=(784,), activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(units=10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train_model(self, batch_size, epochs):
        self.model.fit(self.x_train, self.y_train, batch_size, epochs)

    def evaluate_model(self):
        self.test_loss, self.test_acc = self.model.evaluate(self.x_test, self.y_test)
        print("Test loss: {}\nTest Accuracy: {}".format(self.test_loss, self.test_acc))

    def predict_random_value(self):
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred_classes = np.argmax(self.y_pred, axis=1)

        random_idx = np.random.choice(len(self.x_test))
        x_sample = self.x_test[random_idx]
        y_true = np.argmax(self.y_test, axis=1)
        y_sample_true = y_true[random_idx]
        y_sample_pred_class = self.y_pred_classes[random_idx]

        plt.title("Predicted: {}\nTrue value: {}".format(y_sample_pred_class, y_sample_true), fontsize=16)
        plt.imshow(x_sample.reshape(28, 28), cmap='gray')
        plt.show()

    def confusion_matrix(self):
        y_true = np.argmax(self.y_test, axis=1)
        confusion_mtx = confusion_matrix(y_true, self.y_pred_classes)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel('Preditect label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        plt.show()

    def find_erros(self):
        # Investigate errors in the model
        y_true = np.argmax(self.y_test, axis=1)
        errors = (self.y_pred_classes - y_true != 0)
        y_pred_classes_errors = self.y_pred_classes[errors]
        y_pred_erros = self.y_pred[errors]
        y_true_erros = y_true[errors]
        x_test_erros = self.x_test[errors]

        y_pred_erros_proba = np.max(y_pred_erros, axis=1)
        true_error_proba = np.diagonal(np.take(y_pred_erros, y_true_erros, axis=1))
        diff_errors_pred_true = y_pred_erros_proba - true_error_proba

        sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)
        top_idx_diff_errors = sorted_idx_diff_errors[-5:]

        # show top errors
        num = len(top_idx_diff_errors)
        f, ax = plt.subplots(1, num, figsize=(30, 30))

        for i in range(0, num):
            idx = top_idx_diff_errors[i]
            sample = x_test_erros[idx].reshape(28, 28)
            y_t = y_true_erros[idx]
            y_p = y_pred_classes_errors[idx]
            ax[i].imshow(sample, cmap='gray')
            ax[i].set_title("Predicted Label: {}\nTrue Label: {}".format(y_p, y_t), fontsize=22)

        plt.show()
