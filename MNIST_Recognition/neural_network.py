import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import mnist
from PIL import Image



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
        self.model.save_weights('model_default.h5')
        self.model.summary()

    def train_model(self, batch_size, epochs):
        self.model.fit(self.x_train, self.y_train, batch_size, epochs)

    def evaluate_model(self):
        self.test_loss, self.test_acc = self.model.evaluate(self.x_test, self.y_test)
        return round(self.test_loss, 3), round(self.test_acc, 3)

    def predict_random_value(self):
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred_classes = np.argmax(self.y_pred, axis=1)

        random_idx = np.random.choice(len(self.x_test))
        x_sample = self.x_test[random_idx]
        y_true = np.argmax(self.y_test, axis=1)
        y_sample_true = y_true[random_idx]
        y_sample_pred_class = self.y_pred_classes[random_idx]

        return x_sample, y_sample_pred_class, y_sample_true

    def confusion_matrix(self):
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred_classes = np.argmax(self.y_pred, axis=1)

        y_true = np.argmax(self.y_test, axis=1)
        confusion_mtx = confusion_matrix(y_true, self.y_pred_classes)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel('Preditect label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        img = Image.open("confusion_matrix.png").resize((1200, 800))
        img.save("confusion_matrix.png")

    def reset_model(self):
        self.model.load_weights('model_default.h5')
