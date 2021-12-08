import neural_network

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nn = neural_network.neural_network_mnist()
    nn.prepare_data()
    nn.create_network()

    nn.train_model(1024, 10)

    nn.evaluate_model()
    for i in range(10):
        nn.predict_random_value()

    nn.confusion_matrix()
