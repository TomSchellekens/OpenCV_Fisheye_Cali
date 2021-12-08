import traceback

import PySimpleGUI as Sg
import numpy as np

import neural_network
from PIL import Image

batch_sizes = []

# Create the network in the GUI
nn = neural_network.neural_network_mnist()
nn.prepare_data()
nn.create_network()
test_loss, test_acc = nn.evaluate_model()
nn.confusion_matrix()

layout_train_model = [
    [Sg.Text('Enter the number of epochs:'), Sg.Combo(values=[i for i in range(1, 21)], key='EPOCHS')],
    [Sg.Text('Select the batch size:'), Sg.Combo(values=[str(2 ** i) for i in range(0, 14)], key='BATCH')],
    [Sg.Button('Train Model')],
    [Sg.Text('Test loss: '), Sg.Text(test_loss, key='LOSS')],
    [Sg.Text('Test Accuracy: '), Sg.Text(test_acc, key='ACC')],
    [Sg.Button('Reset Model')]
]

layout_predict_model = [
    [Sg.Text("Predicted value: "), Sg.Text('', key='PRED_VALUE')],
    [Sg.Text("True value: "), Sg.Text('', key='TRUE_VALUE')],
    [Sg.Image(key='IMAGE')],
    [Sg.Button('Predict')]
]

layout_conf_mat = [
    [Sg.Image('confusion_matrix.png',key='CONF_MAT')]
]

layout = [[Sg.Column(layout_train_model, size=(240, 320)), Sg.VSeparator(),
           Sg.Column(layout_predict_model, size=(240, 320))],
          [Sg.Column(layout_conf_mat)]]

Sg.theme('DarkTeal9')

window = Sg.Window('Neural Network GUI', layout, element_justification='c')

while True:  # Event Loop
    event, values = window.Read()
    print(event, values)
    if event in (None, 'Exit'):
        break
    elif event == 'Train Model' or event == 'Reset Model':
        if event == 'Train Model':
            nn.train_model(batch_size=int(values['BATCH']), epochs=values['EPOCHS'])
        elif event == 'Reset Model':
            nn.reset_model()
        test_loss, test_acc = nn.evaluate_model()
        window['LOSS'].update(test_loss)
        window['ACC'].update(test_acc)
        nn.confusion_matrix()
        window['CONF_MAT'].update('confusion_matrix.png')

    elif event == 'Predict':
        image_predict, predicted_value, true_value = nn.predict_random_value()
        # Image come out the model like (1,784) but this must reshape to (28,28)
        # and then convert the image to a real image with size (200,200)
        image_predict = image_predict.reshape(28, 28)
        img = Image.fromarray(np.uint8(image_predict * 255), 'L')
        img = img.resize((200, 200))
        img.save('image.png')
        # Update the fields
        window['PRED_VALUE'].update(predicted_value)
        window['TRUE_VALUE'].update(true_value)
        window['IMAGE'].update('image.png')
window.Close()
