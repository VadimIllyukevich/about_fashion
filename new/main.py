import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import utils

CLASSES = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_train = x_train / 255
y_train = utils.to_categorical(y_train, 10)
model = load_model('smart_nn.h5')


def use_neural_network(self):
    predictions = model.predict(x_train)
    plt.imshow(x_train[self].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    print(CLASSES[np.argmax(predictions[self])])
    print(CLASSES[np.argmax(y_train[self])])


if __name__ == "__main__":
    for x in range(5):
        dressnumber = int(input())
        use_neural_network(dressnumber)