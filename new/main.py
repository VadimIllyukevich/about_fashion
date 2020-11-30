from train import *
import numpy as np
import matplotlib.pyplot as plt


def use_neural_network(self):
    plt.imshow(x_train[self].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    print(classes[np.argmax(predictions[self])])
    print(classes[np.argmax(y_train[self])])


if __name__ == "__main__":
    for x in range(5):
        dressnumber = int(input())
        use_neural_network(dressnumber)
