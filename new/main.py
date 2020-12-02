import numpy as np
from IPython.display import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

CLASSES = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
IMAGEPATH = 'pictures/test_image.jpg'


def use_neural_network():
    Image(IMAGEPATH, width=150, height=150)
    model = load_model('smart_nn.h5')
    img = image.load_img(IMAGEPATH, target_size=(28, 28), color_mode="grayscale")
    x = image.img_to_array(img)
    x = x.reshape(1, 784)
    x = 255 - x
    x /= 255
    prediction = model.predict(x)
    prediction = np.argmax(prediction)
    print("Номер класса:", prediction)
    print("Название Класса", CLASSES[prediction])


if __name__ == "__main__":
    use_neural_network()
