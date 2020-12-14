import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

CLASSES = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
IMAGEPATH = 'pictures/test_image.jpg'


def use_neural_network():
    model = load_model('smart_nn.h5')
    img = image.load_img(IMAGEPATH, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape(1, 150, 150, 3)
    x /= 255
    prediction = model.predict(x)
    prediction = np.argmax(prediction)
    print("Номер класса:", prediction)
    print("Название класса", CLASSES[prediction])


if __name__ == "__main__":
    use_neural_network()
