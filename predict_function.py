import numpy as np
from keras.preprocessing import image
from keras_applications.inception_v3 import preprocess_input


def predict(model, img):
    """Run model prediction on image
    Args:
    model: keras model
    img: PIL format image
    Returns:list of predicted labels and their probabilities """,
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x,data_format='channels_first')
    preds = model.predict(x)
    return preds[0]