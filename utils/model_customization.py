import os
import shutil
import ssl
import collections



ssl._create_default_https_context = ssl._create_unverified_context

from glob import glob
from os import path
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
WIDTH = 299
HEIGHT = 299
MODEL_FILE = 'models/object_detection2.model'


def split_data():
    construction_helmet = glob('data/train/construction_helmet/*.jpg')
    safety_gloves_construction = glob('data/train/safety_gloves_construction/*.jpg')
    safety_vest = glob('data/train/safety_vest/*.jpg')
    safety_goggles = glob('data/train/safety_goggles/*.jpg')

    construction_helmet_train, construction_helmet_test = train_test_split(construction_helmet, test_size=0.20)
    safety_gloves_construction_train, safety_gloves_construction_test = train_test_split(safety_gloves_construction, test_size=0.20)
    safety_vest_train, safety_vest_test = train_test_split(safety_vest, test_size=0.2)
    safety_goggles_train, safety_goggles_test = train_test_split(safety_goggles, test_size=0.2)

    dst_1 = 'data/test/construction_helmet/'

    for f in construction_helmet_test:
        shutil.copy(f, dst_1)

    dst_2 = 'data/test/safety_gloves_construction/'

    for f in safety_gloves_construction_test:
        shutil.copy(f, dst_2)

    dst_3 = 'data/test/safety_vest/'

    for f in safety_vest_test:
        shutil.copy(f, dst_3)

    dst_4 = 'data/test/safety_goggles/'
    for f in safety_goggles_test:
        shutil.copy(f,dst_4)


def model_customization():

    CLASSES = 4

    # setup model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    print("Base Model------>", base_model)
    x = base_model.output
    print("Base Model Output------>", base_model, x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    from keras.preprocessing.image import ImageDataGenerator


    BATCH_SIZE = 32

    # data prep
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
      # print("Validation generator------->", validation_generator)

    EPOCHS = 5
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 320
    VALIDATION_STEPS = 64



    history = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS)

    model.save(MODEL_FILE)


def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


if __name__ == "__main__":
    # split_data()
    # model_customization()
    # predict()
    img = image.load_img('data/test4.jpg', target_size=(HEIGHT, WIDTH))
    preds = predict(load_model(MODEL_FILE), img)
    print("The pred is", preds)

    pred = {'helmet': preds[0],
            'safety_gloves': preds[1],
            'safety_goggles': preds[2],
            'safety_vest': preds[3]}
    print("Sorted ---->",pred)
