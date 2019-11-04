import keras
from keras.models import load_model 
import numpy as np

model = load_model('/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/DigitDetection/ML-digitRecognition/Model//home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/DigitDetection/ML-digitRecognition/Model/model_final.h5')

print(model.summary())


    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(nb_classes, activation='softmax'))

    return model


def setup_to_finetune(model):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True