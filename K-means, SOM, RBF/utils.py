import glob
import random
import os
import cv2
import numpy as np
import math
from keras.models import load_model
from sklearn.model_selection import train_test_split

extractor_model = load_model('/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Multi Layer Perceptron/extract.h5')
extractor_model.pop()
extractor_model.pop()

def splitDataset(dataset_add, sample_per_class, splitRatio):
    train_set = []
    test_set = []

    classes = os.listdir(dataset_add)
    classes = [int(x) for x in classes]
    classes.sort()

    for form in classes:
        class_selected_imgs = []
        class_all_imgs = glob.glob(dataset_add + str(form) + "/*.jpg")

        while len(class_selected_imgs) < sample_per_class:
            if (len(class_all_imgs) > 0):
                index = random.randrange(len(class_all_imgs))
                class_selected_imgs.append(class_all_imgs.pop(index))
            else:
                break  

        train_temp, test_temp = train_test_split(class_selected_imgs, test_size = splitRatio)
        '''train_set.append(train_temp)
        test_set.append(test_temp)'''

        train_set = train_set + train_temp
        test_set = test_set + test_temp

    return train_set, test_set

def extract_features(img_add):
    img = cv2.imread(img_add, 0)
    img_resized = cv2.resize(img, (20, 20), interpolation = cv2.INTER_AREA)
    img_flattened = img_resized.reshape(-1)
    img_features = extractor_model.predict(np.array([img_flattened]))[0]

    return img_features

def sigmoid(x):
    return 1 / (1 + (np.exp(-x)))