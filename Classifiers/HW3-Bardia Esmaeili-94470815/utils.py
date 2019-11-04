from sklearn.model_selection import train_test_split
import glob
import random
import os
import cv2
import numpy as np

def splitDataset(dataset_add, sample_per_class, splitRatio):
    train_set = []
    test_set = []

    classes = [directory[0][directory[0].rfind('/') + 1:] for directory in os.walk(dataset_add)]
    classes.remove('')
    classes = [ int(x) for x in classes]
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
        train_set.append(train_temp)
        test_set.append(test_temp)

    return train_set, test_set

def extract_features(img_add):
    img = cv2.imread(img_add, 0)
    img_resized = cv2.resize(img, (50, 50), interpolation = cv2.INTER_AREA)
    x_sum = np.sum(img_resized, axis=0)
    y_sum = np.sum(img_resized, axis=1)
    img_scaled = np.append(x_sum, y_sum)

    return img_scaled


def getAccuracy(predictions):
    correct = 0
    '''for i in range(len(test_set)):
    if test_set[i][-1] == predictions[i]:
    correct += 1'''
    for pred_class, pred_item in enumerate(predictions):
        for pred in pred_item:
            if pred_class == pred:
                correct += 1
    pred_total_num = 0
    for i in predictions:
        pred_total_num = pred_total_num + len(i)

    return (correct/float(pred_total_num)) * 100.0
