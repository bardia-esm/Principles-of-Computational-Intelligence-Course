import numpy as np
import cv2
import glob
import operator
from utils import splitDataset, extract_features, getAccuracy

def getPredictions(train_set, test_set, k):
    predictions = []
    for test_class, test_item in enumerate(test_set):
        class_pred = []
        for test_img in test_item:
            result = find_neighbors(train_set, test_img, test_class, k)
            class_pred.append(result)
        predictions.append(class_pred)
    return predictions

def find_neighbors(train_set, test_sample, test_class, k):
    test_scaled = extract_features(test_sample)
    knn = {}
    mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0}
    for img_class, class_item in enumerate(train_set):
        for img_add in class_item:
        
            
            img_scaled = extract_features(img_add)
            dist = np.linalg.norm(test_scaled - img_scaled)
            if len(knn) < k:
                knn[str(img_class) + '/' + img_add[img_add.rfind('/') + 1:-4]] = dist
            elif dist < max(knn.items(), key=operator.itemgetter(1))[1]:
                knn[str(img_class) + '/' + img_add[img_add.rfind('/') + 1:-4]] = dist
                del knn[max(knn.items(), key=operator.itemgetter(1))[0]]
                
    print('-------------')
    for key in knn.keys():
        idx = int(key[:key.find('/')])

        mapping[idx] = mapping[idx] + 1
        
    mapping = { k:v for k, v in mapping.items() if v }

    prediction = max(mapping.items(), key=operator.itemgetter(1))[0]
    print(mapping)
    print(test_sample)
    print('prediction is: ', prediction)
    print('true label is: ', test_class)
    return prediction

def main():
    dataset_add = '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/'
    splitRatio = 0.33
    sample_per_class = 20
    k = 1

    train_set, test_set = splitDataset(dataset_add, sample_per_class, splitRatio)
    
    predictions = getPredictions(train_set, test_set, k)

    print(predictions)
    accuracy = getAccuracy(predictions)
    print('Accuracy is :' + str(accuracy) + '%')

main()
     