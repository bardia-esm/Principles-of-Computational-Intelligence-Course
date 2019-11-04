import numpy as np
import cv2
import glob
import operator
from utils import splitDataset, extract_features, getAccuracy

def getPredictions(train_set, test_set, radius):
    predictions = []
    for test_class, test_item in enumerate(test_set):
        class_pred = []
        for test_img in test_item:
            result = find_neighbors(train_set, test_img, radius)
            if result != False:
                class_pred.append(result)
        predictions.append(class_pred)
    return predictions

def find_neighbors(train_set, test_sample, radius):
    test_scaled = extract_features(test_sample)
    neighbors = {}
    mapping = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0}
    
    for img_class, class_item in enumerate(train_set):
        for img_add in class_item:
            img_scaled = extract_features(img_add)

            dist = np.linalg.norm(test_scaled - img_scaled)

            if dist < radius:
                neighbors[str(img_class) + '/' + img_add[img_add.rfind('/') + 1:-4]] = dist
        
    if len(neighbors.keys()) != 0:
        for key in neighbors.keys():
            idx = int(key[:key.find('/')])
            mapping[idx] = mapping[idx] + 1
            
        mapping = { k:v for k, v in mapping.items() if v }
        print('-------------')
        print(mapping)
        prediction = max(mapping.items(), key=operator.itemgetter(1))[0]
        print('test blenogs to class ' + str(prediction))
        return prediction
    else:
        print('-------------')
        print('there is no neighbor in the given radius')
        return False
    


def main():
    dataset_add = '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/'
    splitRatio = 0.33
    sample_per_class = 20
    radius = 6000

    train_set, test_set = splitDataset(dataset_add, sample_per_class, splitRatio)
    predictions = getPredictions(train_set, test_set, radius)
    print('-------------')
    accuracy = getAccuracy(predictions)
    print('Accuracy is :' + str(accuracy) + '%')

main()