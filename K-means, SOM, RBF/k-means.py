import numpy as np
import random
from utils import *

def k_means(k, dataset_add):

    train_set, test_set = splitDataset(dataset_add, 5, 0.3)
    random.shuffle(train_set)
    random.shuffle(test_set)
    #centroids = {}
    centroids = []

    for i in range(k):
        center = random.choice(train_set)
        centroids.append(extract_features(center))
        #centroids[center[115:-4]] = extract_features(center)
        train_set.remove(center)
    
    clusters = {k: [] for k in range(k)}
    cluster_img_add = {k: [] for k in range(k)}


    for _ in range(10):
        for img_add in train_set:
            distance = 0.
            img_center_idx = 0
            img = extract_features(img_add)
            #for center_idx, center_add in enumerate(centroids.keys()):
            for center_idx, center in enumerate(centroids):
                #center = extract_features(center_add)
                temp = np.linalg.norm(center - img)
                if ((temp < distance and center_idx != 0) | (center_idx == 0.)):
                    distance = temp
                    img_center = center_idx
            
            clusters[img_center].append(img)
            cluster_img_add[img_center].append(img_add)
        
        for idx, _ in enumerate(centroids):
            if len(clusters[idx]) > 0:
                centroids[idx] = np.mean(clusters[idx], axis=0)
    
    
k_means(12, '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/')