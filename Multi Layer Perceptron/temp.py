import numpy as np
import math
import os
import glob
import random
from sklearn.model_selection import train_test_split

'''a = np.array([[1., 2., 3.], [3., 4., 6.], [3, 2, 3]])
#a = np.append(a, [[1. for one in range(3)]], axis=0)
#print(a.reshape(3, 2))


input_arr = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
x = np.array([1, 2, 3])
inp_reshaped = np.append(input_arr[0], 1).reshape(1, len(input_arr[0]) + 1)
print(inp_reshaped[0])
print(x.shape)
print(np.dot(inp_reshaped, x))'''

#print(a[:-1, :])
'''x = -2533.1
y =  1 / (1 + math.exp(-x))
print(y)


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


train, test = splitDataset('/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/', 5, 0.3)
for i in train:
    for j in i:
        print(j)'''
print(np.zeros(shape = (1,3)).shape)