from sklearn.model_selection import train_test_split
import glob
import random
import os

def splitDataset(dataset_add, sample_per_class, splitRatio):
    trainSet = []
    testSet = []

    classes = [directory[0][directory[0].rfind('/') + 1:] for directory in os.walk(dataset_add)]
    classes.remove('')
    classes = [ int(x) for x in classes]
    classes.sort()

    for form in classes:
        class_selected_imgs = []
        class_all_imgs = glob.glob(dataset_add + str(form) + "/*.jpg")

        while len(class_selected_imgs) < sample_per_class:
            index = random.randrange(len(class_all_imgs))
            class_selected_imgs.append(class_all_imgs.pop(index))    

        train_temp, test_temp = train_test_split(class_selected_imgs, test_size = splitRatio)
        trainSet.append(train_temp)
        testSet.append(test_temp)

    return trainSet, testSet

base = '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/'

'''classes = [directory[0][directory[0].rfind('/') + 1:] for directory in os.walk(base)]
classes.remove('')
classes = [ int(x) for x in classes]
classes.sort()
print(classes)'''
'''train, test = splitDataset(base, 5, 0.6)

for i in train:
    print('class ', train.index(i))
    for j in i:
        print(str(train.index(i) // 10) + str(train.index(i) % 10) + '/' + j[len(base) + len(str(train.index(i))) + 1: -4])
    print('------------')'''

