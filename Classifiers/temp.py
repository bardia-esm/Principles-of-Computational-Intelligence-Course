from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def separateByClass(train_set):
	separated = {}
	#train_num = sample * (1 - split_ratio)
	for class_name, class_item in enumerate(train_set):
		#img_class = int(img_add[img_add.rfind('classes') + len('classes') + 1: img_add.rfind('/')])
		if class_name not in separated.keys():
				separated[class_name] = []
		for img_add in class_item:
			img_features = extract_features(img_add)
			separated[class_name].append(img_features)

	return separated


def extract_features(img_add):
    img = cv2.imread(img_add, 0)
    img_resized = cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA)
    x_sum = np.sum(img_resized, axis=0)
    y_sum = np.sum(img_resized, axis=1)
    img_scaled = np.append(x_sum, y_sum)

    return img_scaled

x = [['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/0/2016-06-07-14_33_16.338415_0.jpg', 
'/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/0/2016-06-14-18_17_07.141786_0.jpg', 
'/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/0/20387-IMG32_row7_dig2.jpg'],
 ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/1/2016-06-11-19_44_17.863456_1.jpg',
  '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/1/37787-IMG7_row7_dig2.jpg', 
  '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/1/587.jpg'], 
  ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/2/12941-IMG16_row3_dig1.jpg',
   '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/2/2016-06-18-14_12_42.354965_2.jpg',
    '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/2/12286-IMG7_row7_dig1.jpg'],
     ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/3/876.jpg',
      '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/3/2016-06-21-16_23_32.990905_3.jpg', 
      '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/3/2016-06-18-14_11_36.629644_3.jpg'], 
      ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/4/2016-06-15-16_29_12.001483_4.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/4/2016-06-14-21_33_34.695640_4.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/4/2016-06-29-00_41_44.571690_4.jpg'], ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/5/2016-06-20-02_11_01.283405_5.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/5/2016-06-21-15_56_35.449749_5.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/5/34582-IMG27_row10_dig2.jpg'], ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/6/282-IMG13_row6_dig2.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/6/486.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/6/2016-06-14-18_14_11.809515_6.jpg'], ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/7/2016-06-12-22_21_30.565394_7.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/7/239.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/7/801.jpg'], ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/8/2016-06-19-19_38_10.484978_8.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/8/2016-06-14-20_30_41.909482_8.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/8/12047-IMG54_row4_dig2.jpg'], ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/9/890.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/9/697.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/9/2016-06-19-19_38_21.053284_9.jpg'], ['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/10/2016-06-07-12_39_28.640741_10.jpg', '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/10/2016-06-07-15_06_07.687528_10.jpg',
'/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/10/95-IMG11_row13_dig0.jpg'],
['/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/11/14879-IMG41_row14_dig4.jpg', 
'/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/11/2016-06-11-19_46_40.009612_11.jpg',
 '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/11/25228-IMG26_row14_dig3.jpg']]

t = separateByClass(x)
#print(t)

#{0: [array([253, 254, 254, ..., 254, 255, 254], dtype=uint8), array([255, 255, 255, ..., 253, 255, 254], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8)], 1: [array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8)], 2: [array([255, 255, 255, ..., 255, 255, 255], dtype=uint8),array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 253, 255], dtype=uint8)], 3: [array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([254, 253, 252, ..., 255, 254, 254], dtype=uint8), array([255, 255, 255, ..., 255, 254, 255],dtype=uint8)], 4: [array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8)], 5: [array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8)], 6: [array([255, 254, 253, ..., 254, 255, 255],dtype=uint8), array([255, 254, 254, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 254, 255], dtype=uint8)], 7: [array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8)], 8: [array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 254, 255, ..., 252, 253, 255], dtype=uint8), array([254, 255, 254, ..., 254, 254, 254], dtype=uint8)], 9: [array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([253, 253, 253, ..., 254, 254, 255], dtype=uint8)], 10: [array([254, 255, 254, ..., 253, 254, 254], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8)], 11: [array([254, 255, 254, ..., 253, 253, 255], dtype=uint8), array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), array([252, 254, 254, ..., 255, 254, 253], dtype=uint8)]}

a = {0:[(0, 2, 6, 5), (3, 13, 6, 3)], 1:[(2, 52, 13, 2), (9, 1, 2, 9)]}
#x = [(attribute - 1, attribute + 1) for attribute in zip(*a[0])]
#print(zip(*a[0]))
'''for at in zip(*a[0]):
      print(at)'''
t = '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/11/25228-IMG26_row14_dig3.jpg'
v = extract_features(t)
print(v)
index = random.randrange(0)
print(index)




knn_sample = [20, 30, 40, 50, 60, 70, 80, 90, 100]
knn_acc = [16.666666666666664, 19.166666666666668, 11.904761904761903, 15.196078431372548, 23.75, 20.48611111111111, 20.98765432098765, 23.055555555555557, 15.909090909090908]

Bayessian_sample = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
Bayessian_acc = [44.11764705882353, 56.060606060606055, 51.515151515151516, 55.87219343696027, 53.9776462853386, 54.83014861995754, 53.76056964842012, 54.75095785440614, 54.89404641775983, 53.71702637889688, 54.47418221140848]

parsen_sample = [20, 30, 40, 50, 60, 70, 80, 90, 100]
parsen_ acc = [100.0, 100.0, 100.0, 57.14285714285714, 77.77777777777779, 100.0, 91.66666666666666, 76.92307692307693, 100.0]

plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(parsen_sample, Bayessian_acc)