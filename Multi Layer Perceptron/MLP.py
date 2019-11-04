import numpy as np
from utils import *
import keras
import matplotlib.pyplot as plt


num_of_features = 100
num_of_neurons_in_hidden_layer = 10
classes = 12

base_dir = '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/'

hidden_layer_weights = np.random.normal(size=(num_of_features + 1, num_of_neurons_in_hidden_layer))
last_layer_weights = np.random.normal(size=(num_of_neurons_in_hidden_layer + 1, classes))
learning_rate = 0.1
epochs = 100

train_set, test_set = splitDataset(base_dir, 100, 0.3)

acc_arr = []

for epoch in range(epochs):
        total, correct = 0, 0
        for class_num, class_item in enumerate(train_set):
                for img_idx, img_add in enumerate(class_item):
                        total = total + 1
                        inp = extract_features(img_add)
                        inp_reshaped = np.append(inp, 1).reshape(1, len(inp) + 1)

                        hidden_layer = np.dot(inp_reshaped, hidden_layer_weights)
                        hidden_layer = sigmoid(hidden_layer)
                        hidden_layer_reshaped = np.append(hidden_layer, [[1]], axis=1)
                        

                        output_layer = (np.dot(hidden_layer_reshaped, last_layer_weights))
                        output_layer = sigmoid(output_layer)
                        target = keras.utils.to_categorical(class_num, num_classes=12)
                        e = target - output_layer

                        #if np.array_equal(e, np.zeros(shape=(1, classes))) == False:
                        if np.abs(np.sum(e[0], axis=0)/ len(e[0])) > 0.009:
                                last_layer_delta = e * output_layer * (1 - output_layer)
                                new_last_weights = np.dot(hidden_layer_reshaped.reshape(len(hidden_layer_reshaped[0]), 1), last_layer_delta) * learning_rate

                                temp = (np.dot(last_layer_weights[:-1,:], last_layer_delta.reshape(len(last_layer_delta[0]), 1)))
                                temp = temp.reshape(1, len(temp))
                                hidden_layer_delta = hidden_layer * (1 - hidden_layer) * temp
                                new_hidden_weights = np.dot(inp_reshaped.reshape(len(inp_reshaped[0]), 1), hidden_layer_delta) * learning_rate
                                
                                hidden_layer_weights = hidden_layer_weights + new_hidden_weights
                                last_layer_weights = last_layer_weights + new_last_weights
                        
                        else:
                                correct  = correct + 1
    
        print('correct num', correct)
        print('total num', total)
        acc_arr.append((correct / total) * 100)

plt.title(str(round(0.7 * len(train_set[0]))) + ' samples per class')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(range(0, epochs), acc_arr)
plt.axis([0, epochs, 0, 100])
plt.show()