import csv
import random
import math
from utils import splitDataset, extract_features, getAccuracy
 
def separateByClass(train_set):
	separated = {}
	for class_name, class_item in enumerate(train_set):
		if class_name not in separated.keys():
				separated[class_name] = []
		for img_add in class_item:
			img_features = extract_features(img_add)
			separated[class_name].append(img_features)

	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	return summaries
 
def summarizeByClass(separated):
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev + 0.001,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev + 0.001)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, test_set):
	predictions = []
	for test_class, test_item in enumerate(test_set):
		print(test_item)
		class_pred = []
		for test_img in test_item:
			test_features = extract_features(test_img)
			result = predict(summaries, test_features)
			class_pred.append(result)
		predictions.append(class_pred)
		
	return predictions
  
def main():
	dataset_add = '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/'
	splitRatio = 0.33
	sample_per_class = 5

	train_set, test_set = splitDataset(dataset_add, sample_per_class, splitRatio)
	print('train=%d and test=%d' % (len(train_set), len(test_set)))
	# prepare model
	separated_by_class = separateByClass(train_set)
	summaries = summarizeByClass(separated_by_class)
	
	# test model
	predictions = getPredictions(summaries, test_set)
	accuracy = getAccuracy(predictions)
	print('Accuracy: ' + str(accuracy) + '%')
 
main()





'''def loadImages(filename, ):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset'''

'''def loadImages(filename, ):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

base = '/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/Digits Dataset/classes/'

train_sep, test_sep = separateByClass(dataset_add, 5, 0.4)

for img in train:'''
 
'''def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated'''

	 
'''def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries'''

	'''print(len(train_set[0]))
	print(len(test_set[0]))'''