'''
Code to demonstrate in-sample Convex Hull Iteration as a technique to estimate
out-of-convex-hull model performance.
Mayowa Osibodu. 2022.
'''


import numpy as np, pandas as pd
from scipy.spatial import KDTree
from sklearn.datasets import load_iris, load_wine
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#Set up data arrays.
def init():
	#Import Data
	dataset = load_iris() #wine()
	X,Y = dataset.data, dataset.target

	#Sort data expanding outwards from central data point
	tree = KDTree(X)
	median = np.median(X,axis=0)
	sorted_outwards = tree.query(median,k=len(X))[1]

	return X, Y, sorted_outwards


#Calculate Gower distance between two points
def gower(a,b,inHull):
	sum = 0
	maxcoords = []
	mincoords = []
	for index in range(len(a)):
		maxcoords.append(max(inHull[...,index]))
		mincoords.append(min(inHull[...,index]))

	for index in range(len(a)):
		sum+= abs(a[index]-b[index])/(maxcoords[index]-mincoords[index])
	return sum/(len(a))


#Given an in-sample convex hull, calculate prediction accuracy for all in-sample out-of-hull points.
def test(X, Y, model, hullsize, sorted_outwards):

	#Points in the convex hull
	hullX = np.array([X[i] for i in sorted_outwards[:hullsize]])
	hullY = np.ravel([Y[i] for i in sorted_outwards[:hullsize]])

	model.fit(hullX,hullY)

	gowerdistance = []
	accuracy = []
	central_x = X[sorted_outwards[0]]

	#Calculate model accuracy for all external points.
	for i in range(hullsize, len(X)):
		x = X[sorted_outwards[i]]
		y = Y[sorted_outwards[i]]
		gowerdistance.append(gower(x,central_x,hullX))
		accuracy.append([1 if model.predict([x])==y else 0][0])

	return gowerdistance, accuracy



def analyze(X,Y,model,sorted_outwards):

	results = {}

	#Convex hull iteration. Begins with the innermost 20% of the dataset and expands outwards.
	for hullsize in range(int(0.2*len(X)),len(X)):
		distances, accuracies = test(X,Y,model,hullsize,sorted_outwards)
		for distance in range(len(distances)):
			if distances[distance] in results:
				results[distances[distance]].append(accuracies[distance])
			else:
				results[distances[distance]]= [accuracies[distance]]

	#Group the accuracy values into bins.
	bins = {}
	for bin in range(0,2000): #Intervals of 0.001
		binmark = bin/1000.
		bins[binmark] = []
		for distance in results:
			if distance>binmark-0.1 and distance<= binmark:
				for val in results[distance]:
					bins[binmark].append(val)

	x = sorted(bins.keys())
	y = []

	#Averages accuracy values in each bin so we have one accuracy value per interval of Gower distance.
	for element in x:
		if bins[element] != []:
			y.append(float(sum(bins[element]))/len(bins[element]))
		else:
			y.append(None)

	#Remove null values - distances no data point was found at
	data = {}
	for i in range(len(x)):
		if y[i]!=None:
			data[x[i]] = y[i]

	x = np.array(sorted(data.keys()))
	y = [data[i] for i in sorted(data.keys())]

	return x,y


print('\nGenerating data...\n')
X, Y, sorted_outwards = init()

names = ['Decision Tree', 'Logistic Regression', 'SVM', 'KNN', 'Random Forest', 'Neural Net']



classifiers = [
	DecisionTreeClassifier(),
	LogisticRegression(max_iter=1000),
	svm.SVC(),
	neighbors.KNeighborsClassifier(),
	RandomForestClassifier(),
	MLPClassifier(max_iter=1000)]



for pair in zip(names, classifiers):
	name = pair[0]
	model = pair[1]
	print('Working on', name)
	x, y = analyze(X,Y,model, sorted_outwards)
	plt.plot(x,y, label=name)

plt.xlabel('Gower distance')
plt.ylabel('Prediction Accuracy')
plt.grid()
plt.legend(bbox_to_anchor=(1, 1)) # bbox_transform=plt.gcf().transFigure

plt.title('Prediction Accuracy as a Function of Gower Distance:')
plt.show()
