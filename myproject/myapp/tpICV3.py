import numpy as np
import os
from subprocess import call
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from itertools import chain
import time

import pdb

def getLabels(pokemonName, labels):
	if(pokemonName == 'bulbasaur'):
		labels.append(0)
	elif(pokemonName == 'charmander'):
		labels.append(1)
	elif(pokemonName == 'pikachu'):
		labels.append(2)
	elif(pokemonName == 'squirtle'):
		labels.append(3)
	elif(pokemonName == 'ekans'):
		labels.append(4)
	elif(pokemonName == 'onix'):
		labels.append(5)

def readFeatures(featuresPath, pokemonData, file):

	fp = open("%s" % (featuresPath+file), 'r')
	lines = fp.readlines()
	auxiliar = lines[0].split("\n")
	auxiliar = auxiliar[0].split()
	pokemonData.append(auxiliar)
	return pokemonData

def getFeatures(pokemonName, pokemonData, labels):

	fileNames = []
	#path = "/home/rafael/tpICV3/images/"
	pathFeatures = "/home/pokedexapp/pokedex-app/myproject/myapp/"+pokemonName+"Features/"
	path = "/home/pokedexapp/pokedex-app/myproject/myapp/"

	for file in os.listdir(pathFeatures):
		
		readFeatures(pathFeatures, pokemonData, file)
		getLabels(pokemonName, labels)


def readFeaturesQuery(featuresPath, pokemonData):

	fp = open("%s.txt" % (featuresPath), 'r')
	lines = fp.readlines()
	auxiliar = lines[0].split("\n")
	auxiliar = auxiliar[0].split()
	pokemonData.append(auxiliar)
	return pokemonData


def estimatePerformance(pokemonData, labels):

	svms = []
	folds = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
	for trainIndex, testIndex in folds.split(pokemonData, labels):

		trainData, testData = pokemonData[trainIndex], pokemonData[testIndex]
		trainLabels, testLabels = labels[trainIndex], labels[testIndex]

		svm = getSVM(trainData, trainLabels)
		svms.append(svm)
		score = svm.score(testData, testLabels)
		predictions = svm.predict(testData)
		confusionMatrix = confusion_matrix(testLabels, predictions)

		print "********************"
		print ("Score: %f" % (score))
		print ("Confusion Matrix: ")
		print confusionMatrix
		print "********************"

	return svms

def getEnsemble(pokemonData, labels):

	svms = [SVC(C = 10.0, gamma = 0.001, kernel = 'rbf', cache_size = 300, decision_function_shape = 'ovo'),
			SVC(C = 10.0, gamma = 0.001, kernel = 'rbf', cache_size = 300, decision_function_shape = 'ovo'),
			SVC(C = 1.0, gamma = 0.01, kernel = 'rbf', cache_size = 300, decision_function_shape = 'ovo'),
			SVC(C = 10.0, gamma = 0.0001, kernel = 'rbf', cache_size = 300, decision_function_shape = 'ovo'),
			SVC(C = 100000.0, gamma = 1e-08, kernel = 'rbf', cache_size = 300, decision_function_shape = 'ovo')]

	index = 0			
	folds = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
	pokemonData = np.asarray(pokemonData, dtype = np.float32)
	labels = np.asarray(labels, dtype = np.float32)

	for trainIndex, testIndex in folds.split(pokemonData, labels):

		trainData, testData = pokemonData[trainIndex], pokemonData[testIndex]
		trainLabels, testLabels = labels[trainIndex], labels[testIndex]

		svms[index].fit(np.concatenate([trainData, testData]), np.concatenate([trainLabels, testLabels]))
		index += 1

	return svms


def getSVM(trainData, labels):

	C, gamma = getBestParameters(trainData, labels)
	print "*******"
	print C
	print gamma
	print "*******"
	svm = SVC(C = C, gamma = gamma, kernel = 'rbf', cache_size = 300, decision_function_shape = 'ovo')
	svm.fit(trainData, labels)

	return svm

def getBestParameters(pokemonData, labels):

	svm = SVC(kernel = 'rbf', cache_size = 300, decision_function_shape = 'ovo')

	parameters = {'C':np.logspace(-2, 10, 13),
                  'gamma': np.logspace(-9, 3, 13)}

	kf = KFold(n_splits = 5, shuffle= True)
	clf  = GridSearchCV(svm, parameters, n_jobs = 4, cv = kf, refit = True)
	X_train, X_test, y_train, y_test = train_test_split(pokemonData, labels, test_size = 0.2, random_state = 0)

	clf.fit(X_train, y_train)

	return clf.best_params_['C'], clf.best_params_['gamma']


def verifyExtension(imageName):

	extension = imageName.split('.')
	if(extension[1] == 'ppm'): return False
	else: return True

def classifyQuery(svms, imageName):

	queryPath = "/home/pokedexapp/pokedex-app/media/documents/2017/01/04/"
	name = imageName.split('.')
	pokemonData = []
	votes = np.zeros(6)

	if(verifyExtension(imageName)):
		call("convert %s %s.ppm" % (queryPath+imageName, queryPath+name[0]), shell = True)

	image = queryPath+name[0]+".ppm"
	featuresPath = queryPath+name[0]

	call("./generatebic %s %s.txt" % (image, featuresPath), shell = True)
	features = readFeaturesQuery(featuresPath, pokemonData)

	for i in svms:
		 votes[int(i.predict(features))] += 1

	print np.argmax(votes)
	
	majorityVote = np.argmax(votes)

	return majorityVote

def getPokemonPage(votes):

	if(int(votes) == 0):
		return "bulbasaur.html"
	elif(int(votes) == 1):
		return "charmander.html"
	elif(int(votes) == 2):
		return "pikachu.html"
	elif(int(votes) == 3):
		return "squirtle.html"
	elif(int(votes) == 4):
		return "ekans.html"
	elif(int(votes) == 5):
		return "onix.html"
	else:
		return "unknown.html"
 
def main():

	pokemonData = []
	labels = []

	getFeatures('bulbasaur', pokemonData, labels)
	getFeatures('charmander', pokemonData, labels)
	getFeatures('pikachu', pokemonData, labels)
	getFeatures('squirtle', pokemonData, labels)
	getFeatures('ekans', pokemonData, labels)
	getFeatures('onix', pokemonData, labels)

	pokemonData = np.asarray(pokemonData, dtype = np.float32)
	labels = np.asarray(labels, dtype = np.float32)
	
	svms = getEnsemble(pokemonData, labels)
	#estimatePerformance(pokemonData,labels)
	classifyQuery(svms, '1.ppm')


if __name__ == '__main__':
	main()