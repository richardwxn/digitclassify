__author__ = 'newuser'

import numpy as np
import math
from math import log10
from PIL import Image
from sklearn.metrics import confusion_matrix
import heapq
import operator

class Perceptron(object):
    def __init__(self):
        self.weightmatrix=None

    def trainweight(self):
        # Try different intial method here
        # No bias term now
        weightmatrix=np.zeros(10,28*28)
        featurematrix=np.zeros(28*28)

        # bias term
        # weightmatrix=np.zeros(10,28*28+1)
        # featurematrix=np.zeros(28*28+1)

        infile = open("/Users/newuser/Downloads/digitdata/trainingimages.txt", 'r')
        inlabel = open(
            "/Users/newuser/Downloads/digitdata/traininglabels.txt", 'r')
        labels = inlabel.readlines()
        j = 0
        i=0
        for line in infile.readlines():
            for word in line.strip("\n"):
                if word == '+' or word == '#':
                    featurematrix[j] += 1
                j += 1
            i+=1
            j = 0
            tempmax=-10000
            templabel=0
            for index in xrange(10):
                if(np.inner(featurematrix,weightmatrix[index,:])>tempmax):
                    templabel=index

            if(templabel!=labels[i])  :
                weightmatrix


    def classify(self):


class KNN(object):
    def train(self):
    # different distance measurement
    def euclideanDistance(instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def
    def getNeighbors(trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
            dist = euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def getResponse(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getAccuracy(testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0

    def main(self):
        trainingSet=[]
        testSet=[]
        testfile = open("/Users/newuser/Downloads/digitdata/testimages", 'r')
        testlabel = open("/Users/newuser/Downloads/digitdata/testlabels", 'r')
        alllines = testfile.readlines()
        testlabels = testlabel.readlines()
        predictions=[]
        k = 3
        for x in range(len(testlabels)):
            neighbors = getNeighbors(trainingSet, testSet[x], k)
            result = getResponse(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        accuracy = getAccuracy(testSet, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')


