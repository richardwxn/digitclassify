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
        self.featurematrix=None
    def trainweight(self):
        # Try different intial method here
        # No bias term now
        weightmatrix=np.zeros((10,28*28))
        featurematrix=np.zeros(28*28)

        # bias term
        # weightmatrix=np.zeros(10,28*28+1)
        # featurematrix=np.zeros(28*28+1)

        #
        learningrate=0.05

        infile = open("/Users/newuser/Downloads/digitdata/trainingimages.txt", 'r')
        inlabel = open(
            "/Users/newuser/Downloads/digitdata/traininglabels.txt", 'r')
        labels = inlabel.readlines()
        print(labels.__len__())

        intial=infile.tell()
        for epoach in xrange(50):
            j = 0
            count=0
            i=0
            infile.seek(intial)
            for line in infile.readlines():
                for word in line.strip("\n"):
                    if word == '+' or word == '#':
                        featurematrix[j] += 1
                    j += 1
                i+=1
                if i % 28 == 0 and i != 0 and i!=140000:
                    truelabel=labels[i/28-1]
                    count+=1
                    tempmax=-10000
                    templabel=0
                    j=0
                    for index in xrange(10):
                        tempres=np.inner(featurematrix,weightmatrix[index,:])
                        if(tempres>tempmax):
                            templabel=index
                            tempmax=tempres
                    if(templabel!=truelabel):
                        weightmatrix[truelabel,:]+=learningrate*featurematrix[:]
                        weightmatrix[templabel,:]-=learningrate*featurematrix[:]
                    featurematrix.fill(0)

        return weightmatrix
    def classify(self,weightmatrix):
        testfile = open("/Users/newuser/Downloads/digitdata/testimages", 'r')
        testlabel = open("/Users/newuser/Downloads/digitdata/testlabels", 'r')
        alllines = testfile.readlines()
        testlabels = testlabel.read().splitlines()
        featurematrix=np.zeros(28*28)
        templabels=[]
        j = 0
        i=0
        for line in alllines:
            for word in line.strip("\n"):
                if word == '+' or word == '#':
                    featurematrix[j] += 1
                j += 1
            i+=1
            if i % 28 == 0 and i != 0:
                tempmax=-10000
                templabel=0
                j=0
                for index in xrange(10):
                    tempres=np.inner(featurematrix,weightmatrix[index,:])
                    if(tempres>tempmax):
                        templabel=index
                        tempmax=tempres
                templabels.append(str(templabel))
                featurematrix.fill(0)
        templabels=np.array(templabels)
        testlabels=np.array(testlabels)
        accuracy=float(np.sum(templabels==testlabels))/float(testlabels.__len__())
        print(accuracy)
        print(confusion_matrix(testlabels,templabels))

class KNN(object):
    def train(self):
        infile = open("/Users/newuser/Downloads/digitdata/trainingimages.txt", 'r')
        inlabel = open(
            "/Users/newuser/Downloads/digitdata/traininglabels.txt", 'r')
        labels = inlabel.readlines()
        newfeaturematrix=np.zeros(28*28)
        trainingset=[]
        testset=[]
        j = 0
        i=0
        for line in infile.readlines():
            for word in line.strip("\n"):
                if word == '+' or word == '#':
                    newfeaturematrix[j] += 1
                j += 1
            i+=1
            if i % 28 == 0 and i != 0 and i!=140000:
                j=0
                trainingset.append(newfeaturematrix)
                newfeaturematrix.fill(0)
        j = 0
        i=0
        newfeaturematrix=np.zeros(28*28)
        testfile = open("/Users/newuser/Downloads/digitdata/testimages", 'r')
        for line in testfile.readlines():
            for word in line.strip("\n"):
                if word == '+' or word == '#':
                    newfeaturematrix[j] += 1
                j += 1
            i+=1
            if i % 28 == 0 and i != 0:
                j=0
                testset.append(newfeaturematrix)
                newfeaturematrix.fill(0)
        return trainingset,testset
    # different distance measurement
    def euclideanDistance(self,instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

    def getNeighbors(self,trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def getResponse(self,neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        print(classVotes)
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    def getAccuracy(self,testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0

    def main(self):
        testfile = open("/Users/newuser/Downloads/digitdata/testimages", 'r')
        testlabel = open("/Users/newuser/Downloads/digitdata/testlabels", 'r')
        testlabels = testlabel.readlines()
        predictions=[]
        trainingSet,testSet=self.train()
        k = 3
        for x in range(len(testlabels)):
            neighbors = self.getNeighbors(trainingSet, testSet[x], k)
            print(neighbors.__len__())
            result = self.getResponse(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        accuracy = self.getAccuracy(testSet, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')

class Perceptronfornews(object):
    def __init__(self):
        self.weightmatrix=None
        self.featurematrix=None
    def trainweight(self):
        infile = open(
        "/Users/newuser/Downloads/8category/8category.training.txt", 'r')
        worddict=[]
        for line in infile.readlines():
            for word in line.strip("\n").split(' ')[1:]:
                newword = word.split(':')[0]
                worddict.append(newword)
        worddict = np.unique(np.asarray(worddict))
        print(len(worddict))
        weightmatrix=np.ones((8,len(worddict)))
        # Try different intial method here
        # No bias term now

        featurematrix=np.zeros(len(worddict))

        # bias term
        # weightmatrix=np.zeros(10,28*28+1)
        # featurematrix=np.zeros(28*28+1)

        #
        learningrate=0.05

        for line in infile.readlines():
            label = line.strip("\n").split(' ')[0]
            for word in line.strip("\n").split(' ')[1:]:
                newword, count = word.split(':')
                featurematrix[np.where(worddict==newword)]+=int(count)
            tempmax=-10000
            templabel=0
            j=0
            for index in xrange(8):
                tempres=np.inner(featurematrix,weightmatrix[index,:])
                if(tempres>tempmax):
                            templabel=index
                            tempmax=tempres
            if(templabel!=label):
                    weightmatrix[label,:]+=learningrate*featurematrix[:]
                    weightmatrix[templabel,:]-=learningrate*featurematrix[:]
                    featurematrix.fill(0)

        return weightmatrix,worddict
    def classify(self,weightmatrix,worddict):
        testfile = open(
        "/Users/newuser/Downloads/8category/8category.testing.txt", 'r')
        alllines = testfile.readlines()
        featurematrix=np.zeros(len(worddict))
        templabels=[]
        groundlabel=[]
        for line in alllines:
            truelabel=int(line.strip("\n").split(' ')[0])
            groundlabel.append(truelabel)
            for word in line.strip("\n").split(' ')[1:]:
                    newword, count = word.split(':')
                    if newword not in worddict:
                        continue
                    featurematrix[np.where(worddict==newword)]+=int(count)
            tempmax=-10000
            templabel=0
            for index in xrange(8):
                tempres=np.inner(featurematrix,weightmatrix[index,:])
                if(tempres>tempmax):
                        templabel=index
                        tempmax=tempres
                templabels.append(templabel)
                featurematrix.fill(0)
        templabels=np.array(templabels)
        testlabels=np.array(groundlabel)
        accuracy=float(np.sum(templabels==testlabels))/float(testlabels.__len__())
        print(accuracy)
        print(confusion_matrix(testlabels,templabels))
if "__main__" == __name__:
    classifier=Perceptron()
    weight=classifier.trainweight()
    classifier.classify(weight)


    classifier2=Perceptronfornews()
    weight2,wordict=classifier2.trainweight()
    classifier2.classify(weight2,wordict)

    # Knn=KNN()
    # Knn.main()