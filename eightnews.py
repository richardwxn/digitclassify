import numpy as np
import math
from math import log10
from PIL import Image
import heapq
def builddictionary():
    infile = open(
        "/Users/newuser/Downloads/spam_detection/train_email.txt", 'r')
    infile = open(
        "/Users/newuser/Downloads/sentiment/rt-train.txt", 'r')
    global bayesmatrix
    global worddict
    for line in infile.readlines():
        for word in line.strip("\n").split(' ')[1:]:
            newword = word.split(':')[0]
            worddict.append(newword)
    worddict = np.unique(np.asarray(worddict))
    print(len(worddict))
    bayesmatrix=np.ones((8,len(worddict)))

def buildconditional():
    global worddict
    global spam
    global normal
    global pclass
    global bayesmatrix
    infile = open(
        "/Users/newuser/Downloads/spam_detection/train_email.txt", 'r')
    infile = open(
        "/Users/newuser/Downloads/8category/8category.training.txt", 'r')

    for line in infile.readlines():
        label = line.strip("\n").split(' ')[0]
        pclass[int(label)]+=1


        for word in line.strip("\n").split(' ')[1:]:
            newword, count = word.split(':')
            bayesmatrix[int(label),np.where(worddict==newword)]+=int(count)
    for i in xrange(bayesmatrix.shape[0]):
        rowsum=sum(bayesmatrix[i,:])
        for j in xrange(len(worddict)):
            bayesmatrix[i,j]/=rowsum

def classifyspam():
    testfile = open(
        "/Users/newuser/Downloads/spam_detection/test_email.txt", 'r')
    testfile = open(
        "/Users/newuser/Downloads/8category/8category.testing.txt", 'r')
    global pclass
    global worddict,bayesmatrix
    pclass=pclass/np.sum(pclass)
    frequency=np.log(pclass)

    alllines=testfile.readlines()
    truecount=0
    label = []

    for line in alllines:
            truelabel=int(line.strip("\n").split(' ')[0])
            maxpossibility=-100000
            assignedlabel=0
            for possiblelabel in xrange(8):
                curpossibility=0.0
                for word in line.strip("\n").split(' ')[1:]:
                    newword, count = word.split(':')
                    if newword not in worddict:
                        continue
                    curpossibility += np.log(bayesmatrix[possiblelabel][np.where(worddict==newword)])
                curpossibility+=frequency[possiblelabel]
                if curpossibility > maxpossibility:
                    maxpossibility = curpossibility
                    assignedlabel = possiblelabel
            if truelabel == assignedlabel:
                truecount += 1
    k_keys_sorted_by_values=np.zeros((8,20))
    for potential in xrange(9):
        k_keys_sorted_by_values[potential][:]=heapq.nlargest(20, bayesmatrix[potential][:], key=spam.get)
    print('top 20 words for news '+str(k_keys_sorted_by_values))
    print(float(truecount) / len(alllines))

if __name__ == "__main__":
    worddict = []
    numberclass = np.array
    spam = {}
    normal = {}
    pclass = np.zeros((8))
    bayesmatrix=[]
    builddictionary()
    buildconditional()
    classifyspam()
