import numpy as np
import math
from math import log10
from PIL import Image
import heapq
from sklearn.metrics import confusion_matrix
def builddictionary():
    infile = open(
        "/Users/newuser/Downloads/8category/8category.training.txt", 'r')
    global weightmatrix
    global worddict
    for line in infile.readlines():
        for word in line.strip("\n").split(' ')[1:]:
            newword = word.split(':')[0]
            worddict.append(newword)
    worddict = np.unique(np.asarray(worddict))
    print(len(worddict))
    weightmatrix=np.ones((8,len(worddict)))

def buildconditional():
    global worddict
    global spam
    global normal
    global pclass
    global weightmatrix
    global featurematrix
    infile = open(
        "/Users/newuser/Downloads/8category/8category.training.txt", 'r')

    #  Multinominal

    uniquesize=np.zeros((8,1))
    for line in infile.readlines():
        label = line.strip("\n").split(' ')[0]
        pclass[int(label)]+=1


        for word in line.strip("\n").split(' ')[1:]:
            newword, count = word.split(':')
            featurematrix[int(label),np.where(worddict==newword)]+=int(count)
            uniquesize[int(label),0]+=1
    for i in xrange(bayesmatrix.shape[0]):
        rowsum=sum(bayesmatrix[i, :])
        for j in xrange(len(worddict)):
            bayesmatrix[i,j]/=(rowsum+uniquesize[i, 0])

    #   Bernouli
    # for line in infile.readlines():
    #     label = line.strip("\n").split(' ')[0]
    #     pclass[int(label)]+=1
    #
    #     for word in line.strip("\n").split(' ')[1:]:
    #         newword, count = word.split(':')
    #         bayesmatrix[int(label),np.where(worddict==newword)]+=1
    # for i in xrange(bayesmatrix.shape[0]):
    #     for j in xrange(len(worddict)):
    #         bayesmatrix[i,j]/=(pclass[i]+2)


def classifyspam():

    testfile = open(
        "/Users/newuser/Downloads/8category/8category.testing.txt", 'r')
    global pclass
    global worddict,bayesmatrix
    pclass=pclass/np.sum(pclass)
    frequency=np.log(pclass)

    alllines=testfile.readlines()
    truecount=0
    label = []
    groundlabel=[]
    #Multinominal

    for line in alllines:
            truelabel=int(line.strip("\n").split(' ')[0])
            groundlabel.append(truelabel)
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
            label.append(assignedlabel)
            if truelabel == assignedlabel:
                truecount += 1

    print confusion_matrix(groundlabel,label)
    # for line in alllines:
    #         truelabel=int(line.strip("\n").split(' ')[0])
    #         maxpossibility=-100000
    #         assignedlabel=0
    #         appearword=[]
    #         for possiblelabel in xrange(8):
    #             curpossibility=0.0
    #             for word in line.strip("\n").split(' ')[1:]:
    #                 newword, count = word.split(':')
    #
    #                 if newword not in worddict:
    #                     continue
    #                 if possiblelabel == 0:
    #                     appearword.append(newword)
    #                 curpossibility += np.log(bayesmatrix[possiblelabel][np.where(worddict==newword)])
    #             for word in worddict:
    #                 if word not in appearword:
    #                     curpossibility += np.log(1-bayesmatrix[possiblelabel][np.where(worddict==word)])
    #             curpossibility+=frequency[possiblelabel]
    #             if curpossibility > maxpossibility:
    #                 maxpossibility = curpossibility
    #                 assignedlabel = possiblelabel
    #         if truelabel == assignedlabel:
    #             truecount += 1

    # k_keys_sorted_by_values=np.zeros((8,20))
    # for potential in xrange(8):
    #     # k_keys_sorted_by_values[potential][:]=heapq.nlargest(20, bayesmatrix[potential][:], key=spam.get)
    #
    #     k_keys_sorted_by_values[potential][:]=np.argpartition(bayesmatrix[potential][:],-20)[-20:]
    #
    # for index in np.nditer(k_keys_sorted_by_values):
    #     print(worddict[int(index)])
    # print('top 20 words for news '+str(k_keys_sorted_by_values))
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
