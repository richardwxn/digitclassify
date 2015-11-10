__author__ = 'newuser'

import numpy as np
import math
from math import log10
from PIL import Image


def parsetrainingdata():
    infile = open("/Users/newuser/Downloads/digitdata/trainingimages.txt", 'r')
    inlabel = open(
        "/Users/newuser/Downloads/digitdata/traininglabels.txt", 'r')
    labels = inlabel.readlines()
    global bayesmatrix
    global numberclass
    bayesmatrix = np.zeros((10, 28, 28))
    bayesmatrix.fill(1)
    numberclass = np.zeros(10)
    i = 0
    j = 0
    label = int(labels[0])
    for line in infile.readlines():
        for word in line.strip("\n"):
            if word == '+' or word == '#':
                bayesmatrix[label][i % 28][j] += 1
            j += 1
        if i % 28 == 0 and i != 0:
            numberclass[label] += 1
            if(i / 28 < len(labels)):
                label = int(labels[i / 28])
        j = 0
        i += 1
    for x in xrange(0, 10):
        for i in xrange(0, 28):
            for j in xrange(0, 28):
                bayesmatrix[x][i][j] = float(
                    bayesmatrix[x][i][j]) / float((numberclass[x] + 2 * 1))
    # print(bayesmatrix[2])
    totalcount = sum(numberclass)
    print(numberclass)
    for i in xrange(numberclass.size):
        numberclass[i] = float(numberclass[i]) / float(totalcount)
    print(numberclass)
    # print(bayesmatrix[0])
    # print(featuremap)


def classifydigit():
    global bayesmatrix
    global numberclass
    testfile = open("/Users/newuser/Downloads/digitdata/testimages", 'r')
    testlabel = open("/Users/newuser/Downloads/digitdata/testlabels", 'r')
    alllines = testfile.readlines()
    testlabels = testlabel.readlines()
    confusion_matrix = np.zeros((10, 10))
    i, j = 0, 0
    truecount = 0
    for testcasenumber in xrange(len(testlabels)):
        # print(testcasenumber)
        truelabel = int(testlabels[testcasenumber])
        assignedlabel = 0
        maxpossibility = -10000000
        possibleline = alllines[testcasenumber * 28:(testcasenumber + 1) * 28]
        for possiblelabel in xrange(0, 10):
            curpossibility = 0
            count = 0
            j = 0
            for line in possibleline:
                j = 0
                for word in line:
                    if len(word) == 0 or word == ' ':
                        curpossibility += log10(1 - bayesmatrix[possiblelabel][count][j])
                    else:
                        curpossibility += log10(
                            bayesmatrix[possiblelabel][count][j])
                    j += 1
                    if j == 28:
                        break

                count += 1
            curpossibility += log10(numberclass[possiblelabel])
            if curpossibility > maxpossibility:
                maxpossibility = curpossibility
                assignedlabel = possiblelabel
        # print(assignedlabel)
        if truelabel == assignedlabel:
            truecount += 1
        confusion_matrix[assignedlabel][truelabel] += 1
    print(float(truecount) / float(len(testlabels)))
    confusesum = sum(confusion_matrix)
    confusion_matrix / confusesum
    print(confusion_matrix)


def getimages(c):
    global bayesmatrix
    img = Image.new('RGB', (84, 28), "black")
    pixels = img.load()
    xmin = 100
    xmax = -100
    for i in xrange(28):
        for j in xrange(28):
            x = log10(bayesmatrix[c][i][j])
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
    for i in xrange(28):
        for j in xrange(28):
            x = log10(bayesmatrix[c][i][j])
            pixels[i, j] = (i, j, getColor(x, xmin, xmax))
    return img


def oddsRatioMap(c1, c2):
    img = Image.new('RGB', (84, 28), "black")
    pixels = img.load()
    img_c1 = getimages(c1)
    pixel1 = img_c1.load()
    img_c2 = getimages(c2)
    pixel2 = img_c2.load()
    odds_ratio = np.zeros((28, 28))

    x, xmin = 100, 100
    xmax = -100
    for i in xrange(28):
        for j in xrange(28):
            x = log10(bayesmatrix[c1][j][i] / bayesmatrix[c2][j][i])
            if x < xmin:
                xmin = x
            if x > max:
                xmax = x

            for i in xrange(28):
                for j in xrange(28):
                    odds_ratio[j][i] = bayesmatrix[c1][
                        j][i] / bayesmatrix[c2][j][i]
                    x = log10(odds_ratio[j][i])
                    pixels[i, j] = getColor(x, xmin, xmax)
            # print(pixel1[i,j])
            pixels[i + 28, j] = pixel1[i, j]
            pixels[i + 56, j] = pixel2[i, j]
    # print(pixels)
    # print(img)
    img.save("/Users/newuser/Downloads/digitdata/oddratioimage.png")


def getColor(x, xmin, xmax):
    r = 0
    g = 0
    b = 0

    y = x
    m = (240) / (xmin - xmax)
    a = (-240 * xmax) / (xmin - xmax)

    y = y * (m) + a

    h = y
    s = 100
    v = 100
    h = max(0, min(360, h))
    s = max(0, min(100, s))
    v = max(0, min(100, v))

    s /= 100
    v /= 100

    if s == 0:
        r = g = b = v
        r = round(255 * r)
        g = round(255 * g)
        b = round(255 * b)

    else:
        h /= 60
        i = math.floor(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i == 0:
            r = v
            g = t
            b = p
        elif i == 1:
            r = q
            g = v
            b = p

        elif i == 2:
            r = p
            g = v
            b = t
        elif i == 3:
            r = p
            g = q
            b = v
        elif i == 4:
            r = t
            g = p
            b = v
        else:

            r = v
            g = p
            b = q
        r = round(255 * r)
        g = round(255 * g)
        b = round(255 * b)
    return (int(r) << 16) | (int(g) << 8) | int(b)


def builddictionary():
    infile = open(
        "/Users/newuser/Downloads/spam_detection/train_email.txt", 'r')
    infile = open(
        "/Users/newuser/Downloads/sentiment/rt-train.txt", 'r')
    global worddict
    for line in infile.readlines():
        for word in line.strip("\n").split(' ')[1:]:
            newword = word.split(':')[0]
            worddict.append(newword)
    worddict = set(worddict)
    print(len(worddict))


def buildconditional():
    global worddict
    global spam
    global normal
    global pclass
    infile = open(
        "/Users/newuser/Downloads/spam_detection/train_email.txt", 'r')
    infile = open(
        "/Users/newuser/Downloads/sentiment/rt-train.txt", 'r')
    for line in infile.readlines():
        label = line.strip("\n").split(' ')[0]
        if int(label) == 1:
            pclass[0] += 1
        else:
            pclass[1] += 1
        for word in line.strip("\n").split(' ')[1:]:
            newword, count = word.split(':')
            if int(label) == 1:
                if newword in spam:
                    spam[newword] += int(count)
                else:
                    spam[newword] = 1.0
            else:
                if newword in normal:
                    normal[newword] += int(count)
                else:
                    normal[newword] = 1.0
    for word in worddict:
        if word not in spam:
            spam[word] = 1.0
        if word not in normal:
            normal[word] = 1.0
    spamsize = np.sum(np.asarray(list(spam.itervalues())))
    normalsize = np.sum(np.asarray(list(normal.itervalues())))
    for word in spam:
        spam[word] = spam[word] / spamsize
    for word in normal:
        normal[word] = normal[word] / normalsize


def classifyspam():
    testfile = open(
        "/Users/newuser/Downloads/spam_detection/test_email.txt", 'r')
    testfile = open(
        "/Users/newuser/Downloads/sentiment/rt-test.txt", 'r')
    global pclass
    global worddict, spam, normal
    spamfreq = float(pclass[0]) / float(pclass[0] + pclass[1])
    normalfreq = 1 - spamfreq
    spamword, normalword = np.log(spamfreq), np.log(normalfreq)
    assignedlabel = []
    label = []
    for line in testfile.readlines():
        label.append(int(line.strip("\n").split(' ')[0]))
        for word in line.strip("\n").split(' ')[1:]:
            newword, count = word.split(':')
            if newword not in worddict:
                continue
            spamword += np.log(spam[newword])
            normalword += np.log(normal[newword])
        if spamword >= normalword:
            assignedlabel.append(1)

        else:
            assignedlabel.append(-1)
        spamword = np.log(spamfreq)
        normalword = np.log(normalfreq)
    correct = np.sum(np.asarray(assignedlabel) == np.asarray(label))
    print(float(correct) / len(assignedlabel))

# def eightnewsdataset():



if __name__ == "__main__":
    worddict = []
    numberclass = np.array
    spam = {}
    normal = {}
    pclass = [0, 0]
    builddictionary()
    buildconditional()
    classifyspam()

    # bayesmatrix=[]
    # parsetrainingdata()
    # classifydigit()
    #
    # #
    # oddsRatioMap(4,9)
    # oddsRatioMap(5,3)
    # oddsRatioMap(7,9)
    # oddsRatioMap(8,3)

# def classify(self, testData):
# #
# #
# """
# Classify the data based on the posterior distribution over labels.
# #
# You shouldn't modify this method.
# """
# guesses = []
# self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
# for datum in testData:
# posterior = self.calculateLogJointProbabilities(datum)
# guesses.append(posterior.argMax())
# self.posteriors.append(posterior)
#
#   def calculateLogJointProbabilities(self, datum):
# """
# Returns the log-joint distribution over legal labels and the datum.
# Each log-probability should be stored in the log-joint counter, e.g.
# logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
#     #
# To get the list of all possible features or labels, use self.features and
# self.legalLabels.
#     #
#
#   def findHighOddsFeatures(self, label1, label2):
#     """
#     Returns the 100 best features for the odds ratio:
#             P(feature=1 | label1)/P(feature=1 | label2)
#
#     Note: you may find 'self.features' a useful way to loop through all possible features
#     """
#     featuresOdds = []
#
#
#     return featuresOdds
