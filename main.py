__author__ = 'newuser'

import numpy as np
import math
from math import log10
from PIL import Image
def parsetrainingdata():
    infile = open("/Users/newuser/Downloads/digitdata/trainingimages.txt", 'r')
    inlabel=open("/Users/newuser/Downloads/digitdata/traininglabels.txt",'r')
    labels=inlabel.readlines()
    global bayesmatrix
    global numberclass
    bayesmatrix= np.zeros((10,28,28))
    bayesmatrix.fill(1)
    numberclass=np.zeros(10)
    i=0
    j=0
    label=0
    for line in infile.readlines():
        for word in line.strip("\n"):
                if word == '+' or word == '#':
                    bayesmatrix[label][i%28][j]+=1
                j+=1
        if i%27==0 and i!=0 :
            numberclass[label]+=1
            if(i/27<len(labels)):
                label=int(labels[i/27])
        j=0
        i+=1
    for x in xrange(0,10):
        for i in xrange(0,28):
            for j in xrange(0,28):
                bayesmatrix[x][i][j]=float(bayesmatrix[x][i][j])/float((numberclass[x]+2*1))
    # print(bayesmatrix[2])
    totalcount=sum(numberclass)
    print(numberclass)
    for i in xrange(numberclass.size):
        numberclass[i]=float(numberclass[i])/float(totalcount)
    print(numberclass)
    # print(bayesmatrix[0])
    # print(featuremap)
def classifydigit():
    global bayesmatrix
    global numberclass
    testfile = open("/Users/newuser/Downloads/digitdata/testimages", 'r')
    testlabel = open("/Users/newuser/Downloads/digitdata/testlabels", 'r')
    alllines=testfile.readlines()
    testlabels=testlabel.readlines()
    confusion_matrix=np.zeros((10,10))
    i,j=0,0
    truecount=0
    for testcasenumber in xrange(len(testlabels)):
        # print(testcasenumber)
        truelabel=int(testlabels[testcasenumber])
        assignedlabel=0
        maxpossibility=-10000000
        possibleline=alllines[testcasenumber*28:(testcasenumber+1)*28]
        for possiblelabel in xrange(0,10):
            curpossibility=0
            count=0
            j=0
            for line in possibleline:
                j=0
                for word in line:
                    if len(word)==0 or word ==' ':
                        curpossibility+=log10(1-bayesmatrix[possiblelabel][count][j])
                    else:
                        curpossibility+=log10(bayesmatrix[possiblelabel][count][j])
                    j+=1
                    if j==28:
                        break

                count+=1
            curpossibility+=log10(numberclass[possiblelabel])
            if curpossibility>maxpossibility:
                 maxpossibility=curpossibility
                 assignedlabel=possiblelabel
        # print(assignedlabel)
        if truelabel==assignedlabel:
            truecount+=1
        confusion_matrix[assignedlabel][truelabel]+=1
    print(float(truecount)/float(len(testlabels)))


def getimages(c):
    global bayesmatrix
    img = Image.new( 'RGB', (84,28), "black")
    pixels = img.load()
    min = 100
    max = -100
    for i in xrange(28):
        for j in xrange(28):
            x = log10(bayesmatrix[c][i][j])
            if x < min:
                min = x
            if x > max :
                max = x
    for i in xrange(28):
        for j in xrange(28):
            x = log10(bayesmatrix[c][i][j])
            pixels[i,j]=(i,j,getColor(x,min,max))
    return img


def oddsRatioMap(c1, c2):
        img = Image.new( 'RGB', (84,28), "black")
        pixels=img.load()
        img_c1 = getimages(c1)
        img_c2 = getimages(c2)
        odds_ratio=np.array(28,28)

        x,min = 100,100
        max = -100
        for i in xrange(28):
            for j in xrange(28):
                x = log10(bayesmatrix[c1][j][i]/bayesmatrix[c2][j][i])
                if x < min:
                    min = x
                if x > max:
                    max = x

		for i in xrange(28):
			for j in xrange(28):
				odds_ratio[j][i] = bayesmatrix[c1][j][i]/bayesmatrix[c2][j][i]
				x = log10(odds_ratio[j][i])
				pixels[i,j]=(i,j,getColor(x,min,max))
                pixels[i+28,j]=(i+28,j,getColor(x,min,max))
                pixels[i+56,j]=(i+56,j,getColor(x,min,max))

def getColor(x, min, max):
    r = 0
    g = 0
    b = 0

    y = x
    m = (240)/(min-max)
    a = (-240*max)/(min-max)

    y = y*(m) + a

    h = y
    s = 100
    v = 100
    h = math.max(0, math.min(360, h))
    s = math.max(0, math.min(100, s))
    v = math.max(0, math.min(100, v))

    s /= 100
    v /= 100

    if s == 0:
        r = g = b = v
        r = round(255*r)
        g = round(255*g)
        b = round(255*b)

    else:
        h /= 60
        i = math.floor(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i==0:
                r = v
                g = t
                b = p
        elif i == 1:
                r = q
                g = v
                b = p

        elif i== 2:
                r = p
                g = v
                b = t
        elif i==3:
                r = p
                g = q
                b = v
        elif i==4:
                r = t
                g = p
                b = v
        else:

                r = v
                g = p
                b = q
        r = round(255*r)
        g = round(255*g)
        b = round(255*b)
    return (int(r) << 16) | (int(g) << 8)| int(b)



def builddictionary():
    infile = open("/Users/newuser/Downloads/spam_detection/train_email.txt", 'r')
    global worddict
    for line in infile.readlines():
        for word in line.strip("\n").split(' ')[1:]:
            newword=word.split(':')[0]
            worddict.append(newword)
    worddict=set(worddict)
    print(len(worddict))
def buildconditional():
    global worddict
    global spam
    global normal
    global pclass
    infile = open("/Users/newuser/Downloads/spam_detection/train_email.txt", 'r')
    for line in infile.readlines():
        label=line.strip("\n").split(' ')[0]
        if int(label)==0:
            pclass[0] += 1
        else:
            pclass[1] += 1
        for word in line.strip("\n").split(' ')[1:]:
            newword,count=word.split(':')
            if int(label)==0:
                if newword in spam:
                    spam[newword] += int(count)
                else:
                    spam[newword] = 1.0
            else:
                if newword in normal :
                    normal[newword] += int(count)
                else:
                    normal[newword] = 1.0
    for word in worddict:
        if word not in spam:
            spam[word]=1.0
        if word not in normal:
            normal[word]=1.0
    spamsize=sum(spam.itervalues())
    normalsize=sum(normal.itervalues())
    print(normalsize)
    for word in spam:
        spam[word]=float(spam[word])/spamsize*3000
    for word in normal:
        normal[word]=float(normal[word])/normalsize*3000


def bernoulibuildcontional():
    global worddict
    global spam
    global normal
    global pclass
    infile = open("/Users/newuser/Downloads/spam_detection/train_email.txt", 'r')
    for line in infile.readlines():
        label=line.strip("\n").split(' ')[0]
        if int(label)==0:
            pclass[0] += 1
        else:
            pclass[1] += 1
        for word in line.strip("\n").split(' ')[1:]:
            newword,count=word.split(':')
            if int(label)==0:
                if newword not in spam:
                    spam[newword] = 1.0
            else:
                if newword not in normal :
                    normal[newword] = 1.0
    for word in worddict:
        if word not in spam:
            spam[word]=1.0
        if word not in normal:
            normal[word]=1.0
    spamsize=sum(spam.itervalues())
    normalsize=sum(normal.itervalues())
    print(normalsize)
    for word in spam:
        spam[word]=float(spam[word])/spamsize*3000
    for word in normal:
        normal[word]=float(normal[word])/normalsize*3000

def classifyspam():
    testfile = open("/Users/newuser/Downloads/spam_detection/test_email.txt", 'r')
    global pclass
    global worddict,spam,normal
    spamfreq=float(pclass[0])/float(pclass[0]+pclass[1])
    normalfreq=1-spamfreq
    spamword,normalword=1.0,1.0
    assignedlabel=[]
    label=[]
    for line in testfile.readlines():
        label.append(int(line.strip("\n").split(' ')[0]))
        for word in line.strip("\n").split(' ')[1:]:
            newword,count=word.split(':')
            if newword not in worddict:
                continue
            spamword*=spam[newword]
            normalword*=normal[newword]
        if spamfreq*spamword>=normalword*normalfreq:
            assignedlabel.append(0)
        else:
            assignedlabel.append(1)
        # print(spamword)
        # print(normalword)
        spamword=1.0
        normalword=1.0
    correct=0
    for i in xrange(len(assignedlabel)):
        if assignedlabel[i]==label[i]:
            correct+=1
    print(float(correct)/len(assignedlabel))
def trainAndTune(self, featuremap, featurelabel):
    count={}
    for i in xrange(featuremap.__len__()):
        for j in xrange(featuremap[i].__len__()):
            count[featurelabel[i]].insert(j,count[featurelabel[i]][j]+featuremap[i][j])


if __name__ == "__main__":
    worddict=[]
    numberclass=np.array
    spam={}
    normal={}
    pclass=[0,0]
    # builddictionary()
    # buildconditional()
    # classifyspam()

    bayesmatrix=[]
    parsetrainingdata()
    classifydigit()

    #
    # oddsRatioMap(8,3)
    # oddsRatioMap(9,7)
    # oddsRatioMap(5,3)
    # oddsRatioMap(9,4)

# # def classify(self, testData):
# #
# #
# #     """
# #     Classify the data based on the posterior distribution over labels.
# #
# #     You shouldn't modify this method.
# #     """
# #     guesses = []
# #     self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
# #     # for datum in testData:
# #     #   posterior = self.calculateLogJointProbabilities(datum)
# #     #   guesses.append(posterior.argMax())
# #     #   self.posteriors.append(posterior)
#
#   def calculateLogJointProbabilities(self, datum):
#     # """
#     # Returns the log-joint distribution over legal labels and the datum.
#     # Each log-probability should be stored in the log-joint counter, e.g.
#     # logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
#     #
#     # To get the list of all possible features or labels, use self.features and
#     # self.legalLabels.
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
