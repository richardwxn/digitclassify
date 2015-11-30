import numpy
from sklearn.metrics import confusion_matrix
import time
def load_data(xsize,ysize):
    train_labels = []
    with open('/Users/newuser/Downloads/digitdata/traininglabels', 'rb') as f:
        for i, line in enumerate(f):
            train_labels.append(int(line))
    train_labels = numpy.array(train_labels, dtype=int)

    train_x = numpy.zeros((train_labels.shape[0] * 28 * 28))
    with open('/Users/newuser/Downloads/digitdata/trainingimages', 'rb') as f:
        for i, line in enumerate(f):
            for j, char in enumerate(line.strip('\n')):
                if '+' == char or '#' == char:
                    train_x[i * 28 + j] = 1

    train_x = train_x.reshape((train_labels.shape[0], 28 * 28))
    new_train_x = numpy.zeros((train_labels.shape[0], (28/xsize) * (28/ysize)))
    cnt = 0
    # for i in xrange(0, 28*28,2):
    #     if (i+1) % 28 != 0 and i / 28 % 2 == 0:
    #         num=0
    #         for row in xrange(xsize):
    #             for col in xrange(ysize):
    #                 new_train_x[:, cnt] += pow(2,num)*train_x[:, i+28*row+col]
    #                 num+=1
    #         # new_train_x[:, cnt] = train_x[:, i] + train_x[:, i+1]*2 + train_x[:, i+28]*4 + train_x[:, i+29]*8
    #         cnt += 1
    for i in xrange(28/xsize):
        for j in xrange(28/ysize):
            num=0
            for row in xrange(xsize):
                for col in xrange(ysize):
                    new_train_x[:, cnt] += pow(2,num)*train_x[:, i*xsize*28+j*ysize+28*row+col]
                    num+=1
            # new_test_x[:, cnt] = test_x[:, i] + test_x[:, i+1]*2 + test_x[:, i+28]*4 + test_x[:, i+29]*8
            cnt += 1

    train_x = numpy.array(new_train_x, dtype=int)

    test_labels = []
    with open('/Users/newuser/Downloads/digitdata/testlabels', 'rb') as f:
        for i, line in enumerate(f):
            test_labels.append(int(line))
    test_labels = numpy.array(test_labels, dtype=int)

    test_x = numpy.zeros((test_labels.shape[0] * 28 * 28))
    with open('/Users/newuser/Downloads/digitdata/testimages', 'rb') as f:
        for i, line in enumerate(f):
            for j, char in enumerate(line.strip('\n')):
                if '+' == char or '#' == char:
                    test_x[i * 28 + j] = 1

    test_x = test_x.reshape((test_labels.shape[0], 28 * 28))
    new_test_x = numpy.zeros((test_labels.shape[0], (28/xsize) * (28/ysize)))
    cnt = 0
    # for i in xrange(0, 28 * 28, 4):
    #     # if (i+1) % 28 != 0 and (i+1) % 28 <25 and i / 28 % 2 == 0:
    for i in xrange(28/xsize):
        for j in xrange(28/ysize):
            num=0
            for row in xrange(xsize):
                for col in xrange(ysize):
                    # print(cnt)
                    new_test_x[:, cnt] += pow(2,num)*test_x[:, i*xsize*28+j*ysize+28*row+col]
                    num+=1
            # new_test_x[:, cnt] = test_x[:, i] + test_x[:, i+1]*2 + test_x[:, i+28]*4 + test_x[:, i+29]*8
            cnt += 1

    test_x = numpy.array(new_test_x, dtype=int)


    return train_x, train_labels, test_x, test_labels

class BayesClassifier(object):
    def __init__(self, xsize, ysize):
        self.bayesmatrix = None
        self.valuerange=pow(2,xsize*ysize)
        self.xsize=xsize
        self.ysize=ysize

    def fit(self, X, y):
        bayesmatrix = numpy.ones((10, self.valuerange, (28/self.xsize) * (28/self.ysize)), dtype=numpy.float64)
        for k in xrange(10):
            for i in xrange(self.valuerange):
                for j in xrange(X.shape[1]):
                    bayesmatrix[k, i, j] = numpy.sum(X[y==k, j]==i)
        numclass = numpy.zeros(10)
        for i in xrange(10):
            numclass[i] = numpy.sum(y==i) + 1
        bayesmatrix += 1.
        bayesmatrix /= numclass[:, numpy.newaxis, numpy.newaxis]
        self.bayesmatrix = bayesmatrix

    def predict(self, X):
        labels = []
        for i in xrange(X.shape[0]):
            hehe=reduce(lambda x, y: x+y, [numpy.sum(numpy.log(self.bayesmatrix[:, k, X[i, :]==k]), axis=1) for k in xrange(self.valuerange)])
            label=numpy.argmax(hehe)
            labels.append(label)
        return numpy.array(labels)


if "__main__" == __name__:
    size=numpy.array([4,4]).reshape((1,2))
    for combination in size:
        start_time=time.time()
        X, y, test_x, test_y = load_data(combination[0], combination[1])
        clf = BayesClassifier(combination[0],combination[1])
        clf.fit(X, y)
        pr = clf.predict(test_x)
        print(time.time()-start_time)
        print "Confusion Matrix"
        print confusion_matrix(test_y, pr)
        print "Accuracy"
        print numpy.sum(pr == test_y) / float(test_y.shape[0])