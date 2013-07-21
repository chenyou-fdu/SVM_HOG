__author__ = 'Wong Sylvia'

from myHog import hog
import cPickle as pickle
from sklearn import svm
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage import transform
import numpy

filename = "savedlinearSVM100.pkl"
with open(filename, 'r') as f:
    clf = pickle.load(f)
test1 = mpimg.imread("test1.jpg")
test1 = transform.resize(test1, numpy.array([64, 64]))
test1 = rgb2gray(test1)
test1 = hog(test1).ravel()
test0 = mpimg.imread("test0.jpg")
test0 = transform.resize(test0, numpy.array([64, 64]))
test0 = rgb2gray(test0)
test0 = hog(test0).ravel()
print "Test Picture : Human"
print clf.predict(test1)
print "Test Picture : Non-Human"
print clf.predict(test0)

