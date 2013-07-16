from myHog import hog
import cPickle as pickle
from sklearn import svm
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage import transform
import numpy
filename = "saveLinearSVM1.pkl"
with open(filename,'r') as f:
	LinearClf1 = pickle.load(f)
images = []
for i in range(61):
	test = mpimg.imread("test/"+"test ("+ str(i+1) + ").jpg")
	test = transform.resize(test,numpy.array([64,64]))
	test = rgb2gray(test)
	test = hog(test).ravel()
	images.append(test)
print "...Load OK!"
for i in range(len(images)):
	print i+1," : ", LinearClf1.predict(images[i])
'''
test1 = mpimg.imread("test1.jpg")
test1 = transform.resize(test1,numpy.array([64,64]))
test1 = rgb2gray(test1)
test1 = hog(test1).ravel()
#test1 = hog(test1,pixels_per_cell=(8, 8), cells_per_block=(2, 2)).ravel()
test0 = mpimg.imread("test0.jpg")
test0 = transform.resize(test0,numpy.array([64,64]))
test0 = rgb2gray(test0)
#test0 = hog(test0,pixels_per_cell=(8, 8), cells_per_block=(2, 2)).ravel()
test0 = hog(test0).ravel()
print "Test Picture : Human"
print "Linear"
print LinearClf.predict(test1)
print "Rbf"
print RbfClf.predict(test1)
print "Linear1"
print LinearClf1.predict(test1)
print "Test Picture : Non-Human"
print "Linear"
print LinearClf.predict(test0)
print "Rbf"
print RbfClf.predict(test0)
print "Linear1"
print LinearClf1.predict(test0)
'''
