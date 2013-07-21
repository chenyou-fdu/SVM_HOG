__author__ = 'sylvia'

from myHog import hog
import cPickle as pickle
from sklearn import svm
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage import transform
import numpy
filename = "savedLinearSVM19999.pkl"
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

f = open('test/lables.txt')
data = f.readlines()
f.close
for i in range(len(data)):
	data[i] = int(data[i][3])
data = data[0:61]

p = []
correct = 0
for i in range(len(images)):
    p.append(LinearClf1.predict(images[i]))
    if p[i] == data[i]:
        correct+=1
    else:
        print i+1," : ", p[i]
print  correct