__author__ = 'Wong Sylvia'

from skimage import data, io, filter
from skimage.color import rgb2gray
from skimage import data
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
import numpy
from myHog import hog
import cPickle as pickle
imgs = []
results = []
for i in range(2000):
	if i <= 98 and i > 8:
		ori_IMG = 'images/000' + str(i+1) + '.jpg'
		imgs.append(ori_IMG)
	elif i > 98 and i <= 998:
		ori_IMG = 'images/00' + str(i+1) + '.jpg'
		imgs.append(ori_IMG)
	elif i > 998 :
		ori_IMG = 'images/0' + str(i+1) + '.jpg'
		imgs.append(ori_IMG)
	else:
		ori_IMG = 'images/0000' + str(i+1) + '.jpg'
		imgs.append(ori_IMG)

for i in imgs:
	# print i
	img = mpimg.imread(i)
	img = transform.resize(img, numpy.array([64,64]))
	img = rgb2gray(img)
	result = hog(img)
	result = (result.ravel()).tolist()
	results.append(result)
f = open('lables.txt')
data = f.readlines()
f.close
for i in range(len(data)):
	data[i] = int(data[i][10])
data = data[0:2000]
clf = svm.LinearSVC()
clf.fit(results,data)
filename = "savedLinearSVM2000.pkl"
with open(filename,'w') as f:
	pname = pickle.dump(clf,f)
print "DONE!"