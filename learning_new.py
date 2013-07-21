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
#results = []
for i in range(19999):
	if i <= 98 and i > 8:
		ori_IMG = 'images/000' + str(i+1) + '.jpg'
		imgs.append(ori_IMG)
	elif i > 98 and i <= 998:
		ori_IMG = 'images/00' + str(i+1) + '.jpg'
		imgs.append(ori_IMG)
	elif i > 998 and i <= 9998:
		ori_IMG = 'images/0' + str(i+1) + '.jpg'
		imgs.append(ori_IMG)
	elif i > 9998:
		ori_IMG = 'images/'+str(i+1) + '.jpg'
		imgs.append(ori_IMG)
	else:
		ori_IMG = 'images/0000' + str(i+1) + '.jpg'
		imgs.append(ori_IMG)
count = 0
for i in range(len(imgs)):
	if count % 100 == 0:
		print imgs[i]
	count += 1
	#print i
	img = mpimg.imread(imgs[i])
	img = transform.resize(img, numpy.array([64,64]))
	img = rgb2gray(img)
	result = hog(img)

	result = (result.ravel()).tolist()
	#results.append(result)
	imgs[i] = result

f = open('lables.txt')
data = f.readlines()
f.close
for i in range(len(data)):
	data[i] = int(data[i][10])
data = data[0:19999]

LinearClf1 = svm.LinearSVC()
LinearClf1.fit(imgs,data)
filename3 = "savedLinearSVM19999.pkl"

with open(filename3,'w') as f2:
	pname3 = pickle.dump(LinearClf1,f2)
print "DONE!"
