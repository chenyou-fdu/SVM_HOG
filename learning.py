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
for i in range(15000):
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
#	result = hog(img,pixels_per_cell=(8, 8), cells_per_block=(2, 2))
	result = (result.ravel()).tolist()
	#print len(result)
	#results.append(result)
	imgs[i] = result
f = open('lables.txt')
data = f.readlines()
f.close
for i in range(len(data)):
	data[i] = int(data[i][10])
data = data[0:15000]
LinearClf1 = svm.LinearSVC()
#LinearClf = svm.SVC(kernel='linear')
#RbfClf = svm.SVC(kernel='rbf')
#LinearClf.fit(results,data)
LinearClf1.fit(imgs,data)
#RbfClf.fit(results,data)
#filename1 = "saveLinearSVM.pkl"
#filename2 = "saveRbfSVM.pkl"
filename3 = "saveLinearSVM1.pkl"
#with open(filename1,'w') as f:
#	pname1 = pickle.dump(LinearClf,f)
#with open(filename2,'w') as f1:
#	pname2 = pickle.dump(RbfClf,f1)
with open(filename3,'w') as f2:
	pname3 = pickle.dump(LinearClf1,f2)
print "DONE!"
