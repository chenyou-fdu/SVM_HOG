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

imgs = []
#results = []
for i in range(20000,29968):
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
	# if count % 100 == 0:
	# 	print imgs[i]
	count += 1
	#print i
	img = mpimg.imread(imgs[i])
	img = transform.resize(img, numpy.array([64,64]))
	img = rgb2gray(img)
	result = hog(img)

	result = (result.ravel()).tolist()
	#results.append(result)
	imgs[i] = result
print "...Load OK!"


f = open('lables.txt')
data = f.readlines()
f.close
for i in range(len(data)):
	data[i] = int(data[i][10])
data = data[20000:29968]

p = []
correct = 0
for i in range(len(imgs)):
    p.append(LinearClf1.predict(imgs[i]))
    if p[i] == data[i]:
        correct+=1
    else:
        print i+1," : ", p[i]
print  correct