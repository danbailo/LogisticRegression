from core import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import glob

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True

if __name__=="__main__":
	dir_cat = '../input/train/cat/*.png'
	dir_noncat = '../input/train/noncat/*.png'
	
	X = []
	Y = []
	for cat in glob.glob(dir_cat):
		img = np.asarray(Image.open(cat))
		img = np.reshape(img, -1)
		X.append(img)
		Y.append(1)

	#-1 multiplica pelo grid(x1,x2,x3) da imagem, "desempacota" e multiplica - 64x64x3 = 12288
	for noncat in glob.glob(dir_noncat):
		img = np.asarray(Image.open(noncat))
		img = np.reshape(img, -1)
		X.append(img)
		Y.append(0) 	

	X = np.asarray(X)
	Y = np.asarray(Y)
	m = X.shape[0]
	lr = 0.0001

	X = X/255
	X = np.insert(X, obj=0, values=1, axis=1)
	Y = np.expand_dims(Y, axis=1)
	Thetas = np.zeros((X.shape[1],1))
	
	print(X.shape)
	print(Y.shape)

	logistic_regression = LogisticRegression(
		X = X,
		Y = Y,
		lr = lr,
		m = m,
		epochs = 1500,
		Thetas = Thetas
	)

	logistic_regression.split_data()
	X_train, Y_train = logistic_regression.X_train, logistic_regression.Y_train
	X_validate, Y_validate = logistic_regression.X_validate, logistic_regression.Y_validate

	loss_train, acc_train = logistic_regression.fit()
	# loss_validate, acc_validate = logistic_regression.predict()

	plt.figure()
	plt.plot(loss_train, label=fr"$leaning rate = {lr}$")
	# plt.plot(loss_validate, label=fr"$leaning rate = {lr}$")
	plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
	plt.xlabel('epochs')
	plt.ylabel(r'J($\theta$)')	

	plt.figure()
	plt.plot(acc_train)
	# plt.plot(acc_validate)
	plt.xlabel('epochs')
	plt.ylabel(r'acc')		
	
	plt.show()
