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
	lr = 0.0001

	X = X/255
	X = np.insert(X, obj=0, values=1, axis=1)
	Y = np.expand_dims(Y, axis=1)
	
	logistic_regression = LogisticRegression(
		X = X,
		Y = Y,
		lr = lr,
		epochs = 10000
	)

	logistic_regression.split_data()
	X_training, Y_training = logistic_regression.X_training, logistic_regression.Y_training
	X_validation, Y_validation = logistic_regression.X_validation, logistic_regression.Y_validation

	loss_training, acc_training = logistic_regression.fit()
	loss_validation, acc_validation = logistic_regression.predict()

	plt.figure()
	plt.plot(loss_training, label="training")
	plt.plot(loss_validation, label="validation")
	plt.xlabel('epochs')
	plt.ylabel(r'J($\theta$)')	
	plt.legend()

	plt.figure()
	plt.plot(acc_training, label="training")
	plt.plot(acc_validation, label="validation")
	plt.yticks(np.arange(0.0, 1.1, 0.1))
	plt.xlabel('epochs')
	plt.ylabel(r'acc')		
	plt.legend()
	
	plt.show()
