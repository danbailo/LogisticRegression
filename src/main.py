from core import LogisticRegression
from utils import save_img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True

if __name__=="__main__":
	dir_cat = '../data/train/cat/*.png'
	dir_noncat = '../data/train/noncat/*.png'
	
	X = []
	Y = []

	name_fig = []
	for cat in glob.glob(dir_cat):
		name_fig.append(cat.split('/')[-1])
		img = np.asarray(Image.open(cat))
		img = np.reshape(img, -1)
		X.append(img)
		Y.append(1)

	for noncat in glob.glob(dir_noncat):
		name_fig.append(noncat.split('/')[-1])
		img = np.asarray(Image.open(noncat))
		img = np.reshape(img, -1)
		X.append(img)
		Y.append(0) 	

	X = np.asarray(X)
	Y = np.asarray(Y)
	lr = 0.00001
	epochs = 1000

	X = X/255
	X = np.insert(X, obj=0, values=1, axis=1)
	Y = np.expand_dims(Y, axis=1)
	
	logistic_regression = LogisticRegression(
		X = X,
		Y = Y,
		lr = lr,
		epochs = epochs,
		activation = "relu",
		name_fig = name_fig
	)

	X_training, X_validation, Y_training, Y_validation = logistic_regression.split_data()

	loss_training, acc_training, loss_validation, acc_validation =\
	logistic_regression.train(
		X_training, X_validation, 
		Y_training, Y_validation
	)

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
