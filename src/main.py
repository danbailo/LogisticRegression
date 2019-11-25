from core import LogisticRegression
from utils import get_data, save_img
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True

if __name__== "__main__":
	dir_cat = '../data/train/cat/*.png'
	dir_noncat = '../data/train/noncat/*.png'

	X_cat, Y_cat = get_data(dir_cat, "cat")
	X_noncat, Y_noncat = get_data(dir_noncat, "noncat")

	X = X_cat + X_noncat
	Y = Y_cat + Y_noncat

	X = np.asarray(X)
	Y = np.asarray(Y)
	lr = 0.00001
	epochs = 3000

	#normalize data
	X = X/255
	
	X = np.insert(X, obj=0, values=1, axis=1)
	Y = np.expand_dims(Y, axis=1)

	print(X.shape)
	print(Y.shape)

	logistic_regression = LogisticRegression(	
		lr = lr,
		epochs = epochs,
		activation = "sigmoid",
	)
	X_train, X_validate, Y_train, Y_validate = logistic_regression.split_data(X, Y, 0.2)

	print(len(X_train))
	print(len(X_validate))

	print(len(Y_train))
	print(len(Y_validate))

	loss_train, acc_train, loss_validate, acc_validate =\
	logistic_regression.train(
		X_train = X_train,
		Y_train = Y_train,
		X_validate = X_validate,
		Y_validate = Y_validate
	)

	plt.figure()
	plt.plot(loss_train, label="train")
	plt.plot(loss_validate, label="validate")
	plt.xlabel('epochs')
	plt.ylabel(r'J($\theta$)')	
	plt.legend()

	plt.figure()
	plt.plot(acc_train, label="train")
	plt.plot(acc_validate, label="validate")
	plt.yticks(np.arange(0.0, 1.1, 0.1))
	plt.xlabel('epochs')
	plt.ylabel(r'acc')		
	plt.legend()
	
	plt.show()
