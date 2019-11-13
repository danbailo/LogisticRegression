from core import LogisticRegression
from utils import get_data, save_img
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True

if __name__== "__main__":
	dir_cat_train = '../data/train/cat/*.png'
	dir_noncat_train = '../data/train/noncat/*.png'

	dir_cat_validate = '../data/validate/cat/*.png'
	dir_noncat_validate = '../data/validate/noncat/*.png'
	
	data_cat_train = get_data(dir_cat_train, "cat")
	data_noncat_train = get_data(dir_noncat_train, "noncat")

	data_cat_validate = get_data(dir_cat_validate, "cat")	
	data_noncat_validate = get_data(dir_noncat_validate, "noncat")	

	X_training = data_cat_train[0] + data_noncat_train[0]
	Y_training = data_cat_train[1] + data_noncat_train[1]
	
	X_validation = data_cat_validate[0] + data_noncat_validate[0]
	Y_validation = data_cat_validate[1] + data_noncat_validate[1]


	X_training = np.asarray(X_training)
	X_validation = np.asarray(X_validation)
	Y_training = np.asarray(Y_training)
	Y_validation = np.asarray(Y_validation)
	lr = 0.000008
	epochs = 5000

	#normalize data
	X_training = X_training/255
	X_validation = X_validation/255
	
	X_training = np.insert(X_training, obj=0, values=1, axis=1)
	X_validation = np.insert(X_validation, obj=0, values=1, axis=1)
	Y_training = np.expand_dims(Y_training, axis=1)
	Y_validation = np.expand_dims(Y_validation, axis=1)

	logistic_regression = LogisticRegression(	
		lr = lr,
		epochs = epochs,
		activation = "sigmoid",
	)

	loss_training, acc_training, loss_validation, acc_validation =\
	logistic_regression.train(
		X_training = X_training,
		Y_training = Y_training,
		X_validation = X_validation,
		Y_validation = Y_validation
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
