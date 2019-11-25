import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import glob
from tqdm import trange

random.seed(1)

class LogisticRegression:
	def __init__(self, lr, epochs, activation = "sigmoid", name_fig = None):
		self.lr = lr
		self.epochs = epochs
		self.activation = activation
		self.name_fig = name_fig

		self.__loss_train = []
		self.__acc_train = []
		self.__loss_validate = []
		self.__acc_validate = []

	#OS DADOS DISPOSTOS, RELACIONADO AS CLASSES, PRECISAM ESTAR BALANCEADOS?
	def split_data(self, X, Y, ratio):
		smaller = min(len(Y[Y==0]), len(Y[Y==1]))
		ratio_data = int(smaller*ratio)
		X_train = []
		Y_train = []
		X_validate = []
		Y_validate = []
		indexes = []

		values = list(zip(X, Y))

		#PORQUE MELHOR CASO EU EMBARALHE OS DADOS?
		random.shuffle(values)

		# for i,j in values:
		# 	print(i[:3],j)

		data = {key:values[key] for key in range(len(values))}

		for k,v in data.items():
			if v[1][0]==1 and len(X_train) < (smaller - ratio_data) and k not in indexes: #2 class, cat and noncat
				X_train.append(v[0])
				Y_train.append(v[1][0])
				indexes.append(k)
			if v[1][0]==0 and len(X_train) < (smaller - ratio_data)*2 and k not in indexes: #*2 pq ja tem metade disso na lista
				X_train.append(v[0])
				Y_train.append(v[1][0])
				indexes.append(k)

		for k,v in data.items():
			if v[1][0]==1 and len(X_validate) < ratio_data and k not in indexes: #2 class, cat and noncat
				X_validate.append(v[0])
				Y_validate.append(v[1][0])
				indexes.append(k)
			if v[1][0]==0 and len(X_validate) < ratio_data*2 and k not in indexes: #*2 pq ja tem metade disso na lista
				X_validate.append(v[0])
				Y_validate.append(v[1][0])
				indexes.append(k)

		X_train = np.asarray(X_train)
		X_train = np.insert(X_train, obj=0, values=1, axis=1)		
		Y_train = np.asarray(Y_train)
		Y_train = np.expand_dims(Y_train, axis=1)

		X_validate = np.asarray(X_validate)
		X_validate = np.insert(X_validate, obj=0, values=1, axis=1)
		Y_validate = np.asarray(Y_validate)
		Y_validate = np.expand_dims(Y_validate, axis=1)		
	
		return X_train, X_validate, Y_train, Y_validate

	def g(self, Z):
		if self.activation == "sigmoid":
			return 1 / (1 + np.exp(-Z))
		if self.activation == "relu":
			return Z * (Z > 0)

	def cost(self, Y_predicted, Y, m):		
		if self.activation == "sigmoid":
			return (1 / m) * np.sum(-Y * np.log(Y_predicted) - (1 - Y) * (np.log(1 - Y_predicted)))
		if self.activation == "relu":
			return np.sqrt(np.mean((Y - Y_predicted) ** 2))
			
	def fit(self, m, X_train, Y_train):
		Z = X_train.dot(self.Thetas)
		Y_predicted = self.g(Z)
		E = Y_predicted - Y_train
		self.Thetas = self.Thetas - (self.lr * (X_train.T.dot(E))) 
		self.__loss_train.append(self.cost(Y_predicted, Y_train, m))
		self.__acc_train.append(np.sum((Y_predicted >= 0.5) == Y_train) / m)

	def predict(self, m, X_validate, Y_validate):
		Z = X_validate.dot(self.Thetas)
		self.Y_predicted = self.g(Z)
		self.__loss_validate.append(self.cost(self.Y_predicted, Y_validate, m))
		self.__acc_validate.append(np.sum((self.Y_predicted >= 0.5) == Y_validate) / m)		
	
	def train(self, X_train, X_validate, Y_train, Y_validate):
		self.Thetas = np.zeros((X_train.shape[1],1))
		m_train = X_train.shape[0]
		m_validate = X_validate.shape[0]
		for _ in trange(self.epochs):
			self.fit(m_train, X_train, Y_train)
			self.predict(m_validate, X_validate, Y_validate)
		return self.__loss_train, self.__acc_train, self.__loss_validate, self.__acc_validate