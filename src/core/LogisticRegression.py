import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import glob
from tqdm import trange

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True

np.random.seed(1)

class LogisticRegression:
	def __init__(self, X, Y, lr, epochs, activation = "sigmoid"):
		self.X = X
		self.Y = Y
		self.lr = lr
		self.epochs = epochs
		self.activation = activation

		self.__loss_training = []
		self.__acc_training = []
		self.__loss_validation = []
		self.__acc_validation = []

	def split_data(self):		
		smaller = min(len(self.Y[self.Y==0]), len(self.Y[self.Y==1]))

		indexes = []
		X_training, Y_training = [], []
		X_validation, Y_validation = [], []

		state = 1
		while True:
			index = np.random.choice(np.arange(0,len(self.X)))
			while index in indexes:
				index = np.random.choice(np.arange(0,len(self.X)))
			indexes.append(index)
			if state == 1:
				X_training.append(self.X[index])
				Y_training.append(self.Y[index][0])
				if len(X_training) == smaller:
					state = 2
			elif state == 2:
				X_validation.append(self.X[index])
				Y_validation.append(self.Y[index][0])
				if len(X_validation) == int(smaller*0.5):
					state = 3
			else: break
		X_training, X_validation = np.asarray(X_training), np.asarray(X_validation)
		Y_training, Y_validation = np.asarray(Y_training), np.asarray(Y_validation)

		X_training, Y_training = np.insert(X_training, obj=0, values=1, axis=1), np.expand_dims(Y_training, axis=1)
		X_validation, Y_validation = np.insert(X_validation, obj=0, values=1, axis=1), np.expand_dims(Y_validation, axis=1)
		return X_training, X_validation, Y_training, Y_validation

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
			
	def fit(self, m, X_training, Y_training):
		Z = X_training.dot(self.Thetas)
		Y_predicted = self.g(Z)
		E = Y_predicted - Y_training
		self.Thetas = self.Thetas - (self.lr * (X_training.T.dot(E))) 
		self.__loss_training.append(self.cost(Y_predicted, Y_training, m))
		self.__acc_training.append(np.sum((Y_predicted >= 0.5) == Y_training) / m)

	def predict(self, m, X_validation, Y_validation):
		Z = X_validation.dot(self.Thetas)
		Y_predicted = self.g(Z)
		self.__loss_validation.append(self.cost(Y_predicted, Y_validation, m))
		self.__acc_validation.append(np.sum((Y_predicted >= 0.5) == Y_validation) / m)		
	
	def train(self, X_training, X_validation, Y_training, Y_validation):
		self.Thetas = np.zeros((X_training.shape[1],1))
		m_training = X_training.shape[0]
		m_validation = X_validation.shape[0]
		for _ in trange(self.epochs):
			self.fit(m_training, X_training, Y_training)
			self.predict(m_validation, X_validation, Y_validation)
		return self.__loss_training, self.__acc_training, self.__loss_validation, self.__acc_validation