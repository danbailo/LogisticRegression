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
	def __init__(self, lr, epochs, activation = "sigmoid", name_fig = None):
		# self.X_training = X_training
		# self.Y_training = Y_training
		# self.X_validation = X_validation
		# self.Y_validation = Y_validation
		self.lr = lr
		self.epochs = epochs
		self.activation = activation
		self.name_fig = name_fig

		self.__loss_training = []
		self.__acc_training = []
		self.__loss_validation = []
		self.__acc_validation = []

	#OS DADOS DISPOSTOS, RELACIONADO AS CLASSES, PRECISAM ESTAR BALANCEADOS?
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
		self.Y_predicted = self.g(Z)
		self.__loss_validation.append(self.cost(self.Y_predicted, Y_validation, m))
		self.__acc_validation.append(np.sum((self.Y_predicted >= 0.5) == Y_validation) / m)		
	
	def train(self, X_training, X_validation, Y_training, Y_validation):
		self.Thetas = np.zeros((X_training.shape[1],1))
		m_training = X_training.shape[0]
		m_validation = X_validation.shape[0]
		for _ in trange(self.epochs):
			self.fit(m_training, X_training, Y_training)
			self.predict(m_validation, X_validation, Y_validation)
		return self.__loss_training, self.__acc_training, self.__loss_validation, self.__acc_validation