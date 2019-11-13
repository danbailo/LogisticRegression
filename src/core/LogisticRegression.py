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
	def __init__(self, X, Y, lr, epochs):
		self.X = X
		self.Y = Y
		self.lr = lr
		self.epochs = epochs

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
				if len(X_validation) == smaller:
					state = 3
			else: break
		X_training, X_validation = np.asarray(X_training), np.asarray(X_validation)
		Y_training, Y_validation = np.asarray(Y_training), np.asarray(Y_validation)
		self.X_training, self.Y_training = np.insert(X_training, obj=0, values=1, axis=1), np.expand_dims(Y_training, axis=1)
		self.X_validation, self.Y_validation = np.split(self.X_training, 2)[0], np.split(self.Y_training, 2)[0]

	def fit(self):
		loss = []
		acc = []
		self.m = self.X_training.shape[0]
		self.Thetas = np.zeros((self.X_training.shape[1],1))
		for _ in trange(self.epochs):
			Z = self.X_training.dot(self.Thetas)
			H_theta = np.divide(1, 1+np.exp(-Z))
			E = H_theta - self.Y_training
			self.Thetas = self.Thetas - (self.lr * (self.X_training.T.dot(E))) 
			loss.append((1 / self.m) * np.sum(-self.Y_training * np.log(H_theta) - (1-self.Y_training) * (np.log(1 - H_theta))))
			acc.append(np.sum((H_theta >= 0.5) == self.Y_training) / self.m)
		return loss, acc

	def predict(self):
		loss = []
		acc = []
		self.m = self.X_validation.shape[0]
		self.Thetas = np.zeros((self.X_validation.shape[1],1))
		for _ in trange(self.epochs):
			Z = self.X_validation.dot(self.Thetas)
			H_theta = np.divide(1, 1+np.exp(-Z))
			# E = H_theta - self.Y_validation
			# self.Thetas = self.Thetas - (self.lr * (self.X_validation.T.dot(E))) 
			loss.append((1 / self.m) * np.sum(-self.Y_validation * np.log(H_theta) - (1-self.Y_validation) * (np.log(1 - H_theta))))
			acc.append(np.sum((H_theta >= 0.5) == self.Y_validation) / self.m)
		return loss, acc