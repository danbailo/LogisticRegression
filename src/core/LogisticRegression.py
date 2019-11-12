import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import glob
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True

np.random.seed(1)

class LogisticRegression:
	def __init__(self, X, Y, lr, m, epochs, Thetas):
		self.X = X
		self.Y = Y
		self.lr = lr
		self.m = m
		self.epochs = epochs
		self.Thetas = Thetas

	def split_data(self):		
		smaller = min(len(self.Y[self.Y==0]), len(self.Y[self.Y==1]))

		indexes = []
		X_train, Y_train = [], []
		X_validate, Y_validate = [], []

		state = 1
		while True:
			index = np.random.choice(np.arange(0,len(self.X)))
			while True:
				if index in indexes:
					index = np.random.choice(np.arange(0,len(self.X)))
				else: break			
			indexes.append(index)
			if state == 1:
				X_train.append(self.X[index])
				Y_train.append(self.Y[index][0])
				if len(X_train) == smaller:				
					state = 2
			elif state == 2:
				X_validate.append(self.X[index])
				Y_validate.append(self.Y[index][0])
				if len(X_validate) == smaller:
					state = 3
			else: break
		X_train, X_validate = np.asarray(X_train), np.asarray(X_validate)
		Y_train, Y_validate = np.asarray(Y_train), np.asarray(Y_validate)
		self.X_train, self.Y_train,  = np.insert(X_train, obj=0, values=1, axis=1), np.expand_dims(Y_train, axis=1)
		self.X_validate, self.Y_validate = np.insert(X_validate, obj=0, values=1, axis=1), np.expand_dims(Y_validate, axis=1)

	def fit(self):
		loss = []
		acc = []
		for _ in tqdm(range(self.epochs)):
			Z = self.X.dot(self.Thetas)
			H_theta = np.divide(1, 1+np.exp(-Z))
			E = H_theta - self.Y
			self.Thetas = self.Thetas - (self.lr * (self.X.T.dot(E))) 
			loss.append((1 / self.m) * np.sum(-self.Y * np.log(H_theta) - (1-self.Y) * (np.log(1 - H_theta))))
			acc.append(np.sum((H_theta >= 0.5) == self.Y) / self.m)
		return loss, acc

	def predict(self):
		loss = []
		acc = []
		for _ in tqdm(range(self.epochs)):
			Z = self.X.dot(self.Thetas)
			H_theta = np.divide(1, 1+np.exp(-Z))
			E = H_theta - self.Y
			self.Thetas = self.Thetas - (self.lr * (self.X.T.dot(E))) 
			loss.append((1 / self.m) * np.sum(-self.Y * np.log(H_theta) - (1-self.Y) * (np.log(1 - H_theta))))
			acc.append(np.sum((H_theta >= 0.5) == self.Y) / self.m)
		return loss, acc