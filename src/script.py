import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import glob

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = True

def logistic_regression(NUM_IT, alpha, m, X, Y, theta):
    J = np.zeros(NUM_IT)
    acc = np.zeros(NUM_IT)    
    for i in range(NUM_IT):
        z = np.dot(X,theta)
        H_theta = np.divide(1,1+np.exp(-z)) #nessa parte, a gente submete ele a teste depois de treinado.
        E = H_theta - Y
        J[i] = (1/m)*np.sum(-Y*np.log(H_theta) - (1-Y)*(np.log(1-H_theta)))
        theta = theta - (alpha*(np.dot(X.T,E))) 
        acc[i] = np.sum((H_theta >= 0.5) == Y)/m
    return J, H_theta, acc

if __name__=="__main__":
	dir_cat = '../inputs/data/train/cat/*.png'
	dir_noncat = '../inputs/data/train/noncat/*.png'
	
	X = [] #dados de treino
	Y = [] #dados de treino
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
	
	X = X/255
	X = np.insert(X, obj=0, values=1, axis=1)
	Y = np.expand_dims(Y, axis=1)
	theta = np.zeros((X.shape[1],1))
	
	theta = np.zeros((X.shape[1],1))
	# alphas=[,0.000001]

	plt.figure()
	for alpha in [0.01,0.001,0.0001,0.00001, 0.000001]:
		J, H_theta, _ = logistic_regression(100, alpha, X.shape[0], X, Y, theta)
		plt.plot(J, label=fr"$\alpha = {alpha}$")
		plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
		plt.xlabel('Iterações')
		plt.ylabel(r'J($\theta$)')
	plt.savefig('../imgs/all_alphas.pdf',bbox_inches='tight', transparent=True)
	
	plt.figure()
	for alpha in [0.01,0.001,0.0001,0.00001, 0.000001]:
		J, H_theta, _ = logistic_regression(100, alpha, X.shape[0], X, Y, theta)
		if max(J) > 20: continue
		plt.plot(J, label=fr"$\alpha = {alpha}$")
		plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
		plt.xlabel('Iterações')
		plt.ylabel(r'J($\theta$)')	
	plt.savefig('../imgs/out_overflow_alphas.pdf',bbox_inches='tight', transparent=True)

	plt.figure()
	for alpha in [0.01,0.001,0.0001,0.00001, 0.000001]:
		J, H_theta, _ = logistic_regression(2000, alpha, X.shape[0], X, Y, theta)
		if max(J) > 5: continue
		plt.plot(J, label=fr"$\alpha = {alpha}$")
		plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
		plt.xlabel('Iterações')
		plt.ylabel(r'J($\theta$)')	
	plt.savefig('../imgs/out_overflow2_alphas.pdf',bbox_inches='tight', transparent=True)

	plt.figure()
	alpha =0.00001
	J, H_theta, _ = logistic_regression(2000, alpha, X.shape[0], X, Y, theta)
	plt.plot(J, label=fr"$\alpha = {alpha}$")
	plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
	plt.xlabel('Iterações')
	plt.ylabel(r'J($\theta$)')	
	plt.savefig('../imgs/final.pdf',bbox_inches='tight', transparent=True)
	
	plt.show()
