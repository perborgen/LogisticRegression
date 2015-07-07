import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel




df = pd.read_csv("data.csv", header=0)
df.columns = ["grade2","grade1","label"]
#print df[["grade2","grade1","label"]]
X = df[["grade1","grade2"]]
Y = df["label"].map(lambda x: float(x.rstrip(';')))


X = np.array(X)
Y = np.array(Y)
X = min_max_scaler.fit_transform(X)

#X = min_max_scaler.transform(X)



clf = LogisticRegression()
clf.fit(X,Y)
print clf.coef_
print clf.intercept_
print clf.get_params
print clf.predict([0.86980324,0.7849949])


pos = where(Y == 1)
neg = where(Y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
#show()

THETA = [0,0]

def predict(x):
	#z = x[0]*1.4613529092205131 + x[1]*2.3017789179131243
	#z = x[0]*1.00863245 + x[1]*1.1350556

	sigmoid = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return sigmoid

def Hypothesis(theta, x):
	z = 0
	for i in xrange(len(theta)):
		z += x[i]*theta[i]
	G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return G_of_Z 

def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	for i in xrange(m):
		xi = X[i]
		hi = Hypothesis(theta,xi)
#		print 'hi is ', hi
		if Y[i] == 1:
#			print 'Y[i] is ', Y[i]
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
#			print 'Y[i] is ', Y[i]
			error = (1-Y[i]) * math.log(1-hi)
#		print 'error is ', error
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	print 'cost is ', J 
	return J


def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in xrange(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothesis(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in xrange(len(theta)):
		CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta


def Logistic_Regression(X,Y,alpha,theta,num_iters):
	m = len(Y)
	for x in xrange(num_iters):
		new_theta = Gradient_Descent(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			Cost_Function(X,Y,theta,m)
			print 'theta ', theta
			print 'cost is ', Cost_Function(X,Y,theta,m)







#Logistic_Regression(X,Y,0.1,THETA,30000)

theta = [5.0593323921576783, 5.4413493367465211]
print 'our 1: ', round(Hypothesis(theta,[3.50517825,2.60395629]))
print 'their 1: ', clf.predict([3.50517825,2.60395629])

print 'our 2: ', round(Hypothesis(theta,[0.86980324,0.7849949]))
print 'their 2: ', clf.predict([0.86980324,0.7849949])

print 'our 3: ', round(Hypothesis(theta,[-0.2,-0.3]))
print 'their 3: ', clf.predict([-0.2,-0.3])

print 'our 4: ', round(Hypothesis(theta,[-0.1,-0.2]))
print 'their 4: ', clf.predict([-0.1,-0.2])

#skTHETA = [0.41964108,0.14270665]
#ourTHETA = [0.3966494659592947, -0.10868864187921315]

#print Cost_Function(X,Y,skTHETA,8)
#print Cost_Function(X,Y,ourTHETA,8)