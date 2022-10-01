import numpy as np
import scipy as sp
#from scipy.sepcial import softmax

def calcMI(pxy):
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	return np.sum(pxy*np.log(pxy/px[:,None]/py[None,:]))
'''
def calcLogProj(logmp):
	ms_logmp = logmp - np.amax(logmp)
	return -np.log(softmax(ms_logmp))

def calcLogWeightedProj(logmp,weight):
	ms_logmp = logmp - np.amax(logmp)
	return -np.log(softmax(weight * ms_logmp))
'''