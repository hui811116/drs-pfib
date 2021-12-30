import numpy as np

def calcMI(pxy):
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	return np.sum(pxy*np.log(pxy/px[:,None]/py[None,:]))
