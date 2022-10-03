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
def calcEnt(pz):
	return -np.sum(np.log(pz)*pz)


def initPzcx(use_deterministic=0,smooth_val=1e-4,nz=None,nx=None,seed=None):
	rs = np.random.default_rng(seed)
	pzcx = np.zeros((nz,nx))
	if use_deterministic == 1:
		shuffle_zx = rs.permutation(nz)
		for idx, item in enumerate(shuffle_zx):
			pzcx[item,idx] = 1
		shuffle_rest = rs.integers(nz,size=(nx-nz))
		for nn in range(nx-nz):
			pzcx[shuffle_rest[nn],nz+nn]= 1 
		# smoothing 
		pzcx+= 1e-4
	else:
		pzcx= rs.random((nz,nx))
	return pzcx / np.sum(pzcx,axis=0) # normalization

def priorInfo(pxy):
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy / py[None,:]
	pycx = (pxy / px[:,None]).T
	return (px,py,pxcy,pycx)