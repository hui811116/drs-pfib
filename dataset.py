import numpy as np
import sys
import scipy.io as sio

def getDataset(name):
	if name == "syn":
		return synMy()
	elif name == "heartfail":
		return uciHeartFail()
	elif name == "nonunifsyn":
		return synNonUnifX()
	else:
		sys.exit("undefined dataset {:}".format(name))


def synMy():
	gbl_pycx = np.array([[0.90,0.08,0.40],[0.025,0.82,0.05],[0.075,0.10,0.55]])
	gbl_px = np.ones(3,)/3
	gbl_pxy = (gbl_pycx*gbl_px[None,:]).T
	return {'pxy':gbl_pxy,'nx':3, 'ny':3}

def synNonUnifX():
	gbl_pycx = np.array([[0.90,0.08,0.40],[0.025,0.82,0.05],[0.075,0.10,0.55]])
	gbl_px = np.array([0.10,0.30,0.60])
	gbl_pxy = (gbl_pycx * gbl_px[None,:]).T
	return {"pxy":gbl_pxy,'nx':3,"ny":3}

def uciHeart():
	gbl_pxcy = np.array([[0.1926, 0.2685, 0.1516, 0.2004],
						 [0.0574, 0.3796, 0.0665, 0.4159],
						 [0.0034, 0.0463, 0.0133, 0.0366],
						 [0.0034, 0.1574, 0.0027, 0.0711],
						 [0.6115, 0.0833, 0.5027, 0.2004],
						 [0.0439, 0.0093, 0.0878, 0.0366],
						 [0.0709, 0.0463, 0.1622, 0.0194],
						 [0.0169, 0.0093, 0.0133, 0.0194]])
	gbl_py = np.array([0.2379, 0.0868, 0.3023, 0.3730])
	gbl_pxy =gbl_pxcy * gbl_py[None,:]
	return {'pxy':gbl_pxy,'nx':8,'ny':4}

def uciHeartFail():
	with open('heartfail_pxy.mat','rb') as fid:
		tmp_dict = {}
		raw_datamat = sio.loadmat(fid)
	gbl_pxy = raw_datamat['heartfail_pxy']
	(nx,ny) = gbl_pxy.shape
	return {'pxy':gbl_pxy,'nx':nx,'ny':ny}

