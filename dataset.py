import numpy as np
import sys

def synMy():
	gbl_pycx = np.array([[0.90,0.08,0.40],[0.025,0.82,0.05],[0.075,0.10,0.55]])
	gbl_px = np.ones(3,)/3
	gbl_pxy = (gbl_pycx*gbl_px[None,:]).T
	return {'pxy':gbl_pxy,'nx':3, 'ny':3}
