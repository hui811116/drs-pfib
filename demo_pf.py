import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import scipy
import time
import pickle
import pprint
import argparse
from scipy.io import savemat
import algorithm as alg
import dataset as dt
import utils as ut


d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("-beta",type=float,help='the PF beta',default=5.0)
parser.add_argument('-ntime',type=int,help='run how many times per beta',default=25)
parser.add_argument('-penalty',type=float,help='penalty coefficient',default=16.0)
parser.add_argument('-relax',type=float,help='Relaxation parameter for DRS',default=1.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('-sinit',type=float,help='initial step size',default=1e-4)
parser.add_argument('-sscale',type=float,help='Scaling of step size',default=0.25)
parser.add_argument('-maxiter',type=int,help='Maximum number of iterations',default=40000)
parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=None)


args = parser.parse_args()
argdict = vars(args)
data = dt.synMy()

def runAlg(nz,beta,thres,maxiter,**kwargs):
	algout = alg.drsPF(data['pxy'],nz,beta,thres,maxiter,**kwargs)
	return algout

for nn in range(args.ntime):
	output = runAlg(data['ny'],**argdict)
	print('{nidx:<3} run: IZX={IZX:>10.4f}, IZY={IZY:>10.4f}, niter={niter:>10}, converge:{conv:>5}'.format(**{'nidx':nn,**output}))