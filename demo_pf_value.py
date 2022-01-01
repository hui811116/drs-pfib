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
import copy


d_base = os.getcwd()

parser = argparse.ArgumentParser()
#parser.add_argument("opt",type=str,choices=['ib','pf'],help="The objective to optimize")
parser.add_argument("-beta",type=float,help='the PF beta',default=5.0)
parser.add_argument('-ntime',type=int,help='run how many times per beta',default=20)
parser.add_argument('-penalty',type=float,help='penalty coefficient',default=1024.0)
parser.add_argument('-relax',type=float,help='Relaxation parameter for DRS',default=1.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('-sinit',type=float,help='initial step size',default=1e-3)
parser.add_argument('-sscale',type=float,help='Scaling of step size',default=0.25)
parser.add_argument('-maxiter',type=int,help='Maximum number of iterations',default=40000)
parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('-detinit',help='Start from a almost deterministic point',action='count',default=0)
parser.add_argument('-record',help='Record the value decrease',action='count',default=0)


args = parser.parse_args()
argdict = vars(args)

#data = dt.synMy()
#data = dt.uciHeart()
data = dt.uciHeartFail()
px = np.sum(data['pxy'],axis=1)
print('dataset: IXY={:>10.5f}, HX={:>10.5f}'.format(ut.calcMI(data['pxy']),-np.sum(px*np.log(px))) )

def runAlg(nz,beta,thres,maxiter,**kwargs):
	if kwargs['opt'] == "pf":
		algout = alg.drsPF(data['pxy'],nz,beta,thres,maxiter,**kwargs)
	elif kwargs['opt'] == "ib":
		algout = alg.drsIBType1(data['pxy'],nz,beta,thres,maxiter,**kwargs)
	return algout


nz = data['nx']
sinit = copy.copy(argdict['sinit'])
argdict['opt'] = "pf"
argdict['sinit'] = sinit
record_out = None
for nn in range(args.ntime):
	#print('\rProgress: beta={:.2f}, run={:>5}/{:>5}, nz={:>3}, ss_init={:8.2e}'.format(beta,nn,args.ntime,nz,argdict['sinit']),end='',flush=True)
	output = runAlg(nz,**argdict)
	if 'record' in output:
		record_out = output['record']
	print('{nidx:<3} run: nz={nz:>3}, IZX={IZX:>10.4f}, IZY={IZY:>10.4f}, niter={niter:>10}, converge:{conv:>5}'.format(**{'nidx':nn,'nz':nz,**output}))
	

if argdict['record']:
	plt.plot(np.arange(1,record_out.shape[0]+1),record_out-np.amin(record_out))
	plt.yscale('log')
	plt.show()

	# save mat
	savemat_name = 'heartfail_drspv_record_r_{}_c_{}'.format(args.relax,int(args.penalty),sinit)
	#savemat_name = 'heart_drspv_r_{}_c_{}_si_{:4.2e}'.format(args.relax,int(args.penalty),sinit)
	outmat_name = savemat_name.replace('.',"f") + '.mat'
	save_location = os.path.join(d_base,outmat_name)
	savemat(save_location,{'relax':args.relax,'penalty':args.penalty,'sinit':args.sinit,'val_record':record_out,})

	print('simulation complete, saving results to: {}'.format(save_location))