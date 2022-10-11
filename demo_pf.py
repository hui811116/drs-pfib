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
#parser.add_argument("-beta",type=float,help='the PF beta',default=5.0)
parser.add_argument("method",choices=alg.supportedPfAlg())
parser.add_argument('--ntime',type=int,help='run how many times per beta',default=20)
parser.add_argument('--penalty',type=float,help='penalty coefficient',default=1024.0)
parser.add_argument('--relax',type=float,help='Relaxation parameter for DRS',default=1.0)
parser.add_argument('--thres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('--sinit',type=float,help='initial step size',default=1e-3)
parser.add_argument('--sscale',type=float,help='Scaling of step size',default=0.25)
parser.add_argument('--maxiter',type=int,help='Maximum number of iterations',default=40000)
parser.add_argument('--seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('--minbeta',type=float,help='the minimum beta',default=1.0)
parser.add_argument('--maxbeta',type=float,help='the maximum beta',default=20.0)
parser.add_argument('--numbeta',type=int,help='beta geometric space',default=20)
#parser.add_argument('-detinit',help='Start from a almost deterministic point',action='count',default=0)
parser.add_argument('--record',action="store_true",default=False,help='Record the value decrease')
parser.add_argument('--dataset',type=str,default="syn",help="dataset to run")



args = parser.parse_args()
argdict = vars(args)
print(argdict)

d_beta_range = np.geomspace(args.minbeta,args.maxbeta,num=args.numbeta)
#data = dt.uciHeart()
#data = dt.uciHeartFail()
data = dt.getDataset(args.dataset)

px =np.sum(data['pxy'],axis=1)

# algorithm selection
algrun = alg.getPFAlgorithm(args.method)

result_dict= {}
details_dict = {}
nz = data['nx'] + 1 # by theory condinality bound
res_array = np.zeros((args.numbeta*args.ntime*(nz-1)*2,8))   # beta, nz,conv,niter,IZX, IZY, entz,inittype
rcnt =0 
sinit = copy.copy(argdict['sinit'])
while nz >=2:
	argdict['sinit'] = sinit
	for randscheme in range(2):
		argdict['detinit'] = randscheme
		for beta in d_beta_range:
			conv_cnt = 0
			for nn in range(args.ntime):
				runtime_dict = {"convthres":args.thres,**argdict}
				output = algrun(data['pxy'],nz,beta,**runtime_dict)
				pz = np.sum(output['pzcx']*px[None,:],axis=1)
				entz = ut.calcEnt(pz)
				res_array[rcnt,:] = np.array([beta,nz,int(output['conv']),output['niter'],output['IZX'],output['IZY'],entz,randscheme])
				rcnt +=1
				conv_cnt += int(output['conv'])
				if output['conv']:
					izx_str = '{:.2f}'.format(output['IZX']) # FIXME: precision is designed here
					if not result_dict.get(izx_str,False):
						result_dict[izx_str] = np.Inf
						details_dict[izx_str] = {}
					if result_dict[izx_str] >output['IZY']:
						result_dict[izx_str] = output['IZY']
						details_dict[izx_str]['nz'] = nz
						details_dict[izx_str]['randinit'] = randscheme
						details_dict[izx_str]['beta'] = beta
						details_dict[izx_str]['nrun'] = nn 
			status_tex = 'beta,{:.2f},nz,{:},conv_rate,{:.6f},sinit,{:.4f},detinit,{:},ifc_items,{:}'.format(beta,nz,conv_cnt/args.ntime,argdict['sinit'],randscheme,len(details_dict))
			print("{:}".format(status_tex))
			#argdict['sinit'] *= 0.9
	nz -= 1

dataout = []
for k,v in result_dict.items():
	izx_num = float(k)
	dataout.append([izx_num,v])
datanpy = np.array(dataout)

mixy = ut.calcMI(data['pxy'])

# save mat
savemat_name = "pf_{:}_{:}_r{:}_c{:}_si{:4.2e}".format(args.dataset,args.method,args.relax,int(args.penalty),sinit)
outmat_name = savemat_name.replace('.',"")
repeat_cnt = 0
save_location = os.path.join(d_base,outmat_name+".mat")
while os.path.isfile(save_location):
	repeat_cnt += 1
	savemat_name = "pf_{:}_{:}_r{:}_c{:}_si{:4.2e}_{:}".format(args.dataset,args.method,args.relax,int(args.penalty),sinit,repeat_cnt)
	outmat_name = savemat_name.replace(".","")
	save_location = os.path.join(d_base,outmat_name+".mat")

savemat(save_location,{'relax':args.relax,'penalty':args.penalty,'sinit':args.sinit,'infoplane':datanpy,})

print('simulation complete, saving results to: {}'.format(save_location))
# saving numpy array
details_savepath = os.path.join(d_base,outmat_name+".pkl")
with open(details_savepath,"wb") as fid:
	np.save(fid,res_array)
print("details saved to {:}".format(details_savepath))

## plotting the ib curve

plt.scatter(datanpy[:,0],datanpy[:,1],marker="^",c='r')
# DPI bounds
plt.hlines(mixy,0,np.amax(datanpy[:,0]),linestyle=":")
plt.plot([0,mixy],[0,mixy],linestyle="-.")
plt.xlabel(r"$I(Z;X)$")
plt.ylabel(r"$I(Z;Y)$")
plt.savefig(os.path.join(d_base,outmat_name+".png"))
#plt.show()

# for debugging
#print(details_dict)