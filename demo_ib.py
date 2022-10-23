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
import datetime

d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('method',choices=alg.supportedIBAlg())
parser.add_argument('--ntime',type=int,help='run how many times per beta',default=16)
parser.add_argument('--penalty',type=float,help='penalty coefficient',default=8.0)
parser.add_argument('--relax',type=float,help='Relaxation parameter for DRS',default=1.00)
parser.add_argument('--thres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('--sinit',type=float,help='initial step size',default=1e-2)
parser.add_argument('--sscale',type=float,help='Scaling of step size',default=0.25)
parser.add_argument('--maxiter',type=int,help='Maximum number of iterations',default=5000)
parser.add_argument('--seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('--minbeta',type=float,help='the minimum beta',default=1.0)
parser.add_argument('--maxbeta',type=float,help='the maximum beta',default=10.0)
parser.add_argument('--numbeta',type=int,help='beta geometric space',default=16)
parser.add_argument('--record',action="store_true",default=False,help='Record the value decrease')
parser.add_argument('--dataset',type=str,help="dataset to run",default="syn")
parser.add_argument('--save_dir',type=str,default=None,help="output folder")
parser.add_argument('--init',choices=ut.getInitWays(),default="both")

args = parser.parse_args()
argdict = vars(args)
print(argdict)

d_beta_range = np.geomspace(args.minbeta,args.maxbeta,num=args.numbeta)
data = dt.getDataset(args.dataset)
#data = dt.uciHeart()
#data = dt.uciHeartFail()

px = np.sum(data['pxy'],axis=1)

algrun = alg.getIBAlgorithm(args.method)

result_dict= {}
details_dict = {}
# result_array
nz = data['nx'] # accounts for theoretic cardinality bound
result_array = np.zeros((args.numbeta * args.ntime*2*(nz),8))  # beta, niter, conv, izx, izy, entz, inittype, nz
res_cnt = 0
sinit = copy.copy(argdict['sinit'])
init_scheme_range = ut.getInitRange(args.init)
for run_nz in range(2,nz+2):  # should be upto |X|+1, by theorectic cardinality bound
	for randscheme in init_scheme_range:
		argdict['detinit'] = randscheme
		argdict['sinit'] = sinit
		for beta in d_beta_range:
			conv_cnt = 0
			for nn in range(args.ntime):
				runtime_dict = {"convthres":args.thres,**argdict}
				output = algrun(**{"pxy":data["pxy"],"nz":run_nz,"beta":beta,**runtime_dict})
				entz = ut.calcEnt(output['pzcx']@px)
				result_array[res_cnt,:] = np.array([beta,output['niter'],int(output['conv']),output["IZX"],output["IZY"],entz,randscheme,run_nz])
				res_cnt += 1
				conv_cnt += int(output['conv'])
				if output['conv']:
					izx_str = '{:.2f}'.format(output['IZX'])
					if not result_dict.get(izx_str,False):
						result_dict[izx_str] = 0
						details_dict[izx_str] = {}
					if result_dict[izx_str] <output['IZY']:
						result_dict[izx_str] = output['IZY']
						details_dict[izx_str]['beta'] = beta
						details_dict[izx_str]['randinit'] = randscheme
						details_dict[izx_str]['nz'] = nz
						details_dict[izx_str]['nrun'] = nn
			status_tex = 'beta,{:.2f},nz,{:},conv_rate,{:.4f},sinit,{:.4f},detinit,{:},ibc_item,{:}'.format(beta,run_nz,conv_cnt/args.ntime,argdict['sinit'],argdict['detinit'],len(details_dict))
			print('{:}'.format(status_tex))


dataout = []
for k,v in result_dict.items():
	izx_num = float(k)
	dataout.append([izx_num,v])
datanpy = np.array(dataout)

mixy = ut.calcMI(data['pxy'])

# saving the results
tnow = datetime.datetime.now()
if not args.save_dir:
	savebase = "ib_{:}_{:}_results_{:}".format(args.dataset,args.method,tnow.strftime("%Y%m%d"))
else:
	savebase = args.save_dir
d_save_dir = os.path.join(d_base,savebase)
os.makedirs(d_save_dir,exist_ok=True)
# save mat
savemat_name = 'ib_{method:}_{dataset:}_r{relax:.3f}_c{penalty:.2f}_si{sinit:4.2e}'.format(**argdict)
outmat_name = savemat_name.replace('.',"")
repeat_cnt = 0
save_location = os.path.join(d_save_dir,outmat_name+'.mat')
while os.path.isfile(save_location):
	repeat_cnt+=1
	savemat_name = 'ib_{method:}_{dataset:}_r{relax:.3f}_c{penalty:.2f}_si{sinit:4.2e}_{repeat_cnt:}'.format(**{**argdict,"repeat_cnt":repeat_cnt})
	outmat_name = savemat_name.replace('.',"")
	save_location = os.path.join(d_save_dir,outmat_name+'.mat')
savemat(save_location,{'relax':args.relax,'penalty':args.penalty,'sinit':args.sinit,'infoplane':datanpy,'results_array':result_array})

print('simulation complete, saving figure results to: {:}'.format(save_location))
# save details
details_savepath = os.path.join(d_save_dir,outmat_name+".npy")
with open(details_savepath,"wb") as fid:
	np.save(fid,result_array)
print("details saved to {:}".format(details_savepath))

# plotting the ib curve
plt.scatter(datanpy[:,0],datanpy[:,1],marker="^",c='r',label="solution")
plt.hlines(mixy,0,np.amax(datanpy[:,0]),linestyle=":")
plt.plot([0,mixy],[0,mixy],linestyle="-.")
plt.xlabel(r"$I(Z;X)$")
plt.ylabel(r"$I(Z;Y)$")
plt.savefig(os.path.join(d_save_dir,outmat_name+".png"))
