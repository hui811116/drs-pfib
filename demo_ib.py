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
parser.add_argument('method',choices=['admm1','admm2','logadmm1','logadmm2'])
#parser.add_argument("--beta",type=float,help='the PF beta',default=5.0)
parser.add_argument('--ntime',type=int,help='run how many times per beta',default=20)
parser.add_argument('--penalty',type=float,help='penalty coefficient',default=4.0)
parser.add_argument('--relax',type=float,help='Relaxation parameter for DRS',default=1.00)
parser.add_argument('--thres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('--sinit',type=float,help='initial step size',default=1e-2)
parser.add_argument('--sscale',type=float,help='Scaling of step size',default=0.25)
parser.add_argument('--maxiter',type=int,help='Maximum number of iterations',default=20000)
parser.add_argument('--seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('--minbeta',type=float,help='the minimum beta',default=2.0)
parser.add_argument('--maxbeta',type=float,help='the maximum beta',default=10.0)
parser.add_argument('--numbeta',type=int,help='beta geometric space',default=20)
#parser.add_argument('--detinit',help='Start from a almost deterministic point',action='count',default=0)
parser.add_argument('--record',action="store_true",default=False,help='Record the value decrease')
parser.add_argument('--dataset',type=str,help="dataset to run",default="syn")

args = parser.parse_args()
argdict = vars(args)
print(argdict)

d_beta_range = np.geomspace(args.minbeta,args.maxbeta,num=args.numbeta)
if args.dataset == "syn":
	data = dt.synMy()
elif args.dataset == "heartfail":
	data = dt.uciHeartFail()
else:
	sys.exit("undefined dataset {:}".format(args.dataset))
#data = dt.uciHeart()
#data = dt.uciHeartFail()

px = np.sum(data['pxy'],axis=1)

if args.method == "admm1":
	algrun = alg.drsIBType1
elif args.method == "admm2":
	algrun = alg.drsIBType2
elif args.method == "logadmm1":
	algrun = alg.admmIBLogSpaceType1
elif args.method == "logadmm2":
	algrun = alg.admmIBLogSpaceType2
else:
	sys.exit("undefined method {:}".format(args.method))

result_dict= {}
# result_array
result_array = np.zeros((args.numbeta * args.ntime*2,7))  # beta, niter, conv, izx, izy, entz, inittype
res_cnt = 0
nz = data['nx']
sinit = copy.copy(argdict['sinit'])
for randscheme in range(2):
	argdict['detinit'] = randscheme
	argdict['sinit'] = sinit
	for beta in d_beta_range:
		conv_cnt = 0
		for nn in range(args.ntime):
			#print('\rProgress: beta={:.2f}, run={:>5}/{:>5}, nz={:>3}, ss_init={:8.2e}, detinit={:>3}'.format(beta,nn,args.ntime,nz,argdict['sinit'],argdict['detinit']),end='',flush=True)
			runtime_dict = {"convthres":args.thres,**argdict}
			output = algrun(**{"pxy":data["pxy"],"nz":nz,"beta":beta,**runtime_dict})
			entz = ut.calcEnt(output['pzcx']@px)
			result_array[res_cnt,:] = np.array([beta,output['niter'],int(output['conv']),output["IZX"],output["IZY"],entz,randscheme])
			res_cnt += 1
			conv_cnt += int(output['conv'])
			if output['conv']:
				izx_str = '{:.2f}'.format(output['IZX'])
				if not result_dict.get(izx_str,False):
					result_dict[izx_str] = 0
				if result_dict[izx_str] <output['IZY']:
					result_dict[izx_str] = output['IZY']
		status_tex = 'beta,{:.2f},nz,{:},conv_rate,{:.4f},sinit,{:.4f},detinit,{:}'.format(beta,nz,conv_cnt/args.ntime,argdict['sinit'],argdict['detinit'])
		#print('\r{:<200}'.format(status_tex),end='\r',flush=True)
		print('{:}'.format(status_tex))
		#argdict['sinit'] *= 0.9
#print(' '*200+'\r',end='',flush=True)


dataout = []
for k,v in result_dict.items():
	izx_num = float(k)
	dataout.append([izx_num,v])
datanpy = np.array(dataout)

mixy = ut.calcMI(data['pxy'])

# save mat
savemat_name = 'ib_{:}_{:}_r{:}_c{:}_si{:4.2e}'.format(args.dataset,args.method,args.relax,int(args.penalty),sinit)
outmat_name = savemat_name.replace('.',"")
repeat_cnt = 0
save_location = os.path.join(d_base,outmat_name+'.mat')
while os.path.isfile(save_location):
	repeat_cnt+=1
	savemat_name = 'ib_{:}_{:}_r{:}_c{:}_si{:4.2e}_{:}'.format(args.dataset,args.method,args.relax,int(args.penalty),sinit,repeat_cnt)
	outmat_name = savemat_name.replace('.',"")
	save_location = os.path.join(d_base,outmat_name+'.mat')
savemat(save_location,{'relax':args.relax,'penalty':args.penalty,'sinit':args.sinit,'infoplane':datanpy,})

print('simulation complete, saving figure results to: {:}'.format(save_location))
# save details
details_savepath = os.path.join(d_base,outmat_name+".npy")
with open(details_savepath,"wb") as fid:
	np.save(fid,result_array)
print("details saved to {:}".format(details_savepath))

# plotting the ib curve
plt.scatter(datanpy[:,0],datanpy[:,1],marker="^",c='r',label="solution")
plt.hlines(mixy,0,np.amax(datanpy[:,0]),linestyle=":")
plt.plot([0,mixy],[0,mixy],linestyle="-.")
plt.xlabel(r"$I(Z;X)$")
plt.ylabel(r"$I(Z;Y)$")
plt.show()