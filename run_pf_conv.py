import numpy as np
import os
import sys
import time
import pickle
import argparse
import algorithm as alg
import dataset as dt
import utils as ut
import copy
import datetime
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument("method",choices=alg.supportedPfAlg())
parser.add_argument('--relax',type=float,help='Relaxation parameter for DRS',default=1.00)
parser.add_argument('--convthres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('--sinit',type=float,help='initial step size',default=1e-2)
parser.add_argument('--sscale',type=float,help='Scaling of step size',default=0.25)
parser.add_argument("--nrun",type=int,help="number of iteration for a fixed beta",default=10)
parser.add_argument('--maxiter',type=int,help='Maximum number of iterations',default=1000)
parser.add_argument('--seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('--beta',type=float,help="The fixed trade-off parameter",default=5.0)
parser.add_argument('--minc',type=float,help='the minimum penalty coefficient',default=10)
parser.add_argument('--maxc',type=float,help='the maximum penalty coefficient',default=1e5)
parser.add_argument('--numc',type=int,help='geometric space for the penalty coefficient',default=200)
parser.add_argument('--detinit',help='Start from a almost deterministic point',choices=ut.getInitWays(),default="rnd")
parser.add_argument("--nz",type=int,default=3,help="representation dimension")
parser.add_argument('--record',action="store_true",default=False,help='Record the value decrease')
parser.add_argument('--dataset',type=str,help="dataset to run",default="syn")
parser.add_argument('--save_dir',type=str,default=None,help="output folder")

args = parser.parse_args()
argdict = vars(args)
print(argdict)

d_c_range = np.geomspace(args.minc,args.maxc,num=args.numc)
data = dt.getDataset(args.dataset)

px = np.sum(data['pxy'],axis=1)

algrun = alg.getPFAlgorithm(args.method)
infocurve_details = {}
result_array = [] # beta, niter, conv, izx, izy, entz, inittype

for penalty_c in d_c_range:
	conv_cnt = 0
	for nn in range(args.nrun):
		runtime_dict = {"penalty":penalty_c,"detinit":args.detinit,"sinit":args.sinit,
						"sscale":args.sscale,"record":args.record,"relax":args.relax,"seed":args.seed}
		output = algrun(**{"pxy":data['pxy'],'nz':args.nz,"beta":args.beta,
							"convthres":args.convthres,"maxiter":args.maxiter,**runtime_dict})
		entz = ut.calcEnt(output['pzcx']@px)
		result_array.append([args.beta,output['niter'],int(output['conv']),output['IZX'],output['IZY'],entz,args.detinit])
		conv_cnt += int(output['conv'])
		if output['conv']:
			izx_str = "{:.2f}".format(output['IZX'])
			if not izx_str in infocurve_details.keys():
				infocurve_details[izx_str] = {"IZY":np.Inf}
			if infocurve_details[izx_str]['IZY'] > output['IZY']:
				infocurve_details[izx_str] = {"IZY":output['IZY'],
					"penalty":penalty_c,"sinit":args.sinit,"nz":args.nz,"beta":args.beta,
					"detinit":args.detinit,"niter":output['niter']}
	print('penc,{:.5f},beta,{:.4f},nz,{:},conv_rate,{:.6f},sinit,{:.5e},detinit,{:},pfc_items,{:}'.format(
			penalty_c,args.beta,args.nz,conv_cnt/args.nrun,args.sinit,args.detinit,len(infocurve_details),))

result_array = np.array(result_array)

tnow = datetime.datetime.now()
if not args.save_dir:
	savebase = "conv_pf_{dataset:}_{method:}_{timestring:}".format(**{**argdict,"timestring":tnow.strftime("%Y%m%d")})
else:
	savebase=args.save_dir
d_save_dir = os.path.join(os.getcwd(),savebase)
os.makedirs(d_save_dir,exist_ok=True)
save_name= "conv_pf_{dataset:}_{method:}_results_r{relax:.3f}_beta{beta:.4f}_si{sinit:4.2e}".format(**argdict)
save_location = os.path.join(d_save_dir,save_name+".mat")
repeat_cnt = 0
while os.path.isfile(save_location):
	repeat_cnt += 1
	save_name = "conv_pf_{dataset:}_{method:}_results_r{relax:.3f}_beta{beta:.4f}_si{sinit:4.2e}_{repeat_cnt:}".format(**{**argdict,"repeat_cnt":repeat_cnt})
	save_location = os.path.join(d_save_dir,save_name+".mat")
savemat(save_location,{'method':args.method,'dataset':args.dataset,'relax':args.relax,'beta':args.beta,'sinit':args.sinit,'results_array':result_array})
print('simulation complete, saving results to: {:}'.format(save_location))
# saving the configuration
config_path = os.path.join(d_save_dir,save_name+".pkl")
with open(config_path,'wb') as fid:
	pickle.dump(argdict,fid)
# saving numpy array
with open(os.path.join(d_save_dir,save_name+".npy"),'wb') as fid:
	np.save(fid,result_array)
