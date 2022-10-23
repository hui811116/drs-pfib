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
from scipy.io import savemat
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("method",choices=alg.supportedIBAlg())
parser.add_argument("--penalty",type=float,help="penalty coefficient",default=4.0)
parser.add_argument('--relax',type=float,help='Relaxation parameter for DRS',default=1.00)
parser.add_argument('--convthres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('--sinit',type=float,help='initial step size',default=1e-2)
parser.add_argument('--sscale',type=float,help='Scaling of step size',default=0.25)
parser.add_argument("--nrun",type=int,help="number of iteration for a fixed beta",default=10)
parser.add_argument('--maxiter',type=int,help='Maximum number of iterations',default=20000)
parser.add_argument('--seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('--minbeta',type=float,help='the minimum beta',default=1.0)
parser.add_argument('--maxbeta',type=float,help='the maximum beta',default=10.0)
parser.add_argument('--numbeta',type=int,help='beta geometric space',default=20)
parser.add_argument('--detinit',help='Start from a almost deterministic point',action='count',default=0)
#parser.add_argument('--init',choices=ut.getInitWays(),default="both")
parser.add_argument("--nz",type=int,default=3,help="representation dimension")
parser.add_argument('--record',action="store_true",default=False,help='Record the value decrease')
parser.add_argument('--dataset',type=str,help="dataset to run",default="syn")

args = parser.parse_args()
argdict = vars(args)
print(argdict)

d_beta_range = np.geomspace(args.minbeta,args.maxbeta,num=args.numbeta)
data = dt.getDataset(args.dataset)

px = np.sum(data['pxy'],axis=1)

algrun = alg.getIBAlgorithm(args.method)

ibcurve_details= {}
result_array = [] # beta, niter, conv, izx, izy, entz, inittype, nz
if args.detinit==0:
	init_scheme_range = ut.getInitRange("rnd")
else:
	init_scheme_range = ut.getInitRange("det")
for beta in d_beta_range:
	conv_cnt = 0
	for initscheme in init_scheme_range:
		for nidx in range(args.nrun):
			runtime_dict = {"penalty":args.penalty,"detinit":initscheme,"sinit":args.sinit,
							"sscale":args.sscale,"record":args.record,"relax":args.relax,"seed":args.seed}
			output = algrun(**{"pxy":data["pxy"],"nz":args.nz,"beta":beta,
				"convthres":args.convthres,"maxiter":args.maxiter,**runtime_dict})
			entz = ut.calcEnt(output['pzcx']@px)
			result_array.append([beta,output['niter'],output['conv'],output['IZX'],output['IZY'],entz,int(args.detinit),args.nz])
			conv_cnt += int(output['conv'])
			if output['conv']:
				izx_str = '{:.2f}'.format(output['IZX'])
				if not izx_str in ibcurve_details.keys():
					ibcurve_details[izx_str] = {'IZY':-1,}
				if ibcurve_details[izx_str]['IZY']< output['IZY']:
					ibcurve_details[izx_str] = {"IZY":output['IZY'],
							'penalty':args.penalty,'sinit':args.sinit,"nz":args.nz,'beta':beta,
							'detinit':args.detinit,'niter':output['niter']}
		print('beta,{:.4f},nz,{:},conv_rate,{:.6f},sinit,{:.5e},detinit,{:},ibc_items,{:}'.format(
			beta,args.nz,conv_cnt/args.nrun,args.sinit,args.detinit,len(ibcurve_details)))
# numpy results
result_array  =np.array(result_array)

tnow = datetime.datetime.now()

savename = 'ib_{dataset:}_{method:}_r{relax:.3f}_c{penalty:}_si{sinit:4.2e}_det{detinit:}_nz{nz:}_{dtstr:}'.format(**{**argdict,"dtstr":tnow.strftime("%Y%m%d")})
repeat_cnt = 0
safe_save_file = "{:}".format(savename)
while os.path.isfile(safe_save_file+".npy"):
	repeat_cnt += 1
	safe_save_file = "{:}_{:}".format(savename,repeat_cnt)
print("saving results as: {:}".format(safe_save_file))
with open("{:}.npy".format(safe_save_file),"wb") as fid:
	np.save(fid,result_array)
with open("{:}.pkl".format(safe_save_file),'wb') as fid:
	pickle.dump(ibcurve_details,fid)
savemat(os.path.join(os.getcwd(),safe_save_file+".mat"),{'method':args.method,'dataset':args.dataset,'relax':args.relax,'penalty':args.penalty,'sinit':args.sinit,'results_array':result_array})
