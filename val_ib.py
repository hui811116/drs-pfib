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
parser.add_argument('--beta',type=float,help="the trade-off parameter",default=5.0)
#parser.add_argument('--minbeta',type=float,help='the minimum beta',default=1.0)
#parser.add_argument('--maxbeta',type=float,help='the maximum beta',default=10.0)
#parser.add_argument('--numbeta',type=int,help='beta geometric space',default=20)
parser.add_argument("--load",type=int,help="index of the initial point",default=0)
#parser.add_argument('--detinit',help='Start from a almost deterministic point',action='count',default=0)
#parser.add_argument('--init',choices=ut.getInitWays(),default="both")
parser.add_argument("--nz",type=int,default=3,help="representation dimension")
#parser.add_argument('--record',action="store_true",default=True,help='Record the value decrease')
parser.add_argument('--dataset',type=str,help="dataset to run",default="syn")

args = parser.parse_args()
argdict = vars(args)
print(argdict)

data = dt.getDataset(args.dataset)

px = np.sum(data['pxy'],axis=1)

algrun = alg.getIBAlgorithm(args.method)

#ibcurve_details= {}
result_array = [] # beta, niter, conv, izx, izy, entz, inittype, nz

if args.load == 0:
	init_pzcx = ut.initPzcx(0,1e-5,args.nz,len(px),args.seed)
	print("randomly initialized starting point:")
	print(init_pzcx)
elif args.load == 1:
	init_pzcx = ut.loadPzcx(args.load,args.dataset)
	print("loaded initialized point:")
	print(init_pzcx)
else:
	sys.exit("undefined initialization method")

# run ntimes and pick the best loss
# 
conv_flag = False
best_result = {"loss":np.Inf,"progress":np.zeros((1,)),'output':{},'IZX':0,'IZY':0,'HZ':0}
nrun_final = args.nrun
if args.load != 0:
	nrun_final = 1
for nidx in range(nrun_final):
	runtime_dict = {"penalty":args.penalty,"sinit":args.sinit,
					"sscale":args.sscale,"record":True,"relax":args.relax,"seed":args.seed,"load":init_pzcx,'detinit':0}
	output = algrun(**{"pxy":data["pxy"],"nz":args.nz,"beta":args.beta,
		"convthres":args.convthres,"maxiter":args.maxiter,**runtime_dict})
	entz = ut.calcEnt(output['pzcx']@px)
	result_array.append([args.beta,output['niter'],output['conv'],output['IZX'],output['IZY'],entz,args.nz])
	#conv_cnt += int(output['conv'])
	if output['conv']:
		conv_flag = True
		tmp_loss = (1/args.beta)*output['IZX'] - output['IZY']
		if tmp_loss < best_result['loss']:
			best_result['loss'] = tmp_loss
			best_result['record'] = output['record']
			best_result['pzcx'] = output['pzcx']
			best_result['IZX'] = output['IZX']
			best_result['IZY'] = output['IZY']
			best_result['HZ'] = entz
			best_result['niter'] = output['niter']

if not conv_flag:
	sys.exit("no convergence, abort")
# print info for selection
print("best_loss:{:.6f}, niter:{:}".format(best_result['loss'],best_result['niter']))
print("best_pzcx")
print(best_result['pzcx'])

result_array = np.array(result_array)
tnow = datetime.datetime.now()

savename = 'val_ib_{dataset:}_{method:}_r{relax:.3f}_c{penalty:}_si{sinit:.2e}_nz{nz:}_beta{beta:.4f}_load{load:}_{dtstr:}'.format(**{**argdict,"dtstr":tnow.strftime("%Y%m%d")})
repeat_cnt = 0
safe_save_file = "{:}".format(savename)
while os.path.isfile(safe_save_file+".npy"):
	repeat_cnt += 1
	safe_save_file = "{:}_{:}".format(savename,repeat_cnt)
print("saving results as: {:}".format(safe_save_file))
with open("{:}.npy".format(safe_save_file),"wb") as fid:
	np.save(fid,result_array)
with open("{:}.pkl".format(safe_save_file),'wb') as fid:
	pickle.dump(best_result,fid)
savemat(os.path.join(os.getcwd(),safe_save_file+".mat"),{'method':args.method,'dataset':args.dataset,'relax':args.relax,'penalty':args.penalty,'sinit':args.sinit,'record':best_result['record'],'best_loss':best_result['loss']})
