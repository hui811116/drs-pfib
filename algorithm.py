import numpy as np
import sys
import gradient_descent as gd
import utils as ut
import copy

def supportedIBAlg():
	return ['admm1','admm2','logadmm1','logadmm2','admm1ms','admm2ms','ba','admmdec','admmdecms']
def getIBAlgorithm(method):
	if method == "admm1":
		return drsIBType1
	elif method == "admm2":
		return drsIBType2
	elif method == "logadmm1":
		return admmIBLogSpaceType1
	elif method == "logadmm2":
		return admmIBLogSpaceType2
	elif method == "admm1ms":
		return drsIBType1MeanSub
	elif method == "admm2ms":
		return drsIBType2MeanSub
	elif method == 'ba':
		return ibOrig
	elif method == "admmdec":
		return admmIBDec
	elif method == "admmdecms":
		return admmIBDecMs
	else:
		sys.exit("undefined method {:}".format(method))

def supportedPfAlg():
	return ['admm','logadmm','ba','admmms','admmenc']

def getPFAlgorithm(method):
	if method == "admm":
		return drsPF
	elif method == "logadmm":
		return pfLogSpace
	elif method == "admmms":
		return drsPFMeanSub
	elif method == "admmenc":
		return admmPFEnc
	elif method == "ba":
		return ibOrig
	else:
		sys.exit("undefined method {:}".format(method))

def drsPF(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	#pzcx = ut.initPzcx(det_init,1e-3,nz,nx,kwargs['seed'])
	# NOTE: nz<= nx always
	##
	pzcy = pzcx @ pxcy
	dual_y = np.zeros((nz,ny))
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv = False
	while itcnt < maxiter:
		pz = np.sum(pzcx*px[None,:],axis=1)
		erry = pzcy - pzcx@pxcy
		# function value: (beta-1) H(Z) -beta H(Z|Y) + H(Z|X)
		record_mat[itcnt%record_mat.shape[0]] = (beta-1) * np.sum(-pz*np.log(pz)) -beta*np.sum(-pzcy*py[None,:]*np.log(pzcy))\
							+np.sum(-pzcx*px[None,:]*np.log(pzcx)) + np.sum(dual_y*erry)\
							+0.5*penalty* (np.linalg.norm(erry)**2)
		itcnt += 1
		# solve -beta H(Z|Y)
		grad_y = beta*(np.log(pzcy)+1)*py[None,:] + (dual_y+penalty*erry)
		mean_grad_y = grad_y - np.mean(grad_y,axis=0)
		ss_y = gd.validStepSize(pzcy,-mean_grad_y,ss_init,ss_scale)
		if ss_y ==0:
			break
		new_pzcy = pzcy - mean_grad_y*ss_y

		erry = new_pzcy - pzcx @ pxcy
		dual_drs_y= dual_y -(1-alpha)*penalty*erry
		# solve (beta-1)H(Z) + H(Z|X)
		grad_x = (1-beta)*(np.log(pz)+1)[:,None]*px[None,:]-(np.log(pzcx)+1)*px[None,:]-(dual_drs_y+penalty*erry)@pxcy.T
		mean_grad_x = grad_x-np.mean(grad_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_grad_x,ss_init,ss_scale)
		if ss_x == 0:
			break
		new_pzcx = pzcx - ss_x * mean_grad_x
		erry = new_pzcy - new_pzcx@pxcy
		dual_y = dual_drs_y + penalty*erry
		dtvy = 0.5* np.sum(np.fabs(erry),axis=0)
		if np.all(dtvy<convthres):
			conv = True
			break
		else:
			pzcx = new_pzcx
			pzcy = new_pzcy
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,'IZY':mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict
'''
def drsPFArm(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	# function and gradient objects
	fobj_pzcy = gd.pfPzcyFuncObj(py,pxcy,beta,penalty)
	fobj_pzcx = gd.pfPzcxFuncObj(px,pxcy,beta,penalty)
	gobj_pzcy = gd.pfPzcyGradObj(py,pxcy,beta,penalty)
	gobj_pzcx = gd.pfPzcxGradObj(px,pxcy,beta,penalty)

	pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	# call the function value objects
	# NOTE: nz<= nx always
	pzcy = pzcx @ pxcy

	dual_y = np.zeros((nz,ny))
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv = False
	while itcnt < maxiter:
		pz = np.sum(pzcx*px[None,:],axis=1)
		erry = pzcy - pzcx@pxcy
		# function value: (beta-1) H(Z) -beta H(Z|Y) + H(Z|X)
		record_mat[itcnt%record_mat.shape[0]] = (beta-1) * np.sum(-pz*np.log(pz)) -beta*np.sum(-pzcy*py[None,:]*np.log(pzcy))\
							+np.sum(-pzcx*px[None,:]*np.log(pzcx)) + np.sum(dual_y*erry)\
							+0.5*penalty* (np.linalg.norm(erry)**2)
		itcnt += 1
		# solve -beta H(Z|Y)
		grad_y= gobj_pzcy(pzcy,pzcx,dual_y)
		mean_grad_y = grad_y - np.mean(grad_y,axis=0)
		ss_y = gd.validStepSize(pzcy,-mean_grad_y,ss_init,ss_scale)
		if ss_y ==0:
			break
		arm_y = gd.armijoStepSize(pzcy,-mean_grad_y,ss_y,ss_scale,1e-4,fobj_pzcy,gobj_pzcy,**{"pzcx":pzcx,"dual_y":dual_y})
		if arm_y == 0:
			arm_y = ss_y
		new_pzcy = pzcy - mean_grad_y*arm_y

		erry = new_pzcy - pzcx @ pxcy
		dual_drs_y= dual_y -(1-alpha)*penalty*erry
		# solve (beta-1)H(Z) + H(Z|X)
		grad_x = gobj_pzcx(pzcx,new_pzcy,dual_drs_y)
		mean_grad_x = grad_x-np.mean(grad_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_grad_x,ss_init,ss_scale)
		if ss_x == 0:
			break
		arm_x = gd.armijoStepSize(pzcx,-mean_grad_x,ss_x,ss_scale,1e-4,fobj_pzcx,gobj_pzcx,**{"pzcy":new_pzcy,"dual_y":dual_drs_y})
		if arm_x == 0:
			arm_x = ss_x
		new_pzcx = pzcx - arm_x * mean_grad_x

		erry = new_pzcy - new_pzcx@pxcy
		dual_y = dual_drs_y + penalty*erry
		dtvy = 0.5* np.sum(np.fabs(erry),axis=0)
		if np.all(dtvy<convthres):
			conv = True
			break
		else:
			pzcx = new_pzcx
			pzcy = new_pzcy
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,'IZY':mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict
'''
def drsPFMeanSub(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	# function and gradient objects
	fobj_pzcy = gd.pfPzcyFuncObj(py,pxcy,beta,penalty)
	fobj_pzcx = gd.pfPzcxFuncObj(px,pxcy,beta,penalty)
	gobj_pzcy = gd.pfPzcyGradObj(py,pxcy,beta,penalty)
	gobj_pzcx = gd.pfPzcxGradObj(px,pxcy,beta,penalty)
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	#pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	# call the function value objects
	# NOTE: nz<= nx always
	pzcy = pzcx @ pxcy
	dual_y = np.zeros((nz,ny))
	
	# masks
	mask_x = np.ones(pzcx.shape)
	mask_y = np.ones(pzcy.shape)

	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv_flag = False
	while itcnt < maxiter:
		est_pzcy = pzcx @ pxcy
		err_y = pzcy - est_pzcy
		est_pz = np.sum(pzcx * px[None,:],axis=1)
		record_mat[itcnt % record_mat.shape[0]] = -beta*np.sum(-pzcy*py[None,:]*np.log(pzcy)) +(beta-1)*np.sum(-est_pz*np.log(est_pz)) \
													+np.sum(-pzcx*px[None,:]*np.log(pzcx)) + np.sum(dual_y*err_y) +0.5*penalty*(np.linalg.norm(err_y)**2)
		itcnt +=1
		# projection on negative-log space
		# note that the error is still defined on the euclidean space
		grad_y = gobj_pzcy(pzcy,pzcx,dual_y)
		mean_grad_y = grad_y*mask_y - np.mean(grad_y*mask_y,axis=0)
		ss_y = gd.validStepSize(pzcy,-mean_grad_y*mask_y,ss_init,ss_scale)
		if ss_y ==0:
			# there are two cases: 1) some element reaches 1, 2) some elements lower than zero
			if np.any(pzcy-mean_grad_y*mask_y*1e-9<=0.0):
				gmask_eps_y = np.invert(pzcy<1e-9)
				mask_y = np.logical_and(gmask_eps_y,mask_y)
				mean_grad_y = grad_y*mask_y - np.mean(grad_y*mask_y,axis=0)
				ss_y = gd.validStepSize(pzcy,-mean_grad_y*mask_y,ss_init,ss_scale)
			elif np.any(pzcy-mean_grad_y*mask_y*1e-9>=1.0):
				bad_cols = np.any(pzcy-mean_grad_y*mask_y*1e-9>=1.0,axis=0)
				mask_y[:,bad_cols] = 0
				mean_grad_y = grad_y*mask_y - np.mean(grad_y*mask_y,axis=0)
				ss_y = gd.validStepSize(pzcy,-mean_grad_y*mask_y,ss_init,ss_scale)
			else:
				break
		arm_y = gd.armijoStepSize(pzcy,-mean_grad_y*mask_y,ss_y,ss_scale,1e-4,fobj_pzcy,gobj_pzcy,**{"pzcx":pzcx,"dual_y":dual_y})
		if arm_y ==0:
			arm_y = ss_y
		#arm_y = ss_y
		raw_pzcy = pzcy - arm_y * mean_grad_y *mask_y + 1e-9
		new_pzcy = raw_pzcy / np.sum(raw_pzcy,axis=0,keepdims=True)
		err_y = new_pzcy - est_pzcy
		dual_drs_y = dual_y -(1-alpha)*penalty*err_y
		
		grad_x = gobj_pzcx(pzcx,new_pzcy,dual_drs_y)
		mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
		if ss_x == 0:
			if np.any(pzcx-mean_grad_x*mask_x*1e-9<=0.0):
				gmask_eps_x = np.invert(pzcx < 1e-9) # mask small values
				mask_x = np.logical_and(gmask_eps_x,mask_x)
				mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			elif np.any(pzcx-mean_grad_x*mask_x*1e-9>=1.0):
				bad_cols = np.any(pzcx-mean_grad_x*mask_x*1e-9>=1.0,axis=0)
				mask_x[:,bad_cols] = 0
				mean_grad_x = grad_x * mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			else:
				break
		arm_x = gd.armijoStepSize(pzcx,-mean_grad_x*mask_x,ss_x,ss_scale,1e-4,fobj_pzcx,gobj_pzcx,**{"pzcy":new_pzcy,"dual_y":dual_drs_y})
		if arm_x ==0:
			arm_x = ss_x
		#arm_x = ss_x
		raw_pzcx = pzcx - arm_x * mean_grad_x * mask_x + 1e-9
		new_pzcx = raw_pzcx / np.sum(raw_pzcx,axis=0,keepdims=True)
		err_y = new_pzcy - new_pzcx @ pxcy
		dual_y = dual_drs_y + penalty * err_y
		conv_y = 0.5 * np.sum(np.fabs(err_y),axis=0)
		if np.all(conv_y<convthres):
			conv_flag = True
			break
		else:
			pzcx = new_pzcx
			pzcy = new_pzcy
	mizx = ut.calcMI(pzcx * px[None,:])
	mizy = ut.calcMI(pzcy * py[None,:])
	output_dict = {"pzcx":pzcx,"niter":itcnt,"conv":conv_flag,"IZX":mizx,"IZY":mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict


def pfLogSpace(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	ss_fixed = kwargs['sinit']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)
	#pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	# call the function value objects
	# NOTE: nz<= nx always
	pz = np.sum(pzcx*px[None,:],axis=1)
	pzcy = pzcx @ pxcy
	# minus log variables
	mlog_pzcx = -np.log(pzcx)
	mlog_pzcy = -np.log(pzcy)

	dual_y = np.zeros((nz,ny))
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	itcnt = 0
	conv_flag = False
	while itcnt < maxiter:
		exp_mlog_pzcx = np.exp(-mlog_pzcx)
		exp_mlog_pzcy = np.exp(-mlog_pzcy)
		est_pzcy = exp_mlog_pzcx@pxcy
		est_mlog_pzcy = -np.log(est_pzcy)
		err_y = mlog_pzcy - est_mlog_pzcy
		# 
		tmp_pz = np.sum(exp_mlog_pzcx*px[None,:],axis=1)
		record_mat[itcnt % record_mat.shape[0]] = -beta * np.sum(exp_mlog_pzcy*mlog_pzcy*py[None,:]) + (beta-1) * np.sum(-tmp_pz*np.log(tmp_pz))\
												  +np.sum(exp_mlog_pzcx*px[None,:]*mlog_pzcx) + np.sum(dual_y*err_y) + 0.5 * penalty * np.linalg.norm(err_y)**2
		itcnt +=1
		# grad_y
		grad_y = -beta* exp_mlog_pzcy*(1-mlog_pzcy)*py[None,:] + dual_y+ penalty * err_y
		#grad_y = -exp_mlog_pzcy*(1-mlog_pzcy)*py[None,:] + dual_y+ penalty * err_y
		raw_mlog_pzcy = mlog_pzcy - ss_fixed * grad_y
		grad_mlog_pzcy = np.amin(raw_mlog_pzcy,axis=0)
		grad_mlog_pzcy = np.where(grad_mlog_pzcy<=0.0,grad_mlog_pzcy,np.zeros((ny,)))
		#raw_mlog_pzcy -= np.amin(raw_mlog_pzcy,axis=0)
		raw_mlog_pzcy = raw_mlog_pzcy - grad_mlog_pzcy[None,:]
		exp_mlog_pzcy = np.exp(-raw_mlog_pzcy) + 1e-7 # smoothing
		new_pzcy =exp_mlog_pzcy/np.sum(exp_mlog_pzcy,axis=0,keepdims=True)
		new_mlog_pzcy = -np.log(new_pzcy)
		# grad_x
		err_y = new_mlog_pzcy - est_mlog_pzcy
		dual_drs_y= dual_y -(1.0-alpha)* penalty*err_y

		grad_x = (exp_mlog_pzcx *px[None,:]) * (-mlog_pzcx +(beta-1)*(np.log(tmp_pz)+1)[:,None]+1)\
				 - exp_mlog_pzcx * (((dual_drs_y + penalty * err_y)/est_pzcy)@pxcy.T)
		#grad_x = (exp_mlog_pzcx *px[None,:]) * ((-1/beta)*mlog_pzcx +(1-1/beta)*(np.log(tmp_pz)+1)[:,None]+1/beta)\
		#		 - exp_mlog_pzcx * (((dual_drs_y + penalty * err_y)/est_pzcy)@pxcy.T)
		raw_mlog_pzcx = mlog_pzcx - ss_fixed * grad_x
		grad_mlog_pzcx = np.amin(raw_mlog_pzcx,axis=0)
		grad_mlog_pzcx = np.where(grad_mlog_pzcx<=0.0,grad_mlog_pzcx,np.zeros((nx,)))
		#raw_mlog_pzcx -= np.amin(raw_mlog_pzcx,axis=0)
		raw_mlog_pzcx = raw_mlog_pzcx - grad_mlog_pzcx[None,:]
		exp_mlog_pzcx = np.exp(-raw_mlog_pzcx) + 1e-7
		new_pzcx = exp_mlog_pzcx/np.sum(exp_mlog_pzcx,axis=0,keepdims=True)
		new_mlog_pzcx = -np.log(new_pzcx)

		# dual update
		est_pzcy = new_pzcx @ pxcy 
		err_y = new_mlog_pzcy + np.log(est_pzcy)
		dual_y = dual_drs_y+  penalty * err_y
		# convergence 
		conv_y = 0.5 * np.sum(np.fabs(new_pzcy-est_pzcy ),axis=0)
		if np.all(conv_y<convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx = new_mlog_pzcx
			mlog_pzcy = new_mlog_pzcy
	pzcx = np.exp(-mlog_pzcx)
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx * px[None,:])
	mizy = ut.calcMI(pzcy * py[None,:])
	output_dict = {"pzcx":pzcx, "niter":itcnt, "conv":conv_flag,"IZX":mizx,"IZY":mizy}
	if record_flag:
		output_dict["record"] = record_mat[:itcnt]
	return output_dict

'''
def admmPFEnc(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	ss_fixed = kwargs['sinit']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	expand_pycx = np.repeat(np.expand_dims(pycx,axis=1),repeats=nz,axis=1)

	pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	# call the function value objects
	# NOTE: nz<= nx always
	pz = np.sum(pzcx*px[None,:],axis=1)
	#pzcy = pzcx @ pxcy
	dual_z = np.zeros(pz.shape)
	itcnt = 0
	record_mat = np.zeros((1,))
	if record_flag:
		record_mat = np.zeros((maxiter,))
	conv_flag = False
	while itcnt < maxiter:
		est_pzcy = pzcx @ pxcy
		est_pz = np.sum(pzcx * px[None,:],axis=1)
		est_pycz = (est_pzcy/est_pz[:,None]).T
		expand_pycz = np.repeat(np.expand_dims(est_pycz,axis=2),repeats=nx,axis=2)
		err_z = pz - est_pz
		record_mat[itcnt%record_mat.shape[0]] = beta*np.sum(est_pzcy*py[None,:]*np.log(est_pzcy/est_pz[:,None]))\
												-np.sum(-pz*np.log(pz))+np.sum(-pzcx*px[None,:]*np.log(pzcx))\
												+np.sum(dual_z*err_z)+ 0.5 * penalty*(np.linalg.norm(err_z)**2)

		itcnt+=1
		# convex
		raw_pz = np.exp(-dual_z - penalty*err_z)
		new_pz = raw_pz / np.sum(raw_pz,axis=0)

		# dual update
		err_z = new_pz - est_pz
		dual_z += penalty * err_z

		# encoder update
		kl_ker = np.sum(expand_pycx*np.log(expand_pycx/expand_pycz),axis=0)
		raw_pzcx = np.exp(-beta*kl_ker + np.repeat((dual_z + penalty*err_z)[:,None],repeats=nx,axis=1))
		new_pzcx = raw_pzcx/ np.sum(raw_pzcx,axis=0,keepdims=True)

		# error
		est_pz = np.sum(new_pzcx*px[None,:],axis=1)
		err_z= new_pz - est_pz

		conv_z = 0.5*np.sum(np.fabs(err_z))
		if np.all(conv_z < convthres):
			conv_flag=True
			break
		else:
			pz = new_pz
			pzcx = new_pzcx
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {"niter":itcnt,"conv":conv_flag,"IZX":mizx,"IZY":mizy,'pzcx':pzcx}
	if record_flag:
		output_dict['record'] = record_mat
	return output_dict
'''

def admmPFEnc(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	fobj_pzcx = gd.pfVarPzcxFuncObj(px,pxcy,py,beta,penalty)
	fobj_pz = gd.pfVarPzFuncObj(px,beta,penalty)
	gobj_pzcx = gd.pfVarPzcxGradObj(px,pxcy,py,beta,penalty)
	gobj_pz = gd.pfVarPzGradObj(px,beta,penalty)
	entx = np.sum(-px*np.log(px))
	#pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	# call the function value objects
	# NOTE: nz<= nx always
	pz = np.sum(pzcx*px[None,:],axis=1)
	dual_z = np.zeros(pz.shape)
	# masks
	mask_z = np.ones(pz.shape)
	mask_x = np.ones(pzcx.shape)
	itcnt = 0
	record_mat = np.zeros((1,))
	if record_flag:
		record_mat = np.zeros((maxiter,))
	conv_flag = False
	while itcnt < maxiter:
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		est_pzcy = pzcx@pxcy
		qxcz = ((pzcx*px[None,:])/est_pz[:,None]).T
		err_z = est_pz - pz
		record_mat[itcnt%record_mat.shape[0]] = (beta)*np.sum(-pz * np.log(pz))  -beta *np.sum(-est_pzcy*py[None,:]*np.log(est_pzcy))\
									-entx + np.sum(-pzcx*px[None,:]*np.log(qxcz).T)\
									+ np.sum(dual_z * err_z) + 0.5 * penalty * (np.linalg.norm(err_z)**2)
		itcnt+=1
		
		#grad_x = beta*(np.log(est_pzcy)+1)@pxy.T - (np.log(qxcz)*px[:,None]).T + np.repeat((dual_z+penalty*err_z)[:,None],repeats=nx,axis=1)*px[None,:]
		grad_x = gobj_pzcx(pzcx,pz,dual_z,qxcz)
		mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
		if ss_x == 0:
			if np.any(pzcx - mean_grad_x*mask_x*1e-9<=0):
				gmask_x = pzcx>=1e-7
				mask_x = np.logical_and(mask_x,gmask_x)
				mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			elif np.any(pzcx - mean_grad_x*mask_x*1e-9>=1.0):
				bad_cols = np.any(pzcx - mean_grad_x*mask_x*1e-9,axis=0)
				mask_x[:,bad_cols] = 0
				mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			else:
				break
		# FIXME: armijo
		arm_x = gd.armijoStepSize(pzcx,-mean_grad_x*mask_x,ss_x,ss_scale,1e-4,fobj_pzcx,gobj_pzcx,**{"pz":pz,"dual_z":dual_z,"qxcz":qxcz})
		if arm_x ==0:
			arm_x = ss_x
		new_pzcx = pzcx - mean_grad_x * arm_x * mask_x + 1e-9
		new_pzcx /=np.sum(new_pzcx,axis=0,keepdims=True)

		est_pz = np.sum(new_pzcx*px[None,:],axis=1)
		est_pzcy = new_pzcx @ pxcy
		err_z = est_pz - pz

		# dual 
		dual_z += penalty*err_z

		# grad_z
		#grad_z = -beta*(np.log(pz)+1)-(dual_z + err_z*penalty)
		grad_z = gobj_pz(pz,new_pzcx,dual_z)
		mean_grad_z = grad_z*mask_z - np.mean(grad_z * mask_z)
		ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
		if ss_z ==0:
			if np.any(pz-mean_grad_z*1e-9<=0):
				gmask_z =pz>=1e-7
				mask_z = np.logical_and(gmask_z,mask_z)
				mean_grad_z =grad_z*mask_z - np.mean(grad_z*mask_z)
				ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
			elif np.any(pz-mean_grad_z*1e-9>=1.0):
				mask_z = np.zeros(pz.shape)
				mean_grad_z = np.zeros(pz.shape)
				ss_z = 0.0
			else:
				break
		arm_z = gd.armijoStepSize(pz,-mean_grad_z*mask_z,ss_z,ss_scale,1e-4,fobj_pz,gobj_pz,**{"pzcx":new_pzcx,"dual_z":dual_z})
		new_pz = pz - mean_grad_z * ss_z * mask_z + 1e-9
		new_pz/=np.sum(new_pz)
		err_z = est_pz - new_pz
		conv_z = 0.5* np.sum(np.fabs(err_z)) 
		if np.all(conv_z<convthres):
			conv_flag=True
			break
		else:
			pzcx = new_pzcx
			pz = new_pz
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {"niter":itcnt,"conv":conv_flag,"IZX":mizx,"IZY":mizy,'pzcx':pzcx}
	if record_flag:
		output_dict['record'] = record_mat
	return output_dict

def drsIBType1(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/ beta
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	# function objects
	fobj_pz = gd.ibType1PzFuncObj(px,gamma,penalty)
	fobj_pzcx = gd.ibType1PzcxFuncObj(px,pxcy,py,gamma,penalty)
	# gradient objects
	gobj_pz = gd.ibType1PzGradObj(px,gamma,penalty)
	gobj_pzcx = gd.ibType1PzcxGradObj(px,pxcy,py,pycx,gamma,penalty)
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)

	dual_z = np.zeros((nz))
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv = False
	while itcnt < maxiter:
		errz = pz - np.sum(pzcx*px[None,:],axis=1)
		pzcy = pzcx @ pxcy
		# function value, used for tracking
		# IB: (gamma-1) H(Z) -gamma H(Z|X) + H(Z|Y)
		record_mat[itcnt % record_mat.shape[0]] = (gamma-1)*np.sum(-pz*np.log(pz))-gamma*np.sum(-pzcx*px[None,:]*np.log(pzcx))\
													+np.sum(-pzcy*py[None,:]*np.log(pzcy))+np.sum(dual_z*errz)\
													+0.5*penalty*(np.linalg.norm(errz)**2)
		itcnt+=1 
		# drs relaxation step
		dual_drs_z = dual_z - (1-alpha)*penalty*errz

		errz = pz - np.sum(pzcx*px[None,:],axis=1)
		gd_eval_dict = {"pz":pz,"pzcx":pzcx,"dual_z":dual_drs_z}
		grad_z = gobj_pz(**gd_eval_dict)
		mean_grad_z = grad_z - np.mean(grad_z)
		ss_z = gd.validStepSize(pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z == 0:
			break
		# armijo
		arm_z = gd.armijoStepSize(pz,-mean_grad_z,ss_z,ss_scale,1e-4,fobj_pz,gobj_pz,**{"pzcx":pzcx,"dual_z":dual_drs_z})
		if arm_z == 0:
			arm_z = ss_z
		new_pz = pz - arm_z * mean_grad_z
		# solve: (gamma-1)H(Z)
		errz = new_pz - np.sum(pzcx*px[None,:],axis=1)
		dual_z = dual_drs_z+ penalty*errz
		# solve -gamma H(Z|X) + H(Z|Y)

		gd_eval_dict["pz"] = new_pz
		grad_x = gobj_pzcx(**gd_eval_dict)
		mean_grad_x = grad_x - np.mean(grad_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_grad_x,ss_init,ss_scale)
		if ss_x == 0:
			break
		arm_x = gd.armijoStepSize(pzcx,-mean_grad_x,ss_x,ss_scale,1e-4,fobj_pzcx,gobj_pzcx,**{"pz":new_pz,"dual_z":dual_z})
		if arm_x == 0:
			arm_x = ss_x
		new_pzcx = pzcx - arm_x * mean_grad_x
		errz = new_pz - np.sum(new_pzcx*px[None,:],axis=1)
		dtvz = 0.5* np.sum(np.fabs(errz),axis=0)
		if np.all(np.array(dtvz<convthres)):
			conv = True
			break
		else:
			pzcx = new_pzcx
			pz = new_pz
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,'IZY':mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict

def drsIBType1MeanSub(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/ beta
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	# function objects
	fobj_pz = gd.ibType1PzFuncObj(px,gamma,penalty)
	fobj_pzcx = gd.ibType1PzcxFuncObj(px,pxcy,py,gamma,penalty)
	# gradient objects
	gobj_pz = gd.ibType1PzGradObj(px,gamma,penalty)
	gobj_pzcx = gd.ibType1PzcxGradObj(px,pxcy,py,pycx,gamma,penalty)
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	#pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)

	dual_z = np.zeros((nz))
	# masks
	mask_z = np.ones(pz.shape)
	mask_x = np.ones(pzcx.shape)

	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv_flag = False
	while itcnt < maxiter:
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		err_z = pz - est_pz
		est_pzcy = pzcx @ pxcy
		record_mat[itcnt % record_mat.shape[0]] = (gamma-1)*np.sum(-pz*np.log(pz)) - gamma*np.sum(-pzcx*px[None,:]*np.log(pzcx))\
												 +np.sum(-est_pzcy*py[None,:]*np.log(est_pzcy)) + np.sum(dual_z *err_z) + 0.5 * penalty*np.linalg.norm(err_z)
		itcnt +=1
		# 
		drs_dual_z = dual_z -(1-alpha)* penalty * err_z
		grad_z = gobj_pz(pz,pzcx,drs_dual_z)
		mean_grad_z = grad_z*mask_z - np.mean(grad_z*mask_z)
		ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
		if ss_z ==0:
			if np.any(pz-mean_grad_z*mask_z*1e-9<=0.0):
				gmask_eps_z = np.invert(pz<1e-9)
				mask_z = np.logical_and(gmask_eps_z,mask_z)
				mean_grad_z = grad_z *mask_z - np.mean(grad_z*mask_z)
				ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
			elif np.any(pz-mean_grad_z*mask_z*1e-9>=1.0):
				mask_z= np.zeros(pz.shape)
				mean_grad_z = grad_z * mask_z - np.mean(grad_z*mask_z)
				ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
			else:
				break
		arm_z = gd.armijoStepSize(pz,-mean_grad_z*mask_z,ss_z,ss_scale,1e-4,fobj_pz,gobj_pz,**{"pzcx":pzcx,"dual_z":drs_dual_z})
		if arm_z ==0:
			arm_z = ss_z
		raw_pz = pz -mean_grad_z*mask_z*arm_z + 1e-9
		new_pz = raw_pz /np.sum(raw_pz)

		err_z = new_pz - est_pz
		dual_z = drs_dual_z + penalty * err_z

		grad_x = gobj_pzcx(pzcx,new_pz,dual_z)
		mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
		if ss_x ==0:
			if np.any(pzcx - mean_grad_x*mask_x*1e-9<=0):
				gmask_eps_x = np.invert(pzcx<1e-9)
				mask_x = np.logical_and(gmask_eps_x,mask_x)
				mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			elif np.any(pzcx-mean_grad_x*mask_x*1e-9>=1.0):
				bad_cols = np.any(pzcx-mean_grad_x*mask_x*1e-9,axis=0)
				mask_x[:,bad_cols] = 0
				mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			else:
				break
		arm_x = gd.armijoStepSize(pzcx,-mean_grad_x*mask_x,ss_x,ss_scale,1e-4,fobj_pzcx,gobj_pzcx,**{"pz":new_pz,"dual_z":dual_z})
		if arm_x ==0:
			arm_x = ss_x
		raw_pzcx = pzcx - mean_grad_x * mask_x * arm_x + 1e-9
		new_pzcx = raw_pzcx/ np.sum(raw_pzcx,axis=0,keepdims=True)
		err_z = new_pz -np.sum(new_pzcx * px[None,:],axis=1)
		conv_z = 0.5 * np.sum(np.fabs(err_z))
		if np.all(conv_z < convthres):
			conv_flag=True
			break
		else:
			pz = new_pz
			pzcx = new_pzcx
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv_flag,'IZX':mizx,'IZY':mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict

def drsIBType2(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/ beta
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)
	# function and gradient objects
	fobj_pzcx = gd.ibType2PzcxFuncObj(px,pxcy,gamma,penalty)
	fobj_pz   = gd.ibType2PzFuncObj(px,gamma,penalty)
	fobj_pzcy = gd.ibType2PzcyFuncObj(py,pxcy,gamma,penalty)

	gobj_pzcx  = gd.ibType2PzcxGradObj(px,pxcy,gamma,penalty)
	gobj_pz    = gd.ibType2PzGradObj(px,gamma,penalty)
	gobj_pzcy  = gd.ibType2PzcyGradObj(py,pxcy,gamma,penalty)

	#pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)
	pzcy = pzcx @ pxcy

	dual_z = np.zeros((nz))
	dual_zy = np.zeros((nz,ny))
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv = False
	while itcnt < maxiter:
		err_z = np.sum(pzcx*px[None,:],axis=1) - pz
		err_zy = pzcx@pxcy - pzcy
		# loss = (gamma-1)H(Z) + H(Z|Y) + <dual_z,pzcx@px-pz> + <dual_zy,pzcx@pxcy-pzcy> + c/2|errz|^2+c/2|errzcy|^2
		record_mat[itcnt % record_mat.shape[0]] = (gamma-1) * np.sum(-pz*np.log(pz))\
												 +np.sum(-pzcy*py[None,:]*np.log(pzcy))\
												 -gamma * np.sum(-pzcx*px[None,:]*np.log(pzcx))\
												 +np.sum(dual_z*err_z)+np.sum(dual_zy*err_zy)\
												 +penalty*0.5 * (np.linalg.norm(err_z)**2+np.linalg.norm(err_zy)**2)
		itcnt+=1
		# loss = -gamma H(Z|X) + <dual_z,err_z> + <dual_zy, err_zy> + ....
		grad_x = gobj_pzcx(pzcx,pz,pzcy,dual_z,dual_zy)
		mean_gx = grad_x - np.mean(grad_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_gx,ss_init,ss_scale)
		if ss_x ==0:
			break
		arm_x = gd.armijoStepSize(pzcx,-mean_gx,ss_x,ss_scale,1e-4,fobj_pzcx,gobj_pzcx,**{"pz":pz,"pzcy":pzcy,"dual_z":dual_z,"dual_zcy":dual_zy})
		if arm_x == 0:
			arm_x = ss_x
		new_pzcx = pzcx - mean_gx * ss_x
		# relaxation step
		err_z = np.sum(new_pzcx * px[None,:],axis=1) - pz
		err_zy = new_pzcx@pxcy - pzcy
		relax_z = dual_z - (1-alpha)*penalty*err_z
		relax_zy = dual_zy - (1-alpha)*penalty*err_zy
		# gradz and grad_y 
		# loss= (gamma-1) H(Z) + H(Z|Y)
		grad_z = gobj_pz(pz,new_pzcx,relax_z)
		mean_gz = grad_z - np.mean(grad_z)
		grad_y =gobj_pzcy(pzcy,new_pzcx,relax_zy)
		mean_gy = grad_y - np.mean(grad_y,axis=0)
		# joint stepsize selection
		ss_z = gd.validStepSize(pz,-mean_gz,ss_init,ss_scale)
		if ss_z ==0:
			break
		arm_z= gd.armijoStepSize(pz,-mean_gz,ss_z,ss_scale,1e-4,fobj_pz,gobj_pz,**{"pzcx":new_pzcx,"dual_z":relax_z})
		if arm_z == 0:
			arm_z = ss_z
		ss_y = gd.validStepSize(pzcy,-mean_gy,arm_z,ss_scale)
		if ss_y ==0:
			break
		arm_y = gd.armijoStepSize(pzcy,-mean_gy,ss_y,ss_scale,1e-4,fobj_pzcy,gobj_pzcy,**{"pzcx":new_pzcx,"dual_zcy":relax_zy})
		if arm_y == 0:
			arm_y = ss_y
		new_pz = pz - mean_gz * arm_y
		new_pzcy = pzcy - mean_gy * arm_y
		# dual update
		err_z = np.sum(new_pzcx * px[None,:],axis=1) - new_pz
		err_zy = new_pzcx@ pxcy - new_pzcy

		dual_z = relax_z + penalty * err_z
		dual_zy = relax_zy + penalty * err_zy
		# convergence test
		conv_z = 0.5 * np.sum(np.fabs(err_z))
		conv_y = 0.5 * np.sum(np.fabs(err_zy),0)
		if np.all(np.array(conv_y<convthres)) and conv_z<convthres:
			conv = True
			break
		else:
			pzcx = new_pzcx
			pz = new_pz
			pzcy = new_pzcy
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,"IZY":mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict

def drsIBType2MeanSub(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/ beta
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)
	# function and gradient objects
	fobj_pzcx = gd.ibType2PzcxFuncObj(px,pxcy,gamma,penalty)
	fobj_pz   = gd.ibType2PzFuncObj(px,gamma,penalty)
	fobj_pzcy = gd.ibType2PzcyFuncObj(py,pxcy,gamma,penalty)

	gobj_pzcx  = gd.ibType2PzcxGradObj(px,pxcy,gamma,penalty)
	gobj_pz    = gd.ibType2PzGradObj(px,gamma,penalty)
	gobj_pzcy  = gd.ibType2PzcyGradObj(py,pxcy,gamma,penalty)
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	#pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)
	pzcy = pzcx @ pxcy

	dual_z = np.zeros((nz))
	dual_zy = np.zeros((nz,ny))
	# masks
	mask_x = np.ones(pzcx.shape)
	mask_z = np.ones(pz.shape)
	mask_y = np.ones(pzcy.shape)
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv_flag = False
	while itcnt < maxiter:
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		est_pzcy = pzcx@pxcy
		err_z = est_pz - pz
		err_zy = est_pzcy - pzcy
		record_mat[itcnt % record_mat.shape[0]] = (gamma-1) * np.sum(-pz*np.log(pz)) -gamma*np.sum(-pzcx*px[None,:]*np.log(pzcx))\
													+ np.sum(-pzcy*py[None,:]*np.log(pzcy))\
													+ np.sum(dual_z*err_z) + np.sum(dual_zy*err_zy)\
													+0.5 * penalty*(np.linalg.norm(err_z)**2+np.linalg.norm(err_zy)**2)
		itcnt +=1
		grad_x = gobj_pzcx(pzcx,pz,pzcy,dual_z,dual_zy)
		mean_grad_x = grad_x * mask_x - np.mean(grad_x*mask_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
		if ss_x ==0:
			if np.any(pzcx - mean_grad_x*mask_x*1e-9<=0.0):
				gmask_eps_x = np.invert(pzcx<1e-9)
				mask_x = np.logical_and(gmask_eps_x,mask_x)
				mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			elif np.any(pzcx-mean_grad_x*mask_x*1e-9>=1.0):
				bad_cols = np.any(pzcx-mean_grad_x*mask_x*1e-9,axis=0)
				mask_x[:,bad_cols]=0
				mean_grad_x= grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			else:
				break
		arm_x = gd.armijoStepSize(pzcx,-mean_grad_x*mask_x,ss_x,ss_scale,1e-4,fobj_pzcx,gobj_pzcx,**{"pz":pz,"pzcy":pzcy,"dual_z":dual_z,"dual_zcy":dual_zy})
		if arm_x ==0:
			arm_x = ss_x
		raw_pzcx = pzcx - mean_grad_x*mask_x*arm_x + 1e-9
		new_pzcx = raw_pzcx /np.sum(raw_pzcx,axis=0,keepdims=True)
		
		est_pz =np.sum(new_pzcx * px[None,:],axis=1) 
		est_pzcy = new_pzcx@pxcy
		err_z = est_pz - pz
		err_zy = est_pzcy - pzcy
		drs_dual_z = dual_z -(1-alpha)*penalty*err_z
		drs_dual_zy= dual_zy -(1-alpha)*penalty*err_zy

		grad_z = gobj_pz(pz,new_pzcx,drs_dual_z)
		mean_grad_z = grad_z*mask_z - np.mean(grad_z*mask_z)
		# for step size of z as well # this is a convex function, use valid suffices
		ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
		if ss_z ==0:
			if np.any(pz-mean_grad_z*mask_z*1e-9<=0.0):
				gmask_eps_z = np.invert(pz<1e-9)
				mask_z = np.logical_and(gmask_eps_z,mask_z)
				mean_grad_z = grad_z*mask_z - np.mean(grad_z*mask_z)
				ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
			elif np.any(pz-mean_grad_z*mask_z*1e-9>=1.0):
				mask_z = np.zeros(pz.shape)
				mean_grad_z = grad_z*mask_z - np.mean(grad_z*mask_z)
				ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
			else:
				break
		grad_y = gobj_pzcy(pzcy,new_pzcx,drs_dual_zy)
		mean_grad_y = grad_y*mask_y - np.mean(grad_y*mask_y,axis=0)
		ss_y = gd.validStepSize(pzcy,-mean_grad_y*mask_y,ss_z,ss_scale)
		if ss_y ==0:
			if np.any(pzcy-mean_grad_y*mask_y*1e-9<=0.0):
				gmask_eps_y = np.invert(pzcy<1e-9)
				mask_y = np.logical_and(gmask_eps_y,mask_y)
				mean_grad_y = grad_y*mask_y - np.mean(grad_y*mask_y,axis=0)
				ss_y = gd.validStepSize(pzcy,-mean_grad_y*mask_y,ss_z,ss_scale)
			elif np.any(pzcy-mean_grad_y*mask_y*1e-9>=1.0):
				bad_cols = np.any(pzcy-mean_grad_y*mask_y*1e-9>=0,axis=0)
				mask_y[:,bad_cols] = 0
				mean_grad_y = grad_y*mask_y - np.mean(grad_y*mask_y,axis=0)
				ss_y = gd.validStepSize(pzcy,-mean_grad_y*mask_y,ss_z,ss_scale)
			else:
				break
		arm_y = gd.armijoStepSize(pzcy,-mean_grad_y*mask_y,ss_y,ss_scale,1e-4,fobj_pzcy,gobj_pzcy,**{"pzcx":new_pzcx,"dual_zcy":drs_dual_zy})
		if arm_y == 0:
			arm_y = ss_y
		raw_pz = pz - mean_grad_z* mask_z * arm_y + 1e-9
		new_pz = raw_pz/np.sum(raw_pz)
		raw_pzcy = pzcy - mean_grad_y * mask_y * arm_y + 1e-9
		new_pzcy = raw_pzcy/np.sum(raw_pzcy,axis=0,keepdims=True)

		err_z = est_pz - new_pz
		err_zy = est_pzcy - new_pzcy
		dual_z = drs_dual_z + penalty * err_z
		dual_zy = drs_dual_zy + penalty * err_zy
		conv_z = 0.5 * np.sum(np.fabs(err_z))
		conv_zy= 0.5 * np.sum(np.fabs(err_zy),axis=0)
		if np.all(conv_z<convthres) and np.all(conv_zy<convthres):
			conv_flag = True
			break
		else:
			pzcx = new_pzcx
			pz = new_pz
			pzcy = new_pzcy
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv_flag,'IZX':mizx,"IZY":mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict

def admmIBLogSpaceType1(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/beta
	ss_fixed = kwargs['sinit']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	#pzcx = ut.initPzcx(det_init,1e-7,nz,nx,kwargs['seed'])
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-6,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)
	# log variables
	mlog_pz = -np.log(pz)
	mlog_pzcx = -np.log(pzcx)

	dual_z = np.zeros(pz.shape)
	itcnt =0
	record_mat = np.zeros((1,))
	if record_flag:
		record_mat = np.zeros((maxiter,))
	conv_flag = False
	while itcnt < maxiter:
		# update mlogpz
		mexp_mlog_pzcx= np.exp(-mlog_pzcx) # pz in probability space
		mexp_mlog_pz  = np.exp(-mlog_pz)
		tmp_pzcy = mexp_mlog_pzcx@pxcy
		est_pz = np.sum(mexp_mlog_pzcx * px[None,:],axis=1)
		est_mlog_pz = -np.log(est_pz)
		err_z = mlog_pz -est_mlog_pz
		record_mat[itcnt%record_mat.shape[0]]= (gamma-1)*np.sum(mexp_mlog_pz*mlog_pz) -gamma * np.sum(mexp_mlog_pzcx*px[None,:]*mlog_pzcx)\
												+np.sum(-tmp_pzcy*py[None,:]*np.log(tmp_pzcy)) + np.sum(err_z * dual_z) + 0.5 * penalty * np.linalg.norm(err_z)**2
		itcnt += 1
		dual_drs_z = dual_z - (1-alpha)*penalty*err_z
		# gradient z
		grad_z = (gamma-1) * mexp_mlog_pz*(1-mlog_pz) + dual_drs_z + penalty * err_z
		raw_mlog_pz = mlog_pz - grad_z * ss_fixed
		# projection
		# negative log space should be non-negative
		#raw_mlog_pz -= np.amin(raw_mlog_pz)
		prj_mlog_pz = np.amin(raw_mlog_pz)
		prj_mlog_pz = np.where(prj_mlog_pz<=0.0,prj_mlog_pz,0.0) # this should have one element only
		raw_mlog_pz = raw_mlog_pz - prj_mlog_pz
		mexp_mlog_pz = np.exp(-raw_mlog_pz) + 1e-9 # smoothing
		new_pz = mexp_mlog_pz/ np.sum(mexp_mlog_pz)
		new_mlog_pz = -np.log(new_pz)
		# dual update
		err_z = new_mlog_pz -est_mlog_pz
		dual_z = dual_drs_z + penalty * err_z
		# gradient x
		grad_x = -gamma * mexp_mlog_pzcx* (1-mlog_pzcx)*px[None,:] + mexp_mlog_pzcx*((1+np.log(tmp_pzcy))@pxy.T) \
				- (mexp_mlog_pzcx * px[None,:])*np.repeat(((dual_z + penalty * err_z)/est_pz)[:,None],repeats=nx,axis=1)
		raw_mlog_pzcx = mlog_pzcx - grad_x * ss_fixed
		# projection
		#raw_mlog_pzcx -= np.amin(raw_mlog_pzcx,axis=0,keepdims=True)
		prj_mlog_pzcx = np.amin(raw_mlog_pzcx,axis=0)
		prj_mlog_pzcx = np.where(prj_mlog_pzcx<=0.0,prj_mlog_pzcx,np.zeros((nx,)))
		raw_mlog_pzcx = raw_mlog_pzcx - prj_mlog_pzcx[None,:]
		mexp_mlog_pzcx = np.exp(-raw_mlog_pzcx) + 1e-9
		new_pzcx = mexp_mlog_pzcx/ np.sum(mexp_mlog_pzcx,axis=0,keepdims=True)
		new_mlog_pzcx = -np.log(new_pzcx)

		#final error
		est_pz = np.sum(new_pzcx * px[None,:],axis=1)
		conv_z = 0.5 * np.sum(np.fabs(new_pz - est_pz))
		if conv_z < convthres:
			conv_flag = True
			break
		else:
			mlog_pz = new_mlog_pz
			mlog_pzcx = new_mlog_pzcx
	pzcx = np.exp(-mlog_pzcx)
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {"pzcx":pzcx,"niter":itcnt, "conv":conv_flag, 'IZX':mizx,"IZY":mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict

def admmIBLogSpaceType2(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/ beta
	#ss_init = kwargs['sinit']
	#ss_scale = kwargs['sscale']
	ss_fixed = kwargs['sinit']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)
	#pzcx = ut.initPzcx(det_init,1e-7,nz,nx,kwargs['seed'])
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-6,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)
	pzcy = pzcx @ pxcy

	# variables
	mlog_pzcx = -np.log(pzcx)
	mlog_pz   = -np.log(pz)
	mlog_pzcy = -np.log(pzcy)

	dual_z = np.zeros((nz))
	dual_zy = np.zeros((nz,ny))
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv_flag = False
	while itcnt < maxiter:
		mexp_mlog_pzcx = np.exp(-mlog_pzcx)
		mexp_mlog_pz = np.exp(-mlog_pz)
		mexp_mlog_pzcy = np.exp(-mlog_pzcy)
		tmp_pzcy = mexp_mlog_pzcx@pxcy
		est_pz = np.sum(mexp_mlog_pzcx * px[None,:],axis=1)
		err_z =  -np.log(est_pz) -mlog_pz
		err_zy=  -np.log(tmp_pzcy) -mlog_pzcy
		record_mat[itcnt % record_mat.shape[0]] = -gamma * np.sum(mexp_mlog_pzcx*px[None,:]*mlog_pzcx)\
												 + (gamma-1) * np.sum(mexp_mlog_pz*mlog_pz)\
												 + np.sum(mexp_mlog_pzcy * mlog_pzcy*py[None,:])\
												 + np.sum(err_z * dual_z )+ np.sum(err_zy * dual_zy)\
												 + 0.5 * penalty * (np.linalg.norm(err_z)**2 + np.linalg.norm(err_zy)**2)
		itcnt += 1
		# error
		# convex gradient
		grad_x = -gamma * (mexp_mlog_pzcx*(1-mlog_pzcx) * px[None,:]) \
				+ ((dual_z + penalty * err_z)/est_pz)[:,None]*(mexp_mlog_pzcx*px[None,:])\
				+ mexp_mlog_pzcx * ( ((dual_zy + penalty * err_zy)/tmp_pzcy) @ pxcy.T)
		raw_mlog_pzcx = mlog_pzcx - grad_x * ss_fixed
		# projection
		#raw_mlog_pzcx -= np.amin(raw_mlog_pzcx,axis=0,keepdims=True)
		prj_mlog_pzcx = np.amin(raw_mlog_pzcx,axis=0)
		prj_mlog_pzcx = np.where(prj_mlog_pzcx<=0.0,prj_mlog_pzcx,np.zeros((nx,)))
		raw_mlog_pzcx = raw_mlog_pzcx - prj_mlog_pzcx[None,:]
		mexp_mlog_pzcx = np.exp(-raw_mlog_pzcx) + 1e-9
		new_pzcx = mexp_mlog_pzcx/np.sum(mexp_mlog_pzcx,axis=0,keepdims=True)
		# new update
		new_mlog_pzcx = -np.log(new_pzcx)
		# helper 
		est_pz = np.sum(new_pzcx * px[None,:],axis=1)
		est_pzcy = new_pzcx @ pxcy
		est_mlog_pzcy = -np.log(est_pzcy)
		est_mlog_pz = -np.log(est_pz)
		# new error
		err_z = est_mlog_pz - mlog_pz
		err_zy= est_mlog_pzcy -mlog_pzcy
		# drs update
		dual_drs_z = dual_z -(1-alpha)*penalty*err_z
		dual_drs_zy = dual_zy - (1-alpha)*penalty*err_zy
		# update pz pzcy
		mexp_mlog_pz = np.exp(-mlog_pz)
		mexp_mlog_pzcy = np.exp(-mlog_pzcy)
		grad_z = (gamma-1) * mexp_mlog_pz * (1-mlog_pz) - (dual_drs_z + penalty * err_z)
		raw_mlog_pz = mlog_pz - grad_z * ss_fixed
		#raw_mlog_pz -= np.amin(raw_mlog_pz)
		prj_mlog_pz = np.amin(raw_mlog_pz)
		prj_mlog_pz = np.where(prj_mlog_pz<=0.0,prj_mlog_pz,0.0)
		raw_mlog_pz = raw_mlog_pz - prj_mlog_pz
		mexp_mlog_pz = np.exp(-raw_mlog_pz)+1e-9
		new_pz = mexp_mlog_pz/np.sum(mexp_mlog_pz)
		new_mlog_pz = -np.log(new_pz)
		# update pzcy
		grad_y = mexp_mlog_pzcy*(1-mlog_pzcy) * py[None,:] - (dual_drs_zy + penalty * err_zy)
		raw_mlog_pzcy = mlog_pzcy - grad_y * ss_fixed
		#raw_mlog_pzcy -= np.amin(raw_mlog_pzcy,axis=0,keepdims=True)
		prj_mlog_pzcy = np.amin(raw_mlog_pzcy,axis=0)
		prj_mlog_pzcy = np.where(prj_mlog_pzcy<=0.0,prj_mlog_pzcy,np.zeros((ny,)))
		raw_mlog_pzcy = raw_mlog_pzcy - prj_mlog_pzcy[None,:]

		mexp_mlog_pzcy = np.exp(-raw_mlog_pzcy)+1e-9
		new_pzcy = mexp_mlog_pzcy/np.sum(mexp_mlog_pzcy,axis=0,keepdims=True)
		new_mlog_pzcy = -np.log(new_pzcy)

		# dual ascend
		err_z=  est_mlog_pz -new_mlog_pz
		err_zy = est_mlog_pzcy -new_mlog_pzcy
		dual_z = dual_drs_z + penalty * err_z
		dual_zy= dual_drs_zy+ penalty * err_zy
		# convergence
		pz_diff = est_pz - new_pz
		pzcy_diff = est_pzcy - new_pzcy
		conv_z= 0.5 * np.sum(np.abs(pz_diff))
		conv_zy = 0.5 * np.sum(np.abs(pzcy_diff),axis=0)
		if np.all(conv_z < convthres) and np.all(conv_zy < convthres):
			conv_flag = True
			break
		else:
			mlog_pzcx = new_mlog_pzcx
			mlog_pz = new_mlog_pz
			mlog_pzcy = new_mlog_pzcy
	# convert to simplex
	pzcx = np.exp(-mlog_pzcx)
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv_flag,'IZX':mizx,"IZY":mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict

## BA algorithm

def ibOrig(pxy,nz,beta,convthres,maxiter,**kwargs):
	# on IB, the initialization matters
	# use random start (*This is the one used for v2)
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)
	#pzcx = ut.initPzcx(det_init,1e-7,nz,nx,kwargs['seed'])
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)
	pzcy = pzcx @ pxcy
	pycz = ((pzcy * py[None,:])/pz[:,None]).T
	
	# ready to start
	itcnt = 0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv_flag = False
	while itcnt<maxiter:
		record_mat[itcnt % record_mat.shape[0]] = ut.calcMI(pzcx*px[None,:]) -beta * ut.calcMI(pycz*pz[None,:])
		# compute ib kernel
		new_pzcx= np.zeros((nz,nx))
		kl_oprod = np.expand_dims(1./pycz,axis=-1)@np.expand_dims(pycx,axis=1)
		kl_ker = np.sum(np.repeat(np.expand_dims(pycx,axis=1),nz,axis=1)*np.log(kl_oprod),axis=0)
		new_pzcx = np.diag(pz)@np.exp(-beta*kl_ker)
		# standard way, normalize to be valid probability.
		new_pzcx = new_pzcx@np.diag(1./np.sum(new_pzcx,axis=0))
		itcnt+=1
		# total variation convergence criterion
		diff = 0.5 * np.sum(np.fabs(new_pzcx-  pzcx),axis=0)
		if np.all(diff < convthres):
			# reaching convergence
			conv_flag = True
			break
		else:
			# update other probabilities
			pzcx = new_pzcx
			# NOTE: should use last step? or the updated one?
			pz = pzcx@px
			pzcy = pzcx@pxcy
			pycz = ((pzcy * py[None,:])/pz[:,None]).T
	# monitoring the MIXZ, MIYZ
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pycz*pz[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'IZX':mizx,'IZY':mizy,'conv':conv_flag}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict
'''
def admmIBLogFpx(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	assert alpha == 1.0
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	ss_fixed = kwargs['sinit']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	expand_pycx = np.repeat(np.expand_dims(pycx,axis=1),repeats=nz,axis=1) #(y,z,x)
	pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)
	pzcy = pzcx @ pxcy
	qycz = ((pzcy*py[None,:])/pz[:,None]).T

	mlog_pzcx = -np.log(pzcx)
	mlog_qycz = -np.log(qycz)
	dual_ycz = np.zeros(mlog_qycz.shape)
	itcnt =0 
	conv_flag = False
	record_mat = np.zeros((1,))
	if record_flag:
		record_mat = np.zeros((maxiter,))
	while itcnt < maxiter:
		# variational bound
		exp_mlog_pzcx = np.exp(-mlog_pzcx)
		exp_mlog_qycz = np.exp(-mlog_qycz)
		est_pz = np.sum(exp_mlog_pzcx*px[None,:],axis=1)
		est_mlog_pz = -np.log(est_pz)
		est_pzy = exp_mlog_pzcx@pxy
		est_pycz = (est_pzy/est_pz[:,None]).T
		est_mlog_pycz= (-np.log(est_pzy)+np.log(est_pz)[:,None]).T
		expand_mlog_qycz = np.repeat(mlog_qycz[...,None],repeats=nx,axis=2) #(y,z,x)
		err_ycz = est_mlog_pycz - mlog_qycz
		record_mat[itcnt%record_mat.shape[0]] = np.sum(-est_pz*np.log(est_pz)) -np.sum(-exp_mlog_pzcx*px[None,:]*mlog_pzcx)\
												+ beta*np.sum(-(est_pzy.T)*mlog_qycz)\
												+ np.sum(dual_ycz*err_ycz) + 0.5 * penalty*(np.linalg.norm(err_ycz)**2)
		itcnt+=1
		# using fixed point method
		kl_ker = np.sum(expand_pycx*(np.log(expand_pycx)+ expand_mlog_qycz ),axis=0) # (z,x)
		expand_est_pycz = np.repeat(np.expand_dims(est_pycz,axis=2),repeats=nx,axis=2) # (y,z,x)
		expand_penalty = (expand_pycx/expand_est_pycz)-1
		expand_err_ycz = np.repeat(np.expand_dims(dual_ycz+penalty* err_ycz,axis=2),repeats=nx,axis=2)

		penalty_expo_x = np.sum(expand_err_ycz*expand_penalty,axis=0)/(est_pz[:,None])
		grad_x = (exp_mlog_pzcx*px[None,:]) * (np.repeat(-est_mlog_pz[:,None],repeats=nx,axis=1) + mlog_pzcx - beta* kl_ker + penalty_expo_x)
		raw_mlog_pzcx = mlog_pzcx - ss_fixed* grad_x
		raw_mlog_pzcx -= np.amin(raw_mlog_pzcx,axis=0)
		new_pzcx = np.exp(-raw_mlog_pzcx) + 1e-9
		# BA
		#new_pzcx = np.repeat(est_pz[:,None],repeats=nx,axis=1) * np.exp(-beta*kl_ker)
		new_pzcx /= np.sum(new_pzcx,axis=0,keepdims=True)
		new_mlog_pzcx = -np.log(new_pzcx)
		pzy = new_pzcx @ pxy
		pz = new_pzcx @ px
		new_qycz = (pzy/pz[:,None]).T
		# dual update
		est_pz = np.sum(new_pzcx*px[None,:],axis=1)
		est_pzy = new_pzcx @ pxy
		est_pycz = (est_pzy/est_pz[:,None]).T
		est_mlog_pycz = -np.log(est_pycz)
		err_ycz = est_mlog_pycz - mlog_qycz
		dual_ycz =  dual_ycz + penalty * err_ycz

		# update variational decoder
		# BA
		#penalty_expo_y = (1/penalty)*(dual_ycz - beta*(est_pzy.T) )
		#new_qycz = est_pycz * np.exp(-penalty_expo_y) + 1e-9
		new_qycz = est_pycz
		#grad_y= beta*est_pzy.T- dual_ycz - penalty * err_ycz
		#raw_mlog_qycz = mlog_qycz - ss_fixed*grad_y
		#raw_mlog_qycz -= np.amin(raw_mlog_qycz,axis=0)
		#new_qycz = np.exp(-raw_mlog_qycz)+1e-9
		new_qycz /= np.sum(new_qycz,axis=0,keepdims=True)
		new_mlog_qycz = -np.log(new_qycz)
		# err
		#err_ycz = est_mlog_pycz - new_mlog_qycz
		#conv_ycz = 0.5 * np.sum(np.fabs(err_ycz),axis=0)
		diff= 0.5*np.sum(np.fabs(exp_mlog_pzcx-new_pzcx),axis=0)
		if np.all(diff<convthres):
			conv_flag=True
			break
		else:
			mlog_pzcx = new_mlog_pzcx
			mlog_qycz = new_mlog_qycz
	pzcx = np.exp(-mlog_pzcx)
	pzcx /= np.sum(pzcx,axis=0,keepdims=True)
	pzy = pzcx@pxy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzy)
	output_dict = {'niter':itcnt,"conv":conv_flag,'IZX':mizx,'IZY':mizy,'pzcx':pzcx}
	if record_flag:
		output_dict['record'] = record_mat
	return output_dict
'''

def admmIBDec(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax'] # no support for relaxation
	assert alpha == 1.0
	#gamma = 1/ beta
	# use beta
	#ss_init = kwargs['sinit']
	#ss_scale = kwargs['sscale']
	#ss_fixed = kwargs['sinit']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	expand_pycx = np.repeat(np.expand_dims(pycx,axis=1),repeats=nz,axis=1) #(y,z,x)
	#pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)
	#pzcy = pzcx @ pxcy
	qycz = ((pzcx@pxy)/pz[:,None]).T
	dual_z= np.zeros(pz.shape)
	itcnt =0 
	conv_flag = False
	record_mat = np.zeros((1,))
	if record_flag:
		record_mat = np.zeros((maxiter,))
	while itcnt < maxiter:
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		#err_z = np.sum(est_pz*np.log(est_pz/pz))
		err_z = est_pz - pz
		est_pzy = pzcx@pxy
		record_mat[itcnt%record_mat.shape[0]] = np.sum(-pz*np.log(pz))-np.sum(-pzcx*px[None,:]*np.log(pzcx))+beta* np.sum(-est_pzy.T*np.log(qycz))
		itcnt +=1

		# fixed point
		expand_qycz = np.repeat(np.expand_dims(qycz,axis=-1),repeats=nx,axis=-1)
		kl_ker = np.sum(expand_pycx*np.log(expand_pycx/expand_qycz),axis=0)
		new_pzcx= np.exp(-beta*kl_ker-np.repeat((dual_z+penalty*err_z)[:,None],repeats=nx,axis=1))
		new_pzcx /=np.sum(new_pzcx,axis=0,keepdims=True)
		# dual_z
		est_pz = np.sum(new_pzcx*px[None,:],axis=1)
		#err_z = np.sum(est_pz*np.log(est_pz/pz))
		err_z = est_pz - pz
		dual_z += penalty*err_z
		# pz update
		new_pz = np.exp(-dual_z)
		new_pz /= np.sum(new_pz)

		# convergence test
		err_z = est_pz - new_pz
		conv_z = 0.5*np.sum(np.fabs(err_z))
		if np.all(conv_z<convthres):
			conv_flag=True
			break
		else:
			pz = new_pz
			pzcx = new_pzcx
			qycz = ((new_pzcx@pxy)/est_pz[:,None]).T
	pz = pzcx@px
	pzy = pzcx@pxy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzy)
	output_dict = {"niter":itcnt,"conv":conv_flag,"pzcx":pzcx,"IZX":mizx,"IZY":mizy}
	if record_flag:
		output_dict['record'] = record_mat
	return output_dict


def admmIBDecMs(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax'] # no support for relaxation
	assert alpha == 1.0
	gamma = 1/ beta
	# use beta
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	expand_pycx = np.repeat(np.expand_dims(pycx,axis=1),repeats=nz,axis=1) #(y,z,x)
	#pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	if 'load' in kwargs.keys():
		pzcx = kwargs['load']
	else:
		pzcx = ut.initPzcx(det_init,1e-5,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px[None,:],axis=1)
	#pzcy = pzcx @ pxcy
	qycz = ((pzcx@pxy)/pz[:,None]).T
	dual_z= np.zeros(pz.shape)
	mask_x = np.ones(pzcx.shape)
	mask_z = np.ones(pz.shape)
	itcnt =0 
	conv_flag = False
	record_mat = np.zeros((1,))
	if record_flag:
		record_mat = np.zeros((maxiter,))
	while itcnt< maxiter:
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		est_pzy = pzcx@pxy
		err_z = est_pz- pz

		record_mat[itcnt%record_mat.shape[0]] = gamma * np.sum(-pz*np.log(pz)) -gamma *np.sum(-pzcx*px[None,:]*np.log(pzcx))\
												-np.sum(-py*np.log(py)) + np.sum(-est_pzy.T*np.log(qycz))\
												+ np.sum(dual_z*err_z) + 0.5*penalty*np.linalg.norm(err_z)**2

		itcnt+=1 
		expand_qycz = np.repeat(qycz[...,None],repeats=nx,axis=2)
		grad_x = gamma*(np.log(pzcx)+1)*px[None,:] - np.sum(expand_pycx*np.log(expand_qycz),axis=0)*px[None,:]\
				+ np.repeat((dual_z+penalty*err_z)[:,None],repeats=nx,axis=1)
		mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0,keepdims=True)

		ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
		if ss_x == 0:
			if np.any(pzcx- mean_grad_x*mask_x * 1e-9 <=0):
				gmask_x = pzcx > 1e-8
				mask_x = np.logical_and(gmask_x, mask_x)
				mean_grad_x = grad_x * mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			elif np.any(pzcx-mean_grad_x*mask_x*1e-9>=1.0):
				bad_cols= np.any(pzcx- mean_grad_x*mask_x * 1e-9>=0,axis=0)
				mask_x[:,bad_cols] =0
				mean_grad_x = grad_x*mask_x - np.mean(grad_x*mask_x,axis=0)
				ss_x = gd.validStepSize(pzcx,-mean_grad_x*mask_x,ss_init,ss_scale)
			else:
				break
		new_pzcx = pzcx - ss_x * mean_grad_x * mask_x +1e-9
		new_pzcx = new_pzcx/np.sum(new_pzcx,axis=0,keepdims=True)

		est_pz = np.sum(new_pzcx*px[None,:],axis=1)
		err_z = est_pz - pz
		dual_z += penalty*err_z 

		grad_z = -gamma*(np.log(pz)+1) - (dual_z+penalty*err_z)
		mean_grad_z = grad_z *mask_z - np.mean(grad_z*mask_z)
		ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
		if ss_z ==0:
			if np.any(pz-mean_grad_z*mask_z *1e-9<=0):
				gmask_z = pz>1e-8
				mask_z = np.logical_and(gmask_z,mask_z)
				mean_grad_z = grad_z * mask_z - np.mean(grad_z * mask_z)
				ss_z = gd.validStepSize(pz,-mean_grad_z*mask_z,ss_init,ss_scale)
			elif np.any(pz-mean_grad_z*mask_z*1e-9>=1):
				mask_z =np.zeros(pz.shape)
				mean_grad_z = np.zeros(pz.shape)
				ss_z = ss_init
			else:
				break
		new_pz = pz - mean_grad_z *ss_z * mask_z + 1e-9
		new_pz = new_pz /np.sum(new_pz)

		err_z = est_pz - new_pz
		conv_z = 0.5*np.sum(np.fabs(err_z))
		if conv_z < convthres:
			conv_flag = True 
			break
		else:
			pz = new_pz
			pzcx = new_pzcx
			qycz = ((new_pzcx@pxy)/est_pz[:,None]).T
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcx@pxy)
	output_dict = {"niter":itcnt,"conv":conv_flag,'pzcx':pzcx,'IZX':mizx,"IZY":mizy}
	if record_flag:
		output_dict['record'] = record_mat
	return output_dict