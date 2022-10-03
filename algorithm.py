import numpy as np
#from numpy.random import MT19937
#from numpy.random import RandomState, SeedSequence
import sys
import gradient_descent as gd
import utils as ut
import copy

def drsPF(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	pzcx = ut.initPzcx(det_init,1e-3,nz,nx,kwargs['seed'])
	# NOTE: nz<= nx always
	##
	pz = np.sum(pzcx*px,axis=1)
	pzcy = pzcx @ pxcy

	dual_y = np.zeros((nz,ny))
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv = False
	while itcnt < maxiter:
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
		new_pz = np.sum(new_pzcx * px[None,:],axis=1)

		erry = new_pzcy - new_pzcx@pxcy
		dual_y = dual_drs_y + penalty*erry
		dtvy = 0.5* np.sum(np.fabs(erry),axis=0)
		if np.all(np.array(dtvy<convthres)):
			conv = True
			break
		else:
			pzcx = new_pzcx
			pzcy = new_pzcy
			pz = new_pz
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,'IZY':mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict

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

	pzcx = ut.initPzcx(det_init,1e-3,nz,nx,kwargs['seed'])
	# call the function value objects
	# NOTE: nz<= nx always
	pz = np.sum(pzcx*px,axis=1)
	pzcy = pzcx @ pxcy

	dual_y = np.zeros((nz,ny))
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv = False
	while itcnt < maxiter:
		erry = pzcy - pzcx@pxcy
		# function value: (beta-1) H(Z) -beta H(Z|Y) + H(Z|X)
		record_mat[itcnt%record_mat.shape[0]] = (beta-1) * np.sum(-pz*np.log(pz)) -beta*np.sum(-pzcy*py[None,:]*np.log(pzcy))\
							+np.sum(-pzcx*px[None,:]*np.log(pzcx)) + np.sum(dual_y*erry)\
							+0.5*penalty* (np.linalg.norm(erry)**2)
		itcnt += 1
		# solve -beta H(Z|Y)
		#grad_y = beta*(np.log(pzcy)+1)*py[None,:] + (dual_y+penalty*erry)
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
		#grad_x = (1-beta)*(np.log(pz)+1)[:,None]*px[None,:]-(np.log(pzcx)+1)*px[None,:]-(dual_drs_y+penalty*erry)@pxcy.T
		grad_x = gobj_pzcx(pzcx,new_pzcy,dual_drs_y)
		mean_grad_x = grad_x-np.mean(grad_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_grad_x,ss_init,ss_scale)
		if ss_x == 0:
			break
		arm_x = gd.armijoStepSize(pzcx,-mean_grad_x,ss_x,ss_scale,1e-4,fobj_pzcx,gobj_pzcx,**{"pzcy":new_pzcy,"dual_y":dual_drs_y})
		if arm_x == 0:
			arm_x = ss_x
		new_pzcx = pzcx - ss_x * mean_grad_x
		new_pz = np.sum(new_pzcx * px[None,:],axis=1)

		erry = new_pzcy - new_pzcx@pxcy
		dual_y = dual_drs_y + penalty*erry
		dtvy = 0.5* np.sum(np.fabs(erry),axis=0)
		if np.all(np.array(dtvy<convthres)):
			conv = True
			break
		else:
			pzcx = new_pzcx
			pzcy = new_pzcy
			pz = new_pz
	pzcy = pzcx @ pxcy
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,'IZY':mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict

def pfLogSpace(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	#ss_init = kwargs['sinit']
	#ss_scale = kwargs['sscale']
	ss_fixed = kwargs['sinit']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	pzcx = ut.initPzcx(det_init,1e-7,nz,nx,kwargs['seed'])
	# call the function value objects
	# NOTE: nz<= nx always
	pz = np.sum(pzcx*px,axis=1)
	pzcy = pzcx @ pxcy
	# minus log variables
	mlog_pzcx = -np.log(pzcx)
	#mlog_pz = -np.log(pz)
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
		err_y = mlog_pzcy + np.log(est_pzcy)
		# 
		tmp_pz = np.sum(exp_mlog_pzcx*px[None,:],axis=1)
		record_mat[itcnt % record_mat.shape[0]] = -beta * np.sum(exp_mlog_pzcy*mlog_pzcy*py[None,:]) + (beta-1) * np.sum(-tmp_pz*np.log(tmp_pz))\
												  +np.sum(exp_mlog_pzcx*px[None,:]*mlog_pzcx) + np.sum(dual_y*err_y) + 0.5 * penalty * np.linalg.norm(err_y)**2
		itcnt +=1
		# grad_y
		grad_y = -beta* exp_mlog_pzcy*(1-mlog_pzcy)*py[None,:] + dual_y+ penalty * err_y
		raw_mlog_pzcy = mlog_pzcy - ss_fixed * grad_y
		raw_mlog_pzcy -= np.amin(raw_mlog_pzcy,axis=0)
		exp_mlog_pzcy = np.exp(-raw_mlog_pzcy) + 1e-7 # smoothing
		new_mlog_pzcy = -np.log(exp_mlog_pzcy/np.sum(exp_mlog_pzcy,axis=0,keepdims=True))
		# grad_x
		err_y = new_mlog_pzcy + np.log(est_pzcy)
		grad_x = (exp_mlog_pzcx *px[None,:]) * ((beta-1)*(np.log(tmp_pz)[:,None]+1)+1-mlog_pzcx)\
				 - exp_mlog_pzcx * (((dual_y + penalty * err_y)/est_pzcy)@pxcy.T)
		raw_mlog_pzcx = mlog_pzcx - ss_fixed * grad_x
		raw_mlog_pzcx -= np.amin(raw_mlog_pzcx,axis=0)
		exp_mlog_pzcx = np.exp(-raw_mlog_pzcx) + 1e-7
		new_mlog_pzcx = -np.log(exp_mlog_pzcx/np.sum(exp_mlog_pzcx,axis=0,keepdims=True))

		# dual update
		exp_mlog_pzcx = np.exp(-new_mlog_pzcx)
		est_pzcy = exp_mlog_pzcx @ pxcy 
		err_y = new_mlog_pzcy + np.log(est_pzcy)
		dual_y += penalty * err_y
		# convergence 
		conv_y = 0.5 * np.sum(np.fabs(np.exp(-new_mlog_pzcy)-est_pzcy ),axis=0)
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

	pzcx = ut.initPzcx(det_init,1e-3,nz,nx,kwargs['seed'])
	pz = np.sum(pzcx*px,axis=1)

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
	fobj_pzcy = gd.ibType2PzcyFuncObj(py,pxcy,penalty)

	gobj_pzcx  = gd.ibType2PzcxGradObj(px,pxcy,gamma,penalty)
	gobj_pz    = gd.ibType2PzGradObj(px,gamma,penalty)
	gobj_pzcy  = gd.ibType2PzcyGradObj(py,pxcy,penalty)

	pzcx = ut.initPzcx(det_init,1e-3,nz,nx,kwargs['seed'])
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
		#grad_x = (gamma * (np.log(pzcx)+1) + (dual_z + penalty * err_z)[:,None]) * px[None,:]\
		#		+(dual_zy + penalty * err_zy)@pxcy.T
		mean_gx = grad_x - np.mean(grad_x,0)
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
		#grad_z = (gamma-1) * -(np.log(pz)+1) -relax_z - penalty * err_z
		grad_z = gobj_pz(pz,new_pzcx,relax_z)
		mean_gz = grad_z - np.mean(grad_z)
		#grad_y = -(np.log(pzcy)+1) * py[None,:] - relax_zy-penalty*err_zy
		grad_y =gobj_pzcy(pzcy,new_pzcx,relax_zy)
		mean_gy = grad_y - np.mean(grad_y,0)
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
		arm_y = gd.armijoStepSize(pzcy,-mean_gy,ss_init,ss_scale,1e-4,fobj_pzcy,gobj_pzcy,**{"pzcx":new_pzcx,"dual_zcy":relax_zy})
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

def admmIBLogSpaceType1(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/beta
	ss_fixed = kwargs['sinit']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	(nx,ny) = pxy.shape
	(px,py,pxcy,pycx) = ut.priorInfo(pxy)

	pzcx = ut.initPzcx(det_init,1e-7,nz,nx,kwargs['seed'])
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
		err_z = mlog_pz + np.log(est_pz)
		record_mat[itcnt%record_mat.shape[0]]= (gamma-1)*np.sum(mexp_mlog_pz*mlog_pz) -gamma * np.sum(mexp_mlog_pzcx*px[None,:]*mlog_pzcx)\
												+np.sum(-tmp_pzcy*py[None,:]*np.log(tmp_pzcy)) + np.sum(err_z * dual_z) + 0.5 * penalty * np.linalg.norm(err_z)**2
		itcnt += 1
		# gradient z
		grad_z = (gamma-1) * mexp_mlog_pz*(1-mlog_pz) + dual_z + penalty * err_z
		raw_mlog_pz = mlog_pz - grad_z * ss_fixed
		# projection
		raw_mlog_pz = raw_mlog_pz - np.amin(raw_mlog_pz)
		mexp_mlog_pz = np.exp(-raw_mlog_pz) + 1e-7 # smoothing
		mexp_mlog_pz/= np.sum(mexp_mlog_pz)
		new_mlog_pz = -np.log(mexp_mlog_pz)
		# dual update
		err_z = new_mlog_pz + np.log(est_pz)
		dual_z += penalty * err_z
		# gradient x
		grad_x = -gamma * mexp_mlog_pzcx* (1-mlog_pzcx)*px[None,:] + mexp_mlog_pzcx*((1+np.log(tmp_pzcy))@pxy.T) \
				-((dual_z + penalty * err_z)/est_pz)[:,None] * (mexp_mlog_pzcx * px[None,:])
		raw_mlog_pzcx = mlog_pzcx - grad_x * ss_fixed
		# projection
		raw_mlog_pzcx = raw_mlog_pzcx - np.amin(raw_mlog_pzcx,axis=0,keepdims=True)
		mexp_mlog_pzcx = np.exp(-raw_mlog_pzcx) + 1e-7
		mexp_mlog_pzcx/= np.sum(mexp_mlog_pzcx,axis=0,keepdims=True)
		new_mlog_pzcx = -np.log(mexp_mlog_pzcx)

		#final error
		mexp_mlog_pzcx = np.exp(-new_mlog_pzcx)
		est_pz = np.sum(mexp_mlog_pzcx * px[None,:],axis=1)
		err_z = new_mlog_pz + np.log(est_pz)
		conv_z = 0.5 * np.sum(np.abs(np.exp(-new_mlog_pz) - est_pz))
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
	pzcx = ut.initPzcx(det_init,1e-7,nz,nx,kwargs['seed'])
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
				+ mexp_mlog_pzcx * ( (dual_zy + penalty * err_zy)/tmp_pzcy ) @ pxcy.T
		raw_mlog_pzcx = mlog_pzcx - grad_x * ss_fixed
		# projection
		raw_mlog_pzcx -= np.amin(raw_mlog_pzcx,axis=0,keepdims=True)
		mexp_mlog_pzcx = np.exp(-raw_mlog_pzcx) + 1e-7
		# new update
		new_mlog_pzcx = -np.log(mexp_mlog_pzcx/np.sum(mexp_mlog_pzcx,axis=0,keepdims=True))
		# new error
		mexp_mlog_pzcx = np.exp(-new_mlog_pzcx)
		err_z = - np.log(np.sum(mexp_mlog_pzcx*px[None,:],axis=1)) - mlog_pz
		err_zy= - np.log(mexp_mlog_pzcx@pxcy) -mlog_pzcy

		# update pz pzcy
		mexp_mlog_pz = np.exp(-mlog_pz)
		mexp_mlog_pzcy =np.exp(-mlog_pzcy)
		grad_z = (gamma-1) * mexp_mlog_pz * (1-mlog_pz) - (dual_z + penalty * err_z)
		raw_mlog_pz = mlog_pz - grad_z * ss_fixed
		raw_mlog_pz -= np.amin(raw_mlog_pz)
		mexp_mlog_pz = np.exp(-raw_mlog_pz)+1e-7
		new_mlog_pz = -np.log(mexp_mlog_pz/np.sum(mexp_mlog_pz))
		# update pzcy
		grad_y = mexp_mlog_pzcy*(1-mlog_pzcy) * py[None,:] - (dual_zy + penalty * err_zy)
		raw_mlog_pzcy = mlog_pzcy - grad_y * ss_fixed
		raw_mlog_pzcy -= np.amin(raw_mlog_pzcy,axis=0,keepdims=True)
		mexp_mlog_pzcy = np.exp(-raw_mlog_pzcy)+1e-7
		new_mlog_pzcy = -np.log(mexp_mlog_pzcy/np.sum(mexp_mlog_pzcy,axis=0,keepdims=True))

		# dual ascend
		err_z=  - np.log(np.sum(mexp_mlog_pzcx*px[None,:],axis=1)) -new_mlog_pz
		err_zy = - np.log(mexp_mlog_pzcx@pxcy) -new_mlog_pzcy
		dual_z += penalty * err_z
		dual_zy+= penalty * err_zy
		# convergence
		pz_diff = np.exp(-new_mlog_pz) - np.sum(mexp_mlog_pzcx * px[None,:],axis=1)
		pzcy_diff = np.exp(-new_mlog_pzcy) - mexp_mlog_pzcx @ pxcy
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
