import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
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
	rs = RandomState(MT19937(SeedSequence(kwargs['seed'])))
	(nx,ny) = pxy.shape

	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy / py[None,:]
	pycx = (pxy / px[:,None]).T

	pzcx = np.zeros((nz,nx))
	#sel_idx = rs.permutation(nx)
	# controlling the initial point
	if det_init == 1:
		shuffle_zx = rs.permutation(nz)
		for idx, item in enumerate(shuffle_zx):
			pzcx[item,idx] = 1
		shuffle_rest = rs.randint(nz,size=(nx-nz))
		for nn in range(nx-nz):
			pzcx[shuffle_rest[nn],nz+nn]= 1 
		# smoothing 
		pzcx+= 5e-3
	else:
		pzcx= rs.rand(nz,nx)	

	# NOTE: nz<= nx always
	##

	pzcx /= np.sum(pzcx,axis=0) # normalization
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
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,'IZY':mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict


def drsIBType1(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/ beta
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	rs = RandomState(MT19937(SeedSequence(kwargs['seed'])))
	(nx,ny) = pxy.shape

	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy / py[None,:]
	pycx = (pxy / px[:,None]).T

	pzcx = np.zeros((nz,nx))
	# random initialization
	if det_init ==0:
		pzcx= rs.rand(nz,nx)
	else:
		# deterministic start
		shuffle_zx = rs.permutation(nz)
		for idx, item in enumerate(shuffle_zx):
			pzcx[item,idx] = 1
		shuffle_rest = rs.randint(nz,size=(nx-nz))
		for nn in range(nx-nz):
			pzcx[shuffle_rest[nn],nz+nn]= 1 
		# smoothing 
		pzcx+= 1e-3

	pzcx /= np.sum(pzcx,axis=0)
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
		# IB: (gamma-1) H(Z) -gamma H(Z|X) + H(Z|Y)
		record_mat[itcnt % record_mat.shape[0]] = (gamma-1)*np.sum(-pz*np.log(pz))-gamma*np.sum(pzcx*px[None,:]*np.log(pzcx))\
													+np.sum(pzcy*py[None,:]*np.log(pzcy))+np.sum(dual_z*errz)\
													+0.5*penalty*(np.linalg.norm(errz)**2)
		itcnt += 1
		
		dual_drs_z = dual_z - (1-alpha)*penalty*errz
		# solve: (gamma-1)H(Z)
		grad_z = (1-gamma)*(np.log(pz)+1) + dual_drs_z + penalty*errz
		mean_grad_z = grad_z - np.mean(grad_z)
		ss_z = gd.validStepSize(pz,-mean_grad_z,ss_init,ss_scale)
		if ss_z ==0:
			break
		new_pz = pz - mean_grad_z*ss_z
		
		errz = new_pz - np.sum(pzcx*px[None,:],axis=1)
		dual_z = dual_drs_z+ penalty*errz
		# solve -gamma H(Z|X) + H(Z|Y)
		pzcy = pzcx @ pxcy
		grad_x =  (gamma*(np.log(pzcx)+1) - (np.log(pzcy)+1)@pycx -(dual_z+penalty*errz)[:,None])*px[None,:]
		mean_grad_x = grad_x-np.mean(grad_x,axis=0)
		ss_x = gd.validStepSize(pzcx,-mean_grad_x,ss_init,ss_scale)
		if ss_x == 0:
			break
		new_pzcx = pzcx - ss_x * mean_grad_x
		
		errz = new_pz - np.sum(new_pzcx*px[None,:],axis=1)
		dtvz = 0.5* np.sum(np.fabs(errz),axis=0)
		if np.all(np.array(dtvz<convthres)):
			conv = True
			break
		else:
			pzcx = new_pzcx
			pzcy = new_pzcx @ pxcy
			pz = new_pz
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,'IZY':mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict

def drsIBType1FPX(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/ beta
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	det_init = kwargs['detinit']
	record_flag = kwargs['record']
	rs = RandomState(MT19937(SeedSequence(kwargs['seed'])))
	(nx,ny) = pxy.shape

	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy / py[None,:]
	pycx = (pxy / px[:,None]).T

	pzcx = np.zeros((nz,nx))
	# random initialization
	if det_init ==0:
		pzcx= rs.rand(nz,nx)
	else:
		# deterministic start
		shuffle_zx = rs.permutation(nz)
		for idx, item in enumerate(shuffle_zx):
			pzcx[item,idx] = 1
		shuffle_rest = rs.randint(nz,size=(nx-nz))
		for nn in range(nx-nz):
			pzcx[shuffle_rest[nn],nz+nn]= 1 
		# smoothing 
		pzcx+= 1e-3

	pzcx /= np.sum(pzcx,axis=0)
	pz = np.sum(pzcx*px,axis=1)

	dual_z = np.zeros((nz))
	itcnt =0
	record_mat = np.zeros((1))
	if record_flag:
		record_mat = np.zeros((maxiter))
	conv = False
	while itcnt < maxiter:
		itcnt += 1
		errz = pz - np.sum(pzcx*px[None,:],axis=1)
		pzcy = pzcx @ pxcy
		# IB: (gamma-1) H(Z) -gamma H(Z|X) + H(Z|Y)
		record_mat[itcnt % record_mat.shape[0]] = (gamma-1)*np.sum(-pz*np.log(pz))-gamma*np.sum(pzcx*px[None,:]*np.log(pzcx))\
													+np.sum(pzcy*py[None,:]*np.log(pzcy))+np.sum(dual_z*errz)\
													+0.5*penalty*(np.linalg.norm(errz)**2)
		dual_drs_z = dual_z - (1-alpha)*penalty*errz
		copy_pz = copy.deepcopy(pz)
		ccnt = 0
		while ccnt < 10:
			ccnt +=1
			errz = copy_pz - np.sum(pzcx*px[None,:],axis=1)
			grad_z = (1-gamma) * (np.log(copy_pz)+1) + dual_drs_z + penalty * errz
			mean_grad_z = grad_z - np.mean(grad_z)
			ss_z = gd.validStepSize(copy_pz,-mean_grad_z,ss_init,ss_scale)
			if ss_z == 0:
				break
			copy_pz = copy_pz - ss_z * mean_grad_z
		new_pz = copy_pz
		# solve: (gamma-1)H(Z)
		errz = new_pz - np.sum(pzcx*px[None,:],axis=1)
		dual_z = dual_drs_z+ penalty*errz
		# solve -gamma H(Z|X) + H(Z|Y)
		ccnt = 0
		copy_pzcx = copy.deepcopy(pzcx)
		while ccnt< 10:
			ccnt += 1
			err_z = new_pz - np.sum(copy_pzcx * px[None,:],axis=1)
			pzcy = copy_pzcx@ pxcy
			grad_x = (gamma * (np.log(copy_pzcx)+1) - (np.log(pzcy)+1)@pycx - (dual_z+penalty*err_z)[:,None])*px[None,:]
			mean_grad_x = grad_x - np.mean(grad_x,axis=0)
			ss_x = gd.validStepSize(copy_pzcx,-mean_grad_x,ss_init,ss_scale)
			if ss_x == 0:
				break
			copy_pzcx = copy_pzcx - mean_grad_x * ss_x
		new_pzcx = copy_pzcx	
		errz = new_pz - np.sum(new_pzcx*px[None,:],axis=1)
		dtvz = 0.5* np.sum(np.fabs(errz),axis=0)
		if np.all(np.array(dtvz<convthres)):
			conv = True
			break
		else:
			pzcx = new_pzcx
			pzcy = new_pzcx @ pxcy
			pz = new_pz
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
	rs = RandomState(MT19937(SeedSequence(kwargs['seed'])))
	(nx,ny) = pxy.shape

	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy / py[None,:]
	pycx = (pxy / px[:,None]).T

	pzcx = np.zeros((nz,nx))
	# random initialization
	if det_init ==0:
		pzcx= rs.rand(nz,nx)
	else:
		# deterministic start
		shuffle_zx = rs.permutation(nz)
		for idx, item in enumerate(shuffle_zx):
			pzcx[item,idx] = 1
		shuffle_rest = rs.randint(nz,size=(nx-nz))
		for nn in range(nx-nz):
			pzcx[shuffle_rest[nn],nz+nn]= 1 
		# smoothing 
		pzcx+= 1e-3

	pzcx /= np.sum(pzcx,axis=0)
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
		itcnt += 1
		err_z = np.sum(pzcx*px[None,:],axis=1) - pz
		err_zy = pzcx@pxcy - pzcy
		# loss = (gamma-1)H(Z) + H(Z|Y) + <dual_z,pzcx@px-pz> + <dual_zy,pzcx@pxcy-pzcy> + c/2|errz|^2+c/2|errzcy|^2
		record_mat[itcnt % record_mat.shape[0]] = (gamma-1) * np.sum(-pz*np.log(pz))\
												 +np.sum(-pzcy*py[None,:]*np.log(pzcy))\
												 +np.sum(dual_z*err_z)+np.sum(dual_zy*err_zy)\
												 +penalty*0.5 * (np.linalg.norm(err_z)**2+np.linalg.norm(err_zy)**2)
		# loss = -gamma H(Z|X) + <dual_z,err_z> + <dual_zy, err_zy> + ....
		grad_x = gamma * (np.log(pzcx)+1)*px[None,:] + (dual_z + penalty * err_z)[:,None] * px[None,:]\
				+(dual_zy + penalty * err_zy)@pxcy.T
		mean_gx = grad_x - np.mean(grad_x,0)
		ss_x = gd.validStepSize(pzcx,-mean_gx,ss_init,ss_scale)
		if ss_x ==0:
			break
		new_pzcx = pzcx - mean_gx * ss_x
		# relaxation step
		err_z = np.sum(new_pzcx * px[None,:],axis=1) - pz
		err_zy = new_pzcx@pxcy - pzcy
		relax_z = dual_z - (1-alpha)*penalty*err_z
		relax_zy = dual_zy - (1-alpha)*penalty*err_zy
		# gradz and grad_y 
		# loss= (gamma-1) H(Z) + H(Z|Y)
		grad_z = (gamma-1) * -(np.log(pz)+1) -relax_z - penalty * err_z
		mean_gz = grad_z - np.mean(grad_z)
		grad_y = -(np.log(pzcy)+1) * py[None,:] - relax_zy-penalty*err_zy
		mean_gy = grad_y - np.mean(grad_y,0)
		# joint stepsize selection
		ss_z = gd.validStepSize(pz,-mean_gz,ss_init,ss_scale)
		if ss_z ==0:
			break
		ss_y = gd.validStepSize(pzcy,-mean_gy,ss_z,ss_scale)
		if ss_y ==0:
			break
		new_pz = pz - mean_gz * ss_y
		new_pzcy = pzcy - mean_gy * ss_y
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
	mizx = ut.calcMI(pzcx*px[None,:])
	mizy = ut.calcMI(pzcy*py[None,:])
	output_dict = {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,"IZY":mizy}
	if record_flag:
		output_dict['record'] = record_mat[:itcnt]
	return output_dict
