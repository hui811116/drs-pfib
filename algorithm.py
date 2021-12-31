import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import sys
import gradient_descent as gd
import utils as ut


def drsPF(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	rs = RandomState(MT19937(SeedSequence(kwargs['seed'])))
	(nx,ny) = pxy.shape

	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy / py[None,:]
	pycx = (pxy / px[:,None]).T

	#sel_idx = rs.permutation(nx)
	# controlling the initial point
	pzcx = np.zeros((nz,nx))
	shuffle_zx = rs.permutation(nz)
	for idx, item in enumerate(shuffle_zx):
		pzcx[item,idx] = 1
	# smoothing 
	pzcx+= 2e-3
	# NOTE: nz<= nx always
	##
	#pzcx= rs.rand(nz,nx)
	#pzcx = pycx

	pzcx /= np.sum(pzcx,axis=0) # normalization
	pz = np.sum(pzcx*px,axis=1)
	pzcy = pzcx @ pxcy

	dual_y = np.zeros((nz,ny))
	itcnt =0
	conv = False
	while itcnt < maxiter:
		itcnt += 1
		#
		erry = pzcy - pzcx@pxcy
		# solve -beta H(Z|Y)
		grad_y = beta*(np.log(pzcy)+1)*py[None,:] + (dual_y+penalty*erry)
		mean_grad_y = grad_y - np.mean(grad_y)
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
	return {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,'IZY':mizy}


def drsIBType1(pxy,nz,beta,convthres,maxiter,**kwargs):
	penalty = kwargs['penalty']
	alpha = kwargs['relax']
	gamma = 1/ beta
	ss_init = kwargs['sinit']
	ss_scale = kwargs['sscale']
	rs = RandomState(MT19937(SeedSequence(kwargs['seed'])))
	(nx,ny) = pxy.shape

	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy / py[None,:]
	pycx = (pxy / px[:,None]).T

	#sel_idx = rs.permutation(nx)

	# random initialization
	#pzcx= rs.rand(nz,nx)

	# deterministic start
	pzcx = np.zeros((nz,nx))
	shuffle_zx = rs.permutation(nz)
	for idx, item in enumerate(shuffle_zx):
		pzcx[item,idx] = 1
	# smoothing 
	pzcx+= 2e-3

	pzcx /= np.sum(pzcx,axis=0)
	pz = np.sum(pzcx*px,axis=1)

	dual_z = np.zeros((nz))
	itcnt =0
	conv = False
	while itcnt < maxiter:
		itcnt += 1
		#
		errz = pz - np.sum(pzcx*px[None,:],axis=1)
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
	return {'pzcx':pzcx,'niter':itcnt,'conv':conv,'IZX':mizx,'IZY':mizy}
