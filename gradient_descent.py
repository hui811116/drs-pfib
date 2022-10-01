import numpy as np


def validStepSize(prob,update,init_stepsize,bk_beta):
	step_size = init_stepsize
	test = prob+init_stepsize * update
	#it_cnt = 0 # for debugging purpose only
	while np.any(test>1.0) or np.any(test<0.0):
		if step_size <= 1e-11:
			return 0
		step_size = step_size*bk_beta
		test = prob + step_size * update
		#it_cnt += 1
	return step_size

# now, we need armijo step-size selection or back-tracking step-size for faster convergence
def armijoStepSize(prob,update,ss_init,ss_scale,**kwargs):
	pass

# convenient gradient projection to probability simplex
def meanSubtractedGrad(grad,**kwargs):
	return grad - np.mean(grad,0)

## for privacy funnel
# ----------------------
def pfPzcyFuncObj(py,pxcy,beta,penalty):
	def genFuncObj(pzcy,pzcx,dual_y):
		err_y = pzcy - pzcx @ pxcy
		return beta * np.sum(pzcy * py[None,:] * np.log(pzcy)) + np.sum(dual_y * err_y) + 0.5 * penalty * np.linalg.norm(err_y)**2
	return genFuncObj

def pfPzcyGradObj(py,pxcy,beta,penalty):
	def genGradObj(pzcy,pzcx,dual_y):
		err_y = pzcy - pzcx @ pxcy
		return beta * (np.log(pzcy)+1) * py[None,:] + (dual_y + penalty * err_y)

# consists of H(Z) and H(Z|X)
def pfPzcxFuncObj(px,pxcy,beta,penalty):
	def genFuncObj(pzcx,pzcy,dual_y):
		tmp_pz = np.sum(pzcx * px[None,:],1)
		err_y = pzcy - pzcx @ pxcy
		return -(beta-1)*np.sum(tmp_pz * np.log(tmp_pz)) - np.sum(pzcx * px[None,:] * np.log(pzcx)) + np.sum(dual_y * err_y) + 0.5 * penalty * np.linalg.norm(err_y)**2
	return genFuncObj
def pfPzcxGradObj(px,pxcy,beta,penalty):
	def genGradObj(pzcx,pzcy,dual_y):
		tmp_pz = np.sum(pzcx * px[None,:],1)
		err_y = pzcy - pzcx @ pxcy
		return -(beta-1) * (np.log(pz)+1)[:,None]* px[None,:] - (np.log(pzcx)+1)*px[None,:]-(dual+penalty*err_y) @ pxcy.T
	return genGradObj
# ------------------------

## for ADMM-IB type I
# ----------------------------------
def ibType1PzFuncObj(px,gamma,penalty):
	def genFuncObj(pz,pzcx,dual_z,**kwargs):
		err_z = pz - np.sum(pzcx * px[None,:],axis=1)
		return (gamma -1) * np.sum(-pz * np.log(pz)) + np.sum(dual_z * err_z) + 0.5 * penalty * np.linalg.norm(err_z)**2
	return genFuncObj
def ibType1PzcxFuncObj(px,pxcy,py,gamma,penalty):
	def genFuncObj(pz,pzcx,dual_z,**kwargs):
		err_x = pz - np.sum(pzcx * px[None,:],axis=1)
		pzcy = pzcx @ pxcy
		return -gamma * np.sum(-pzcx * px[None,:] * np.log(pzcx)) + np.sum(-pzcy * py[None,:]*np.log(pzcy)) + np.sum(dual_z*err_x)+0.5*penalty*np.linalg.norm(err_x)**2
	return genFuncObj
def ibType1PzGradObj(px,gamma,penalty):
	def genGradObj(pz,pzcx,dual_z,**kwargs):
		err_z = pz - np.sum(pzcx*px[None,:],axis=1)
		return -(gamma-1) * (np.log(pz)+1) +dual_z + penalty * err_z
	return genGradObj

def ibType1PzcxGradObj(px,pxcy,py,gamma,penalty):
	def genGradObj(pz,pzcx,dual_z,**kwargs):
		err_x = pz - np.sum(pzcx * px[None,:],axis=1)
		pzcy = pzcx @ pxcy
		return (gamma * (np.log(pzcx)+1)  - (np.log(pzcy)+1) @ pycx  - (dual_z + penalty * err_z)[:,None]) * px[None,:]
	return genGradObj
# -------------------------
## for ADMM-IB type II
# -------------------------
def ibType2PzcxFuncObj(px,pxcy,gamma,penalty):
	def genFuncObj(pzcx,pz,pzcy,dual_z,dual_zcy,**kwargs):
		err_z = np.sum(pzcx *px[None,:],axis=1) - pz
		err_y = pzcx @ pxcy - pzcy
		return -gamma*np.sum(-pzcx * px[None,:] * np.log(pzcx))\
				 + np.sum(dual_z*err_z) + np.sum(dual_zcy * err_y)\
				 + 0.5*penalty * (np.linalg.norm(err_z)**2 + np.linalg.norm(err_y)**2)
	return genFuncObj
'''
def ibType2QFuncObj(px,py,pxcy,gamma,penalty):
	def genFuncObj(pzcx,pz,pzcy,dual_z,dual_zcy,**kwargs):
		err_z = np.sum(pzcx * px[None,:],axis=1) - pz
		err_y = pzcx @ pxcy - pzcy
		return (gamma-1)*np.sum(-pz * np.log(pz)) + np.sum(-pzcy * py[None,:] * np.log(pzcy) )\
				+ np.sum(dual_z * err_z) + np.sum(dual_zcy * err_y)\
				+ 0.5*penalty * ( np.linalg.norm(err_z)**2 + np.linalg.norm(err_y)**2)
	return genFuncObj
'''
def ibType2PzFuncObj(px,gamma,penalty):
	def genFuncObj(pzcx,pz,dual_z,**kwargs):
		err_z = np.sum(pzcx * px[None,:],axis=1) - pz
		return (gamma-1) * np.sum(-pz * np.log(pz)) + np.sum(daul_z * err_z) + 0.5 * penalty * np.linalg.norm(err_z)**2
	return genFuncObj
def ibType2PzcyFuncObj(py,pxcy,penalty):
	def genFuncObj(pzcx,pzcy,dual_zcy,**kwargs):
		err_y = pzcx @ pxcy - pzcy
		return np.sum(-pzcy * py[None,:] * np.log(pzcy)) + np.sum(dual_zcy * err_y) + 0.5 * penalty * np.linalg.norm(err_y) ** 2

def ibType2PzcxGradObj(px,pxcy,gamma,penalty):
	def genGradObj(pzcx,pz,pzcy,dual_z,dual_zcy,**kwargs):
		err_z = np.sum(pzcx*px[None,:],axis=1) - pz
		err_y = pzcx @ pxcy - pzcy
		return gamma * (np.log(pzcx)+1)*px[None,:]\
				+ (dual_z+penalty * err_z)[:,None]@px[None,:]\
				+ (dual_zcy+penalty *err_y) @ pxcy.T
	return genGradObj
# actually, this has two gradients...
'''
def ibType2QGradObj(px,pxcy,gamma,penalty):
	def genGradObj(pzcx,pz,pxcy,dual_z,dual_zcy,**kwargs):
		err_z = np.sum(pzcx*px[None,:],axis=1) - pz
		err_y = pzcx @ pxcy - pzcy
		# two gradients
		grad_z = -(gamma-1)*(np.log(pz)+1)- dual_z - penalty * err_z
		grad_y = -(np.log(pzcy)+1) * py[None,:] - dual_zcy - penalty * err_y
		return grad_z,grad_y
	return genGradObj
'''
def ibType2PzGradObj(px,gamma,penalty):
	def genGradObj(pzcx,pz,dual_z,**kwargs):
		err_z = np.sum(pzcx * px[None,:],axis=1) - pz
		return -(gamma-1)*(np.log(pz)+1) - dual_z - penalty * err_z
	return genGradObj
def ibType2PzcyGradObj(py,pxcy,penalty):
	def genGradObj(pzcx,pzcy,dual_zcy,**kwargs):
		err_y = pzcx @ pxcy - pzcy
		return -(np.log(pzcy)+1) * py[None,:] -dual_zcy - penalty * err_y
	return genGradObj
# -------------------------