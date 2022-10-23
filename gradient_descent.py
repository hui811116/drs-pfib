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
def armijoStepSize(prob,update,alpha,ss_beta,c1,obj_func,obj_grad,**kwargs):
	ss = alpha
	f_next = obj_func(prob+ss*update,**kwargs) 
	f_now = obj_func(prob,**kwargs)
	now_grad = obj_grad(prob,**kwargs)
	now_grad = meanSubtractedGrad(now_grad)
	while f_next > f_now + c1*ss*np.sum(update*now_grad):
		if ss <= 1e-11:
			ss=0
			break
		ss *= ss_beta
		f_next = obj_func(prob+ss*update,**kwargs)
	return ss

# convenient gradient projection to probability simplex
def meanSubtractedGrad(grad,**kwargs):
	return grad - np.mean(grad,axis=0)


## for ADMM-IB type I
# ----------------------------------
def ibType1PzFuncObj(px,gamma,penalty):
	def genFuncObj(pz,pzcx,dual_z,**kwargs):
		err_z = pz - np.sum(pzcx * px[None,:],axis=1)
		return (gamma -1) * np.sum(-pz * np.log(pz)) + np.sum(dual_z * err_z) + 0.5 * penalty * np.linalg.norm(err_z)**2
	return genFuncObj
def ibType1PzcxFuncObj(px,pxcy,py,gamma,penalty):
	def genFuncObj(pzcx,pz,dual_z,**kwargs):
		err_z = pz - np.sum(pzcx * px[None,:],axis=1)
		pzcy = pzcx @ pxcy
		return -gamma * np.sum(-pzcx * px[None,:] * np.log(pzcx)) + np.sum(-pzcy * py[None,:]*np.log(pzcy)) + np.sum(dual_z*err_z)+0.5*penalty*np.linalg.norm(err_z)**2
	return genFuncObj
def ibType1PzGradObj(px,gamma,penalty):
	def genGradObj(pz,pzcx,dual_z,**kwargs):
		err_z = pz - np.sum(pzcx*px[None,:],axis=1)
		return -(gamma-1) * (np.log(pz)+1) +dual_z + penalty * err_z
	return genGradObj

def ibType1PzcxGradObj(px,pxcy,py,pycx,gamma,penalty):
	def genGradObj(pzcx,pz,dual_z,**kwargs):
		err_z = pz - np.sum(pzcx * px[None,:],axis=1)
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
		#return -np.sum(-pzcx * px[None,:] * np.log(pzcx))\
		#		 + np.sum(dual_z*err_z) + np.sum(dual_zcy * err_y)\
		#		 + 0.5*penalty * (np.linalg.norm(err_z)**2 + np.linalg.norm(err_y)**2)
	return genFuncObj

def ibType2PzFuncObj(px,gamma,penalty):
	def genFuncObj(pz,pzcx,dual_z,**kwargs):
		err_z = np.sum(pzcx * px[None,:],axis=1) - pz
		return (gamma-1) * np.sum(-pz * np.log(pz)) + np.sum(dual_z * err_z) + 0.5 * penalty * np.linalg.norm(err_z)**2
		#return (1-1/gamma) * np.sum(-pz * np.log(pz)) + np.sum(dual_z * err_z) + 0.5 * penalty * np.linalg.norm(err_z)**2
	return genFuncObj
def ibType2PzcyFuncObj(py,pxcy,gamma,penalty):
	def genFuncObj(pzcy,pzcx,dual_zcy,**kwargs):
		err_y = pzcx @ pxcy - pzcy
		return np.sum(-pzcy * py[None,:] * np.log(pzcy)) + np.sum(dual_zcy * err_y) + 0.5 * penalty * np.linalg.norm(err_y) ** 2
		#return (1/gamma)*np.sum(-pzcy * py[None,:] * np.log(pzcy)) + np.sum(dual_zcy * err_y) + 0.5 * penalty * np.linalg.norm(err_y) ** 2
	return genFuncObj

def ibType2PzcxGradObj(px,pxcy,gamma,penalty):
	def genGradObj(pzcx,pz,pzcy,dual_z,dual_zcy,**kwargs):
		err_z = np.sum(pzcx*px[None,:],axis=1) - pz
		err_y = pzcx @ pxcy - pzcy
		return gamma * (np.log(pzcx)+1)*px[None,:]\
				+ (dual_z+penalty * err_z)[:,None]@px[None,:]\
				+ (dual_zcy+penalty *err_y) @ pxcy.T
		#return (np.log(pzcx)+1)*px[None,:]\
		#		+ (dual_z+penalty * err_z)[:,None]@px[None,:]\
		#		+ (dual_zcy+penalty *err_y) @ pxcy.T
	return genGradObj

def ibType2PzGradObj(px,gamma,penalty):
	def genGradObj(pz,pzcx,dual_z,**kwargs):
		err_z = np.sum(pzcx * px[None,:],axis=1) - pz
		return -(gamma-1)*(np.log(pz)+1) - dual_z - penalty * err_z
		#return -(1-1/gamma)*(np.log(pz)+1) - dual_z - penalty * err_z
	return genGradObj
def ibType2PzcyGradObj(py,pxcy,gamma,penalty):
	def genGradObj(pzcy,pzcx,dual_zcy,**kwargs):
		err_y = pzcx @ pxcy - pzcy
		return -(np.log(pzcy)+1) * py[None,:] -dual_zcy - penalty * err_y
		#return -(1/gamma)*(np.log(pzcy)+1) * py[None,:] -dual_zcy - penalty * err_y
	return genGradObj
# -------------------------

# privacy funnel

def pfPzcxFuncObj(px,pxcy,beta,penalty):
	def genFuncObj(pzcx,pzcy,dual_y,**kwargs):
		err_y = pzcy - pzcx @ pxcy
		pz = np.sum(pzcx * px[None,:],axis=1)
		return (beta-1) * np.sum(-pz * np.log(pz)) + np.sum(-pzcx*px[None,:]*np.log(pzcx)) + np.sum(dual_y*err_y) + 0.5 * penalty*np.linalg.norm(err_y)**2
		#return (1-1/beta) * np.sum(-pz * np.log(pz)) + (1/beta)*np.sum(-pzcx*px[None,:]*np.log(pzcx)) + np.sum(dual_y*err_y) + 0.5 * penalty*np.linalg.norm(err_y)**2
	return genFuncObj

def pfPzcyFuncObj(py,pxcy,beta,penalty):
	def genFuncObj(pzcy,pzcx,dual_y,**kwargs):
		err_y = pzcy - pzcx @ pxcy
		return -beta * np.sum(-pzcy * py[None,:] * np.log(pzcy)) + np.sum(dual_y * err_y) + 0.5 * penalty * np.linalg.norm(err_y)**2
		#return -np.sum(-pzcy * py[None,:] * np.log(pzcy)) + np.sum(dual_y * err_y) + 0.5 * penalty * np.linalg.norm(err_y)**2
	return genFuncObj

def pfPzcxGradObj(px,pxcy,beta,penalty):
	def genGradObj(pzcx,pzcy,dual_y,**kwargs):
		err_y = pzcy - pzcx @ pxcy
		pz = np.sum(pzcx* px[None,:],axis=1) 
		return (np.repeat((1-beta)*(np.log(pz)+1)[:,None],repeats=len(px),axis=1) -(np.log(pzcx)+1))*px[None,:] - (dual_y+penalty*err_y) @ pxcy.T
		#return (np.repeat((1/beta-1)*(np.log(pz)+1)[:,None],repeats=len(px),axis=1) -(1/beta)*(np.log(pzcx)+1))*px[None,:] - (dual_y+penalty*err_y) @ pxcy.T
	return genGradObj

def pfPzcyGradObj(py,pxcy,beta,penalty):
	def genGradObj(pzcy,pzcx,dual_y,**kwargs):
		err_y = pzcy - pzcx @ pxcy
		return beta * (np.log(pzcy)+1) * py[None,:] + dual_y + penalty * err_y
		#return (np.log(pzcy)+1) * py[None,:] + dual_y + penalty * err_y
	return genGradObj

# var privacy funnel
def pfVarPzcxFuncObj(px,pxcy,py,beta,penalty):
	def genFuncObj(pzcx,pz,dual_z,qxcz,**kwargs):
		est_pzcy = pzcx@pxcy
		err_z = np.sum(pzcx*px[None,:],axis=1) - pz
		return -beta*np.sum(-est_pzcy*py[None,:]*np.log(est_pzcy))-np.sum(pzcx*px[None,:]*np.log(qxcz.T)) + np.sum(dual_z*err_z)+0.5*penalty*(np.linalg.norm(err_z)**2)
	return genFuncObj

def pfVarPzFuncObj(px,beta,penalty):
	def genFuncObj(pz,pzcx,dual_z,**kwargs):
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		err_z = est_pz - pz
		return beta*np.sum(-pz*np.log(pz)) +np.sum(dual_z*err_z) + 0.5*penalty*(np.linalg.norm(err_z)**2)
	return genFuncObj
def pfVarPzcxGradObj(px,pxcy,py,beta,penalty):
	def genGradObj(pzcx,pz,dual_z,qxcz,**kwargs):
		est_pzcy = pzcx@pxcy
		err_z = np.sum(pzcx*px[None,:],axis=1) - pz
		return beta*(np.log(est_pzcy)+1)@(pxcy*py[None,:]).T - np.log(qxcz.T)*px[None,:] + np.repeat((dual_z+penalty*err_z)[:,None],repeats=len(px),axis=1)*px[None,:]
	return genGradObj

def pfVarPzGradObj(px,beta,penalty):
	def genGradObj(pz,pzcx,dual_z,**kwargs):
		est_pz = np.sum(pzcx*px[None,:],axis=1)
		err_z = est_pz - pz
		return -beta*(np.log(pz)+1) - (dual_z + penalty*err_z)
	return genGradObj
