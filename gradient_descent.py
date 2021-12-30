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
