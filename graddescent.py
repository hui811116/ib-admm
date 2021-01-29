import numpy as np

def validStepSize(prob,update,init_stepsize,bk_beta):
	step_size = init_stepsize
	test = prob+init_stepsize * update
	it_cnt = 0
	while np.any(test>1.0) or np.any(test<0.0):
		if step_size <= 1e-9:
			return 0
		step_size = np.maximum(1e-9,step_size*bk_beta)
		test = prob + step_size * update
		it_cnt += 1
	return step_size

def backtrackStepSize(prob,update,init_stepsize,bk_alpha,bk_beta,obj_func,obj_grad,**kwargs):
	# obtain bk_alpha from global
	step_size = init_stepsize
	ss_alpha = bk_alpha
	val_0 = obj_func(prob,**kwargs)
	grad_0 = obj_grad(prob,**kwargs)
	step_0 = validStepSize(prob,update,init_stepsize,0.5)
	if step_0 == 0:
		return 0
	val_tmp = obj_func(prob+update*step_0,**kwargs)
	while val_tmp> val_0+ss_alpha * step_0 *(np.sum(grad_0*update)):
		step_0 *= bk_beta
		if step_0 <= 1e-9:
			return 0
		val_tmp = obj_func(prob+update*step_0,**kwargs)
	return step_0