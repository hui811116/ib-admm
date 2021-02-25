import numpy as np
import copy

def validStepSize(prob,update,init_stepsize,bk_beta):
	step_size = init_stepsize
	test = prob+init_stepsize * update
	it_cnt = 0
	while np.any(test>1.0) or np.any(test<0.0):
		if step_size <= 1e-9:
			return 0
		step_size = step_size*bk_beta
		test = prob + step_size * update
		it_cnt += 1
	return step_size

def armijoStepSize(prob,update,alpha,ss_beta,c1,obj_func,obj_grad,**kwargs):
	ss = alpha
	f_next = obj_func(prob+ss*update,**kwargs) 
	f_now = obj_func(prob,**kwargs)
	(now_grad,_) = obj_grad(prob,**kwargs)
	while f_next > f_now + c1*ss*np.sum(update*now_grad):
		if ss <= 1e-9:
			ss=0
			break
		ss *= ss_beta
		f_next = obj_func(prob+ss*update,**kwargs)
	return ss

def wolfeStepSize(prob,update,init_ss,ss_beta,ss_c1,obj_func,obj_grad,lambda_reg,**kwargs):
	# FIXME:
	# Currently only the sufficient descent condition implemented,
	# Should include the curvature condition as well (or strong curvature condition)
	# Current version is exclusively for second order Newton's method...
	ss = init_ss
	f_now = obj_func(prob,**kwargs)
	f_next= obj_func(prob+ss*update,**kwargs)

	(now_grad,grad_lambda) = obj_grad(prob,**kwargs)
	if not lambda_reg.size:
		raw_grad = now_grad + grad_lambda
		now_grad = raw_grad - lambda_reg
	while f_next + f_now + ss_c1*ss*np.sum(update*now_grad):
		if ss <= 1e-9:
			ss = 0
			break
		ss *= ss_beta
		f_next = obj_func(prob+ss*update,**kwargs)
	return 0