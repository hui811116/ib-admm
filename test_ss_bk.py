import numpy as np
import matplotlib.pyplot as plt
import sys


# use simple negative entropy function to test the correctness of the implementation of backtracking stepsize selection
ndim = 3
pz = np.random.rand(ndim,)
pz/= np.sum(pz)
opt_pz = np.ones((ndim,))/ndim
opt_val = -np.log(ndim)

glb_bk_beta= 0.25
glb_bk_alpha = 0.4
conv_thres = 1e-4

def get_grad_obj():
	def grad_obj(pz):
		raw_grad = pz*np.log(pz)
		return raw_grad - np.mean(raw_grad)
	return grad_obj
def get_func_obj():
	def func_obj(pz):
		return np.sum(pz*np.log(pz))
	return func_obj

def validStepSize(prob,update,init_stepsize,bk_beta):
	step_size = init_stepsize
	test = prob+init_stepsize * update
	it_cnt = 0
	while np.any(test>1.0) or np.any(test<0.0):
		if it_cnt > 10000:
			print(step_size)
			sys.exit()
		if step_size <= 1e-9:
			return 0
		step_size = np.maximum(1e-9,step_size*bk_beta)
		test = prob + step_size * update
		it_cnt += 1
	return step_size


def backtrackStepSize(prob,update,init_stepsize,bk_beta,obj_func,obj_grad):
	# obtain bk_alpha from global
	step_size = init_stepsize
	ss_alpha = glb_bk_alpha
	val_0 = obj_func(prob)
	grad_0 = obj_grad(prob)
	step_0 = validStepSize(prob,update,init_stepsize,0.0125)
	if step_0 == 0:
		return 0
	val_tmp = obj_func(prob+update*step_0)
	while val_tmp > val_0+ss_alpha * step_0 *(np.sum(grad_0*update)):
		step_0 *= bk_beta
		if step_0 <= 1e-7:
			return 0
		val_tmp = obj_func(prob+update*step_0)
	return step_0

valFunc = get_func_obj()
gradEnt = get_grad_obj()
f_ss = backtrackStepSize(pz,-gradEnt(pz),1.0,glb_bk_beta,valFunc,gradEnt)
pz_new = pz - f_ss * gradEnt(pz);
fval = valFunc(pz)
itcnt = 0
maxiter = 100000
while (fval-opt_val > conv_thres) and (itcnt<maxiter):
	itcnt+=1
	s_grad = gradEnt(pz_new)
	next_ss = backtrackStepSize(pz_new,-s_grad,1.0,glb_bk_beta,valFunc,gradEnt)
	if next_ss == 0:
		break
	tmp_pz = pz_new -s_grad*next_ss
	tmp_val = valFunc(tmp_pz)
	pz_new = tmp_pz
print('optValue:{:.6f},getValue:{:.6f},iter:{}'.format(opt_val,valFunc(pz_new),itcnt))