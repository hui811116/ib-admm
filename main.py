import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import sys
import os
import pickle
import copy

exp_name = 'class_3'

gbl_pycx = np.array([[0.7,0.3,0.075],[0.15,0.50,0.025],[0.15,0.20,0.90]])
gbl_px = np.ones(3,)/3
gbl_pxy = gbl_pycx*gbl_px[:,None].T


parm_qlevel = gbl_pxy.shape[1]
parm_conv_thres = 1e-5
parm_max_iter = 50000
beta_range = np.logspace(0,1.0,16)
num_avg = 5
result_dict = {'beta':beta_range,'methods':['orig','gd','alm_breg'],
			   'prob_xy':gbl_pxy,'max_iter':parm_max_iter,'avg_num':num_avg}
# gradient descent parameters
_bk_alpha = 0.45
_bk_beta = 0.5
_parm_grad_norm_thres = parm_conv_thres

# ALM parameters
_parm_c_init = 4.0
_parm_omega_init = 4.0

# for debugging
d_debug = False
dbg_beta = 1.88
dbg_max_iter = 50000

def ib_sol_log(pxy,beta,ib_dict,**kwargs):
	pzcx = ib_dict['prob_zcx']
	pycz = ib_dict['prob_ycz']
	niter = ib_dict['niter']
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pz = pzcx @ px
	mixz = calc_mi(pzcx,px)
	miyz = calc_mi(pycz,pz)
	ib_loss = mixz-beta * miyz
	logout = []
	for k,v in kwargs.items():
		if k == 'method':
			logout.append('Method:{:<8}'.format(v))
	logout.append('Loss:{:>6.4f}'.format(mixz-beta*miyz))
	logout.append('IXZ:{:>6.4f}'.format(mixz))
	logout.append('IYZ:{:>6.4f}'.format(miyz))
	logout.append('Iter:{:>8}'.format(niter))
	print(';'.join(logout))
	

def ib_orig(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pycx = np.transpose(np.diag(1./px)@pxy)
	pxcy = pxy@np.diag(1./py)
	# on IB, the initialization matters
	# use random start (*This is the one used for v2)
	sel_idx = np.random.permutation(nx)
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	pzcx = np.random.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx[:nz,:] = pycx
	pycz = pycx@ np.transpose(1/pz[:,None]*pzcx*px[None,:])
	
	# use random start
	#sel_idx = np.random.permutation(nx)
	#pycz = pycx[:,sel_idx[:qlevel]]
	#pz = px[sel_idx[:qlevel]]
	#pz /= np.sum(pz)
	#pzcx = np.ones((nz,nx))/nz
	
	# ready to start
	itcnt = 0
	while itcnt<max_iter:
		# compute ib kernel
		new_pzcx= np.zeros((nz,nx))
		kl_oprod = np.expand_dims(1./pycz,axis=-1)@np.expand_dims(pycx,axis=1)
		kl_ker = np.sum(np.repeat(np.expand_dims(pycx,axis=1),nz,axis=1)*np.log(kl_oprod),axis=0)
		new_pzcx = np.diag(pz)@np.exp(-beta*kl_ker)

		# standard way, normalize to be valid probability.
		new_pzcx = new_pzcx@np.diag(1./np.sum(new_pzcx,axis=0))
		itcnt+=1
		# total variation convergence criterion
		#diff = 0.5* np.sum(np.fabs(new_pzcx-pzcx))
		diff = 0.5 * np.sum(np.fabs(pz-new_pzcx@px)) # making the comparison the same
		if diff < conv_thres:
			# reaching convergence
			break
		else:
			# update other probabilities
			pzcx = new_pzcx
			# NOTE: should use last step? or the updated one?
			pz = pzcx@px
			pzcy = pzcx@pxcy
			pycz = np.transpose(np.diag(1./pz)@pzcy@np.diag(py))
	# monitoring the MIXZ, MIYZ
	mixz = calc_mi(pzcx,px)
	miyz = calc_mi(pycz,pz)
	return {'prob_zcx':pzcx,'prob_ycz':pycz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':True}

def calc_mi(pzcx,px):
	pz = pzcx@px
	inner_ker = (1./pz)[:,None]*pzcx
	maps = (inner_ker == 0)
	ker_val = np.where(maps,0.0,np.log(inner_ker+1e-7))
	return np.sum(pzcx*px[None,:]*ker_val) # avoiding overflow
'''
def prob_simplex_l2(py):
	nz = len(py)
	y_srt = np.flip(np.sort(py)) # along each pzcx given a x # flip to make it descending
	y_cum = np.cumsum(y_srt)
	_tmp_arr = 1./np.arange(1,nz+1)
	max_log = (y_srt + _tmp_arr * (1.0-y_cum))>0
	rho_idx = np.sum(max_log.astype('int32'))
	lamb_all = (1.0/rho_idx) * (1.0-y_cum[rho_idx-1])
	return np.maximum(py+np.repeat(np.expand_dims(lamb_all,axis=0),nz,axis=0),0)
'''
def ib_gd(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	pzcx = np.random.rand(nz,nx)
	pzcx = (1/np.sum(pzcx,axis=0))[None,:]*pzcx
	'''
	pz = pzcx @ px
	pzcy = pzcx @ pxcy
	pycz = np.transpose((1/pz)[:,None]*pzcy*py[None,:])
	'''
	sel_idx = np.random.permutation(nx)
	pycz = pycx[:,sel_idx[:qlevel]]
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	pzcx = np.ones((nz,nx))/nz
	# naive method
	_step_size = _global_step_size_init
	# ready to start
	itcnt = 0
	# gradient descent control
	flag_valid = True
	tmp_step = _step_size
	while (itcnt < max_iter) and flag_valid:
		# compute ib kernel
		nabla_iyz = np.transpose(np.log((1./py)[:,None]*pycz))@ pycx
		grad = (np.log((1./pz)[:,None]*pzcx)-beta*nabla_iyz )*px[None,:]
		# NOTE: 
		#     if all pzcx are non zero, then the ineq. eq. conditions gives
		#     lambda_i = -mean(grad)
		if np.any(grad != grad):
			flag_valid = False
			break
		zm_grad = grad - np.mean(grad,axis=0)
		tmp_step = validStepSize(pzcx,-grad,tmp_step,_bk_beta)
		if tmp_step == 0:
			flag_valid = False
			break
		new_pzcx = pzcx - tmp_step * zm_grad
		grad_norm = np.linalg.norm(zm_grad,axis=0)
		# theoretically, lambda >= 0. sometimes this is violated
		# ignore at the start of the gradient descent
		
		# in order to make sure the implementation is correct,
		# you should use naive method to calculate the steepest descent coefficient
		
		itcnt+=1
		dtv = 0.5 * np.sum(np.fabs(pz-new_pzcx@px))
		#pzcx_diff = 0.5 * np.sum(np.fabs(new_pzcx-pzcx))
		#if np.sum(grad_norm) < _parm_grad_norm_thres:
		if dtv< conv_thres:
			# termination condition reached
			# making the comparison at the same criterion
			break
		else:
			# updating step size
			pzcx = new_pzcx
			pz = pzcx @ px
			pzcy = pzcx@pxcy
			pycz = np.transpose((1./pz)[:,None]*pzcy*py[None,:])

	# monitoring the MIXZ, MIYZ
	mixz = calc_mi(pzcx,px)
	miyz = calc_mi(pycz,pz)
	if not flag_valid:
		mixz = 0
		miyz = 0
	return {'prob_zcx':pzcx,'prob_ycz':pycz,'niter':itcnt,'valid':flag_valid,'IXZ':mixz,'IYZ':miyz}

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

def backtrackStepSize(prob,update,init_stepsize,bk_beta,obj_func,obj_grad,**kwargs):
	# obtain bk_alpha from global
	step_size = init_stepsize
	ss_alpha = _bk_alpha
	val_0 = obj_func(prob,**kwargs)
	grad_0 = obj_grad(prob,**kwargs)
	step_0 = validStepSize(prob,update,init_stepsize,0.0125)
	if step_0 == 0:
		return 0
	val_tmp = obj_func(prob+update*step_0,**kwargs)
	while val_tmp> val_0+ss_alpha * step_0 *(np.sum(grad_0*update)):
		step_0 *= bk_beta
		if step_0 <= 1e-9:
			return 0
		val_tmp = obj_func(prob+update*step_0,**kwargs)
	return step_0

def getPzFuncObj(beta,px,pen_c,b_omega,use_breg=False):
	def val_obj(pz,pzcx,mu_z):
		pen_z =  pz-pzcx@px
		return (beta-1)*np.sum(pz*np.log(pz))+np.sum(mu_z*(pen_z))+pen_c/2*(np.sum(pen_z**2)) # ignore G, not relevent
	def val_obj_breg(pz,pzcx,mu_z,pz_last):
		pen_z = pz -pzcx@px
		return (beta-1)*np.sum(pz*np.log(pz))+np.sum(mu_z*(pen_z))+pen_c/2*(np.sum(pen_z**2)) \
				+b_omega*np.sum(pz*np.log(pz/pz_last))
	if use_breg:
		return val_obj_breg
	else:
		return val_obj

def getPzGradObj(beta,px, pen_c,b_omega,use_breg=False):
	def grad_obj(pz,pzcx,mu_z):
		pen_z = pz - pzcx@px
		raw_grad = (beta-1)*(np.log(pz)+1)+mu_z+pen_c*pen_z
		return raw_grad-np.mean(raw_grad)
	def grad_obj_breg(pz,pzcx,mu_z,pz_last):
		pen_z = pz - pzcx@px
		raw_grad = (beta-1)*(np.log(pz)+1)+mu_z+pen_c*pen_z+b_omega*(np.log(pz/pz_last))
		return raw_grad - np.mean(raw_grad)
	if use_breg:
		return grad_obj_breg
	else:
		return grad_obj

def getPzcxFuncObj(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pz,pzcx,mu_z):
		pen_z = pz-pzcx@px
		pzcy = pzcx@pxcy
		return np.sum(pzcx*np.log(pzcx)*px[None,:])-beta*np.sum(pzcy*np.log(pzcy)*py[None,:])\
				+np.sum(mu_z*pen_z)+pen_c*np.sum(pen_z**2)
	return val_obj
		
def getPzcxGradObj(beta,px,pxcy,pycx,pen_c):
	def grad_obj(pz,pzcx,mu_z):
		pen_z = pz-pzcx@px
		pzcy = pzcx@pxcy
		raw_grad = (np.log(pzcx)+1-beta+beta*np.transpose(pycx.T*np.log(1./pzcy.T))\
					-mu_z[:,None]-(pen_c*pen_z)[:,None] )*px[None,:]
		return raw_grad - np.mean(raw_grad,axis=0)
	return grad_obj


def ib_alm(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	# revision record
	# align the steps in the draft
	# 1. pzcx@px-pz -----> pz-pzcx@px
	# 2. step size is now a diminishing sequence
	# -----------------------------------------------------
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	sel_idx = np.random.permutation(nx)
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	pzcx = np.random.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx[:nz,:] = pycx
	pzcy = pzcx@pxcy
	# ready to start
	itcnt = 0
	# gradient descent control
	_step_size = _global_step_size_init
	ss_pzcx = copy.copy(_step_size)
	ss_pz = copy.copy(_step_size)
	# defined in global variables
	isvalid = True
	_parm_c = _parm_c_init
	#_parm_mu_z = 0.5*(np.random.rand(nz))+0.5 # it seems like the initial values matters pretty much
	_parm_mu_z = np.zeros((nz,))# this performs poorly...
	# recording the gradient norm
	while itcnt < max_iter:
		itcnt+=1
		# NOTE: the two steps need not be in this fixed order
		#       there are randomized version ADMM
		pen_z = pz- pzcx@px
		grad_pzcx = (np.log(pzcx)+1-beta-beta*np.transpose(pycx.T@np.log(pzcy.T))
					-_parm_mu_z[:,np.newaxis]-_parm_c*pen_z[:,np.newaxis])*px[None,:]
		# FIXING the direction
		mean_grad_pzcx = grad_pzcx - np.mean(grad_pzcx,axis=0) # uniform mean, could be expectation from pzcx?
		# calculate update norm
		#norm_pzcx = np.linalg.norm(mean_grad_pzcx,2,axis=0)
		ss_pzcx = validStepSize(pzcx,-1 * mean_grad_pzcx, ss_pzcx,_bk_beta)
		if ss_pzcx == 0:
			isvalid = False
			break
		new_pzcx = pzcx - ss_pzcx * mean_grad_pzcx
		pen_z = pz- new_pzcx@px
		grad_pz = (beta-1)*(np.log(pz)+1)+_parm_mu_z+_parm_c*pen_z
		# gradient mean
		mean_grad_pz   = grad_pz   - np.mean(grad_pz)
		# update probabilities
		ss_pz   = validStepSize(pz,-1 * mean_grad_pz, ss_pz,_bk_beta)
		if ss_pz == 0:
			isvalid = False
			break
		new_pz   = pz   - ss_pz * mean_grad_pz
		# FIXME
		# for fair comparison, use total variation distance as termination criterion
		pen_z = new_pz - new_pzcx@px
		dtv_pen = 0.5* np.sum(np.fabs(pen_z))
		
		# markov chain condition
		# probability update
		pzcy = new_pzcx @ pxcy
		pzcx = new_pzcx
		pz = new_pz
		# mu update
		_parm_mu_z = _parm_mu_z + _parm_c * pen_z
		# ALM c update
		if dtv_pen < conv_thres:
			break
		
	# monitoring the MIXZ, MIYZ
	mixz = calc_mi(pzcx,px)
	miyz = calc_mi(pzcy,py)
	pen_check = 0.5*np.sum(np.fabs(pz-pzcx@px))
	if pen_check>=conv_thres:
		isvalid = False
	if not isvalid:
		if pen_check<conv_thres:
			print('WARNING: INVALID but the penalty converged')
		mixz = 0
		miyz = 0
	return {'prob_zcx':pzcx,'prob_z':pz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':isvalid}
'''
def ib_alm_v2(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	# The difference to V1:
	# the step size starts with a fixed constant
	# however, this will make the step size a varying sequence
	# In comparison, the stepsize of V1 is a converging sequence
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	sel_idx = np.random.permutation(nx)
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	pzcx = np.random.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx[:nz,:] = pycx

	pzcy = pzcx@pxcy
	# ready to start
	itcnt = 0
	# gradient descent control
	_step_size = _global_step_size_init
	# defined in global variables
	isvalid = True
	_parm_c = _parm_c_init
	_parm_mu_z = np.zeros((nz,))
	# recording the gradient norm
	while itcnt < max_iter:
		itcnt+=1
		# NOTE: the two steps need not be in this fixed order
		#       there are randomized version ADMM
		# NOTE: B^T is equivalent to post multiplying px
		grad_pzcx = (np.log(pzcx)+1-beta-beta*np.transpose(pycx.T@np.log(pzcy.T))
					+_parm_mu_z[:,np.newaxis]+_parm_c*(pz-pzcx@px)[:,np.newaxis])*px[None,:]
		mean_grad_pzcx = grad_pzcx - np.mean(grad_pzcx,axis=0)
		#norm_pzcx = np.linalg.norm(mean_grad_pzcx,2,axis=0)
		ss_pzcx = validStepSize(pzcx,-1 * mean_grad_pzcx, _step_size,_bk_beta)
		# update probabilities
		new_pzcx = pzcx - ss_pzcx * mean_grad_pzcx

		grad_pz = (beta-1)*(np.log(pz)+1)+_parm_mu_z+_parm_c*(pz-new_pzcx@px)
		mean_grad_pz   = grad_pz   - np.mean(grad_pz)
		#norm_pz   = np.linalg.norm(mean_grad_pz,2)
		# FIXME,
		# how to exploit strongly convex?
		ss_pz   = validStepSize(pz,-1 * mean_grad_pz, _step_size,_bk_beta)
		new_pz   = pz   - ss_pz * mean_grad_pz

		if ss_pzcx ==0 or ss_pz == 0:
			isvalid = False
			break
		# FIXME
		# for fair comparison, use total variation distance as termination criterion
		#dtv_pzcx = 0.5*np.sum(np.fabs(ss_pzcx*mean_grad_pzcx))
		#dtv_pz   = 0.5*np.sum(np.fabs(ss_pz*mean_grad_pz))
		#if dtv_pzcx < conv_thres and dtv_pz < conv_thres:
		#	break
		# or Instead, use gradient norm as termination criterion
		#if np.sum(norm_pzcx)< _parm_grad_norm_thres and norm_pz < _parm_grad_norm_thres:
		#	break
		
		# markov chain condition
		pzcy = new_pzcx @ pxcy
		# penalty update
		pen_z = new_pz-new_pzcx@px
		if 0.5* np.sum(np.fabs(pen_z))<conv_thres:
			# 1-norm for penalty function reached its goal
			break
		# mu update
		_parm_mu_z = _parm_mu_z + _parm_c * pen_z
		# probability update
		pzcx = new_pzcx
		pz = new_pz
		
	# monitoring the MIXZ, MIYZ
	mixz = calc_mi(pzcx,px)
	miyz = calc_mi(pzcy,py)
	pen_check = 0.5*np.sum(np.fabs(pz-pzcx@px))
	if pen_check>=conv_thres:
		isvalid = False
	if not isvalid:
		mixz = 0
		miyz = 0
	return {'prob_zcx':pzcx,'prob_z':pz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':isvalid}
'''



def ib_alm_breg(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	# this version, compared to previous ones is that it uses bregman divergence
	# which in our case, is the KL divergence on the pz updates.
	# This introduces a hyperparameter on the Bregman divergence
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	sel_idx = np.random.permutation(nx)
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	pz_delay = copy.copy(pz)
	pzcx = np.random.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx[:nz,:] = pycx
	pzcy = pzcx@pxcy
	# ready to start
	itcnt = 0
	# gradient descent control
	_step_size = _global_step_size_init
	ss_pzcx = copy.copy(_step_size)
	#ss_pz = copy.copy(_step_size)
	# defined in global variables
	isvalid = True
	_parm_c = _parm_c_init
	_parm_mu_z = np.zeros((nz,))
	# bregman divergence parameter
	dbd_omega = _parm_omega_init # FIXME
	# stepsize control
	pz_func_obj = getPzFuncObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pz_grad_obj = getPzGradObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pzcx_func_obj = getPzcxFuncObj(beta,px,py,pxcy,pycx,_parm_c)

	# recording the gradient norm
	while itcnt < max_iter:
		itcnt+=1
		# DEBUGGING
		#if d_debug:
		#	print('F_val:{:.6f},G_val:{:.6f}'.format(
		#		pz_func_obj(pz,pzcx,_parm_mu_z,pz_delay),pzcx_func_obj(pz,pzcx,_parm_mu_z)))
		# NOTE: the two steps need not be in this fixed order
		#       there are randomized version ADMM
		pen_z = pz- pzcx@px
		grad_pzcx = (np.log(pzcx)+1-beta-beta*np.transpose(pycx.T@np.log(pzcy.T))
					-_parm_mu_z[:,np.newaxis]-_parm_c*pen_z[:,np.newaxis])*px[None,:]
		# FIXING the direction
		mean_grad_pzcx = grad_pzcx - np.mean(grad_pzcx,axis=0) # uniform mean, could be expectation from pzcx?
		# calculate update norm
		# FIXME
		ss_pzcx = validStepSize(pzcx,-1 * mean_grad_pzcx, ss_pzcx,_bk_beta)
		#ss_pzcx = validStepSize(pzcx,-1 * mean_grad_pzcx, _step_size,_bk_beta)
		if ss_pzcx == 0:
			isvalid = False
			break
		new_pzcx = pzcx - ss_pzcx * mean_grad_pzcx
		pen_z = pz- new_pzcx@px
		grad_pz = (beta-1)*(np.log(pz)+1)+_parm_mu_z+_parm_c*pen_z + dbd_omega*(np.log(pz)-np.log(pz_delay))

		# gradient mean
		mean_grad_pz   = grad_pz   - np.mean(grad_pz)
		# change from sample mean to expectation,
		# also, employ steepest descent
		#mean_grad_pz = grad_pz - np.sum(pz*grad_pz)
		#pz_dir = -(1/(beta-1+dbd_omega))*(pz * mean_grad_pz)
		# use backtracking method
		tmp_dict = {
			'pzcx': new_pzcx,
			'mu_z': _parm_mu_z,
			'pz_last': pz_delay,
		}
		#ss_pz = backtrackStepSize(pz,pz_dir,1.0,_bk_beta,pz_func_obj,pz_grad_obj,**tmp_dict)
		ss_pz = backtrackStepSize(pz,-mean_grad_pz,1.0,_bk_beta,pz_func_obj,pz_grad_obj,**tmp_dict)
		# update probabilities
		#ss_pz   = validStepSize(pz,-1 * mean_grad_pz, ss_pz,_bk_beta)
		if ss_pz == 0:
			isvalid = False
			break
		new_pz   = pz   - ss_pz * mean_grad_pz
		# FIXME
		# for fair comparison, use total variation distance as termination criterion
		pen_z = new_pz - new_pzcx@px
		dtv_pen = 0.5* np.sum(np.fabs(pen_z))
		
		# markov chain condition
		# probability update
		pzcy = new_pzcx @ pxcy
		pzcx = new_pzcx
		pz_delay = copy.copy(pz)
		pz = new_pz
		# mu update
		_parm_mu_z = _parm_mu_z + _parm_c * pen_z
		# ALM c update
		if dtv_pen < conv_thres:
			break
	# monitoring the MIXZ, MIYZ
	mixz = calc_mi(pzcx,px)
	miyz = calc_mi(pzcy,py)
	pen_check = 0.5*np.sum(np.fabs(pz-pzcx@px))
	if pen_check>=conv_thres:
		isvalid = False
	if not isvalid:
		if pen_check<conv_thres:
			print('WARNING: INVALID but the penalty converged')
		mixz = 0
		miyz = 0
	return {'prob_zcx':pzcx,'prob_z':pz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':isvalid}


# FIXME
'''
def ib_alm_exp(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]

	if beta == 1.0:
		# TODO:
		# the update does not follow exponential gradient, but becomes slave of pzcx
		pzcx = np.ones((nz,nx))/qlevel
		pz = pzcx *px[:,None]
		pzcy = pzcx@pxcy
		return {'prob_pzcx':pzcx,'prob_pzcy':pzcy,'niter':1,'IXZ':0,'IYZ':0,'valid':1}
	# on IB, the initialization matters
	# initialize forward decoder
	pzcx = np.random.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcy = pzcx@pxcy
	#pz = np.random.rand(nz)
	pz = np.ones((nz,))
	pz = pz/np.sum(pz)
	pz_delay = copy.copy(pz)
	# counter
	itcnt = 0

	# Augmented Lagrange Method parameters
	alm_mu_z = np.zeros((nz,))
	alm_cz = _parm_c_init
	# valid check
	isvalid = True
	while itcnt < max_iter:
		itcnt+=1		
		pen_z  = pz-pzcx@px
		exp_expo = -beta*(np.log(1./pzcy)@pycx.T)+(alm_mu_z+alm_cz*(pz-pzcx@px))[:,None]
		new_pzcx = np.exp(exp_expo-np.mean(exp_expo,axis=0)[None,:])
		new_pzcx = new_pzcx*(1/ np.sum(new_pzcx,axis=0) )[None,:]
		if np.any(new_pzcx!=new_pzcx):
			isvalid=False
			break
			#print(pzcx)
			#print(pzcx@pxcy)
			#sys.exit()
		pen_z = pz-new_pzcx@px
		pz_expo = (1/(beta-1+_parm_omega_init))*np.log(pz_delay) \
				 -(_parm_omega_init/(beta-1+_parm_omega_init))*(alm_mu_z+alm_cz*(pen_z))
		new_pz = np.exp(pz_expo-np.mean(pz_expo))
		if np.any(new_pz!=new_pz):
			isvalid = False
			break
			#print(pzcx)
			#print(pen_z)
			#print(alm_mu_z)
			#sys.exit()
		new_pz = new_pz/np.sum(new_pz)
		pen_z = new_pz - new_pzcx@px
		# update augmented lagrange penalty factors
		alm_mu_z = alm_mu_z + alm_cz * pen_z
		dtv_z = 0.5*np.sum(np.fabs(pen_z))
		#print(dtv_z)
		if dtv_z < conv_thres:
			break
		pzcx = new_pzcx
		pzcy = pzcx @ pxcy
		pz_delay = copy.copy(pz)
		pz = new_pz
		# mutual information calculation
	mixz = calc_mi(pzcx,px)
	miyz = calc_mi(pzcy,py)
	if not isvalid:
		mixz = 0
		miyz = 0
	return {'prob_pzcx':pzcx,'prob_pzcy':pzcy,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':isvalid}
'''
'''
def dist_gen(x_alpha, ycx_alpha,**kwargs):
	px = np.random.dirichlet(x_alpha)
	pycx = np.zeros((ycx_alpha.shape))
	for it in range(ycx_alpha.shape[1]):
		pycx[:,it] = np.random.dirichlet(ycx_alpha[:,it])
	return np.transpose(pycx@np.diag(px))
'''

def run_sim(method,navg):
	_self_rescue_max_iter = 100*navg
	if method  == 'orig':
		ib_func = ib_orig
	elif method == 'gd':
		ib_func = ib_gd
	#elif method == 'alm_dbg':
	#	ib_func = ib_alm
	elif method == 'alm':
		ib_func = ib_alm
	#elif method == 'alm_nc':
	#	ib_func = ib_alm_v2
	elif method == 'alm_breg':
		ib_func = ib_alm_breg
	#elif method == 'alm_exp':
	#	ib_func = ib_alm_exp
	else:
		sys.exit('Undefined method:{}'.format(method))
	result = {'IXZ':[],'IYZ':[],'ITER':[],'VALID':[]}
	all_valid = np.zeros((len(beta_range),2))
	print('Running method:{}'.format(method))
	for bidx, tmp_beta in enumerate(beta_range):
		tmp_mixz = []
		tmp_miyz = []
		tmp_iter = []
		for it in range(navg):
			ib_res = ib_func(gbl_pxy,parm_qlevel,parm_conv_thres,tmp_beta,parm_max_iter)
			tmp_mixz.append(ib_res['IXZ'])
			tmp_miyz.append(ib_res['IYZ'])
			tmp_iter.append(ib_res['niter'])
			all_valid[bidx,int(ib_res['valid'])]+=1
			
		print('beta:{:>5.4f}, success rate: {:>5.4f}'.format(tmp_beta,np.sum(all_valid[bidx,1])/navg))
		result['IXZ'].append(np.array(tmp_mixz))
		result['IYZ'].append(np.array(tmp_miyz))
		result['ITER'].append(np.array(tmp_iter))
		result['VALID'].append(all_valid)
	return result

if d_debug:
	#ib_alm_res = ib_orig(gbl_pxy,parm_qlevel,parm_conv_thres,dbg_beta,dbg_max_iter)
	ib_alm_res = ib_alm_breg(gbl_pxy,parm_qlevel,parm_conv_thres,dbg_beta,dbg_max_iter)
	print(ib_alm_res)
	sys.exit('Debugging mode ends')

# main script

for item in result_dict['methods']:
	tmp_res = run_sim(item,num_avg)
	tmp_name = 'res_'+item
	if not result_dict.get(tmp_name,False):
		result_dict[tmp_name] = tmp_res
	else:
		print('Warning: overwriting {}'.format(tmp_name))


# saving the result
with open('ibgd_exp_{}.pkl'.format(exp_name),'wb') as fid:
	pickle.dump(result_dict,fid)
