import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import sys
import myutils as ut
import myentropy as ent
import graddescent as gd
import copy
import time

# MACRO

# TODO: make the algorithm a class
#       so it is possible to run any algorithm with a single construction

def selectAlg(method,**kwargs):
	if method =='orig':
		return ib_orig
	elif method == 'gd':
		return ib_gd
	elif method == 'alm':
		return ib_alm_breg
	elif method == 'sec':
		return ib_alm_sec
	elif method == 'dev':
		return ib_alm_dev
	elif method == 'bayat':
		return admmib_bayat
	elif method == 'mv':
		return ib_mv
	elif method == 'drs':
		return ib_drs
	elif method == 'acc_drs':
		return ib_drs_acc
	elif method == 'drs_mark':
		return ib_drs_mark
	return None

def ib_orig(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pycx = np.transpose(np.diag(1./px)@pxy)
	pxcy = pxy@np.diag(1./py)
	# on IB, the initialization matters
	# use random start (*This is the one used for v2)
	sel_idx = rs.permutation(nx)
	pzcx = rs.rand(nz,nx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	
	pzcx /= np.sum(pzcx,axis=0)[None,:]
	pz = np.sum(pzcx *px[None,:],axis=1)
	pycz = pycx@ np.transpose(1/pz[:,None]*pzcx*px[None,:])
	
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
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pycz,pz)
	return {'prob_zcx':pzcx,'prob_ycz':pycz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':True}


def ib_alm_dev(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	_bk_beta = kwargs['backtracking_beta']
	_ls_init = kwargs['line_search_init']
	# learning rate scheduler
	ls_schedule = ut.getLsSchedule(_ls_init)
	ls_idx = 0
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	# DEBUG
	# FIXME: tuning the bregman regularization for pzcx, make it a parameter once complete debugging
	# Initial result, this will not improve rate of convergence...
	
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	sel_idx = rs.permutation(nx)
	pzcx = rs.rand(nz,nx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	
	pzcx /= np.sum(pzcx,axis=0)[None,:]
	pz = np.sum(pzcx*px[None,:],axis=1)
	pz_delay = copy.copy(pz)
	pzcy = pzcx@pxcy
	# ready to start
	itcnt = 0
	# gradient descent control
	# defined in global variables
	_parm_c = kwargs['penalty_coeff']
	_parm_mu_z = np.zeros((nz,))
	# bregman divergence parameter
	dbd_omega = kwargs['breg_omega']
	# stepsize control
	pz_func_obj = ent.getPzFuncObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pz_grad_obj = ent.getPzGradObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pzcx_func_obj = ent.getPzcxFuncObj(beta,px,py,pxcy,pycx,_parm_c)
	pzcx_grad_obj = ent.getPzcxGradObj(beta,px,pxcy,pycx,_parm_c)
	while itcnt < max_iter:
		itcnt+=1
		'''
		if itcnt == ls_schedule[ls_idx][0]:
			_ls_init = ls_schedule[ls_idx][1]
			ls_idx += 1
			if ls_idx == len(ls_schedule):
				ls_idx -=1 # stay at the end....
		'''
		pen_z = pz- pzcx@px		
		(mean_grad_pzcx,_) = pzcx_grad_obj(pzcx,pz,_parm_mu_z)
		mean_grad_pzcx/=beta
		# calculate update norm
		# unit step size for every update
		ss_pzcx = gd.validStepSize(pzcx,-mean_grad_pzcx, _ls_init,_bk_beta)
		if ss_pzcx == 0:
			break
		arm_ss_pzcx = gd.armijoStepSize(pzcx,-mean_grad_pzcx,ss_pzcx,_bk_beta,1e-4,pzcx_func_obj,pzcx_grad_obj,
									**{'pz':pz,'mu_z':_parm_mu_z},)
		if arm_ss_pzcx==0:
			arm_ss_pzcx = ss_pzcx
		new_pzcx = pzcx - arm_ss_pzcx * mean_grad_pzcx
		
		# TODO
		# implement the over-relaxed ADMM to update convex primal variable
		(mean_grad_pz,_) = pz_grad_obj(pz,new_pzcx,_parm_mu_z,pz_delay)
		mean_grad_pz/=beta
		ss_pz = gd.validStepSize(pz,-mean_grad_pz,_ls_init,_bk_beta)
		# update probabilities
		if ss_pz == 0:
			break
		arm_ss_pz = gd.armijoStepSize(pz,-mean_grad_pz,ss_pz,_bk_beta,1e-4,pz_func_obj,pz_grad_obj,
									**{'pzcx':new_pzcx,'mu_z':_parm_mu_z,'pz_last':pz_delay})
		
		if arm_ss_pz == 0:
			arm_ss_pz = ss_pz
		new_pz   = pz   - arm_ss_pz * mean_grad_pz
		
		## End Developing section
		# for fair comparison, use total variation distance as termination criterion
		pen_z = new_pz - np.sum(new_pzcx*px[None,:],axis=1)
		dtv_pen = 0.5* np.sum(np.fabs(pen_z))
		# markov chain condition
		# probability update
		pzcy = new_pzcx @ pxcy
		pzcx = new_pzcx
		pz_delay = copy.copy(pz)
		pz = new_pz
		# mu update
		_parm_mu_z = _parm_mu_z + _parm_c * pen_z
		if dtv_pen < conv_thres:
			break
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
	pen_check = 0.5*np.sum(np.fabs(pz-np.sum(pzcx*px[None,:],axis=1) ))
	isvalid = (pen_check<=conv_thres)
	
	return {'prob_zcx':pzcx,'prob_z':pz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':isvalid}


def ib_gd(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	# system
	_bk_beta = kwargs['backtracking_beta']
	_ls_init = kwargs['line_search_init']
	ls_schedule = ut.getLsSchedule(_ls_init)
	ls_idx = 0
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	sel_idx = rs.permutation(nx)
	pzcx = rs.rand(nz,nx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	
	pzcx /=np.sum(pzcx,axis=0)[None,:]
	pz = np.sum(pzcx * px[None,:],axis=1)
	pzcy = pzcx@pxcy

	pycz = ((1/pz) @pzcy @ py).T
	# naive method
	grad_obj = ent.getLibGDGradObj(beta,px,py,pxcy,pycx)
	func_obj = ent.getLibGDFuncObj(beta,px,py,pxcy,pycx)
	# ready to start
	itcnt = 0
	# gradient descent control
	while (itcnt < max_iter):
		itcnt+=1
		'''
		if itcnt == ls_schedule[ls_idx][0]:
			_ls_init = ls_schedule[ls_idx][1]
			ls_idx += 1
			if ls_idx == len(ls_schedule):
				ls_idx -=1 # stay at the end....
		'''
		# compute ib kernel
		(mean_grad,_) = grad_obj(pzcx,pz,pycz)
		mean_grad/=beta
		# NOTE: 
		#     if all pzcx are non zero, then the ineq. eq. conditions gives
		#     lambda_i = -mean(grad)
		if np.any(mean_grad != mean_grad):
			break
		tmp_step = gd.validStepSize(pzcx,-mean_grad,_ls_init,_bk_beta)
		if tmp_step == 0:
			break
		arm_step = gd.armijoStepSize(pzcx,-mean_grad,tmp_step,_bk_beta,1e-4,func_obj,grad_obj,
									**{'pz':pz,'pycz':pycz})
		if arm_step == 0:
			arm_step = tmp_step
		new_pzcx = pzcx - arm_step * mean_grad
		
		dtv = 0.5 * np.sum(np.fabs(pz-np.sum(new_pzcx*px[None,:],axis=1)))
		if dtv< conv_thres:
			# termination condition reached
			# making the comparison at the same criterion
			break
		else:
			# updating step size
			pzcx = new_pzcx
			pz = np.sum(pzcx*px[None,:],axis=1)
			pzcy = pzcx@pxcy
			pycz = np.transpose((1./pz)[:,None]*pzcy*py[None,:])
			pycz = pycz * (1./np.sum(pycz,axis=0))[None,:]  # must be done for valid probability constraint
			

	# monitoring the MIXZ, MIYZ
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pycz,pz)
	pen_check = 0.5*np.sum(np.fabs(pz-pzcx@px))
	flag_valid = (pen_check<=conv_thres)
	
	return {'prob_zcx':pzcx,'prob_ycz':pycz,'niter':itcnt,'valid':flag_valid,'IXZ':mixz,'IYZ':miyz}
# ----------------------------------------------------
# DEVELOPING
# ----------------------------------------------------
def ib_alm_sec(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	_bk_beta = kwargs['backtracking_beta']
	#_bk_alpha = kwargs['backtracking_alpha']
	ls_idx = 0
	_ls_init = kwargs['line_search_init']
	ls_schedule = ut.getLsSchedule(_ls_init)
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
	pzcx = np.random.rand(nz,nx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	
	pzcx /= np.sum(pzcx,axis=0)[None,:]
	pz = np.sum(pzcx*px[None,:],axis=1)
	pz_delay = copy.copy(pz)
	pzcy = pzcx@pxcy
	# ready to start
	itcnt = 0
	isvalid = False
	# gradient descent control
	# defined in global variables
	_parm_c = kwargs['penalty_coeff']
	_parm_mu_z = np.zeros((nz,))
	# bregman divergence parameter
	dbd_omega = kwargs['breg_omega']
	# stepsize control
	pz_func_obj = ent.getPzFuncObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pz_grad_obj = ent.getPzGradObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pz_newton_obj = ent.getPzNewtonObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pzcx_func_obj = ent.getPzcxFuncObj(beta,px,py,pxcy,pycx,_parm_c)
	pzcx_grad_obj = ent.getPzcxGradObj(beta,px,pxcy,pycx,_parm_c)
	pzcx_newton_obj = ent.getPzcxNewtonObj(beta,px,pxcy,pycx,_parm_c)
	while itcnt < max_iter:
		itcnt += 1
		'''
		# SECOND ORDER METHOD FOR PZCX
		# pzcx direction update
		(pdir_pzcx,lambda_pzcx) = pzcx_newton_obj(pzcx,pz,_parm_mu_z)
		ss_pzcx = gd.validStepSize(pzcx,-pdir_pzcx,1.0,_bk_beta)
		if ss_pzcx == 0:
			break
		arm_ss_pzcx = gd.wolfeStepSize(pzcx,-pdir_pzcx,ss_pzcx,_bk_beta,1e-4,
			pzcx_func_obj,pzcx_grad_obj,lambda_pzcx,**{'pz':pz,'mu_z':_parm_mu_z})
		if arm_ss_pzcx == 0:
			# FIXME, find a better stepsize: at least reducing the objective function
			arm_ss_pzcx = ss_pzcx
		# update pzcx
		new_pzcx = pzcx -pdir_pzcx*arm_ss_pzcx
		'''
		# Use first order method for faster computation
		(pdir_pzcx, lambda_pzcx) = pzcx_grad_obj(pzcx,pz,_parm_mu_z) # first order lambda
		ss_pzcx = gd.validStepSize(pzcx,-pdir_pzcx,_ls_init,_bk_beta) # the starting constant matters!!
		if ss_pzcx == 0:
			break
		arm_ss_pzcx = gd.armijoStepSize(pzcx,-pdir_pzcx,ss_pzcx,_bk_beta,1e-4,pzcx_func_obj,pzcx_grad_obj,
									**{'pz':pz,'mu_z':_parm_mu_z})
		if arm_ss_pzcx == 0:
			arm_ss_pzcx = ss_pzcx
		# update pzcx
		new_pzcx = pzcx - pdir_pzcx*arm_ss_pzcx
		# pz direction update
		# SECOND ORDER METHOD FOR PZ
		(pdir_pz,lambda_pz) = pz_newton_obj(pz,new_pzcx,_parm_mu_z,pz_delay)
		ss_pz = gd.validStepSize(pz,-pdir_pz,1.0,_bk_beta)
		if ss_pz == 0:
			break
		arm_ss_pz = gd.wolfeStepSize(pz,-pdir_pz,ss_pz,_bk_beta,1e-4,
			pz_func_obj,pz_grad_obj,lambda_pz,**{'pzcx':new_pzcx,'mu_z':_parm_mu_z,'pz_last':pz_delay})
		if arm_ss_pz == 0:
			# FIXME. find a better stepsize: at least reducing the objective function
			arm_ss_pz = ss_pz
		# update pz
		new_pz = pz - pdir_pz * arm_ss_pz
		# update lagrange multiplier approxmators
		new_penz = new_pz - np.sum(new_pzcx*px[None,:],axis=1)
		_parm_mu_z += _parm_c*new_penz
		# update probabilities
		pzcx = new_pzcx
		pz_delay = copy.copy(pz)
		pz = new_pz
		pzcy = new_pzcx @ pxcy
		# termination condition
		dtv = 0.5 * np.sum(np.fabs(new_penz))
		if dtv< conv_thres:
			isvalid = True
			break
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
	pen_check = 0.5*np.sum(np.fabs(pz-pzcx@px))
	isvalid = (pen_check<=conv_thres)
	
	return {'prob_zcx':pzcx,'prob_z':pz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':isvalid}

def admmib_bayat(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	# initialize three sets
	_parm_c = kwargs['penalty_coeff']
	_bk_beta = kwargs['backtracking_beta']
	_ls_init = kwargs['line_search_init']
	ls_schedule = ut.getLsSchedule(_ls_init)
	ls_idx = 0
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	sel_idx = rs.permutation(nx)
	pzcx = rs.rand(nz,nx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	
	pzcx /= np.sum(pzcx,axis=0)[None,:]
	pz = np.sum(pzcx * px[None,:],axis=1)
	pzcy = pzcx@pxcy

	_parm_mu_z  = np.zeros((nz,))
	_parm_mu_zy = np.zeros((nz,ny))
	bayat_pz_obj    = ent.getBayatFuncObjPz(beta,px,py,pxcy,pycx,_parm_c)
	bayat_pzcx_obj  = ent.getBayatFuncObjPzcx(beta,px,py,pxcy,pycx,_parm_c)
	bayat_pzcy_obj  = ent.getBayatFuncObjPzcy(beta,px,py,pxcy,pycx,_parm_c)
	bayat_pz_grad   = ent.getBayatGradObjPz(beta,px,py,pxcy,pycx,_parm_c)
	bayat_pzcx_grad = ent.getBayatGradObjPzcx(beta,px,py,pxcy,pycx,_parm_c)
	bayat_pzcy_grad = ent.getBayatGradObjPzcy(beta,px,py,pxcy,pycx,_parm_c)
	#lad = 1/beta
	isvalid = False
	itcnt = 0 
	while itcnt < max_iter:
		# first attempt, three block method, three penalties
		# forcing 
		itcnt += 1
		if itcnt == ls_schedule[ls_idx][0]:
			_ls_init = ls_schedule[ls_idx][1]
			ls_idx += 1
			if ls_idx == len(ls_schedule):
				ls_idx -=1 # stay at the end....
		# the order of step is another thing to tune...
		# step1: pzcx
		(mean_grad_pzcx,_) = bayat_pzcx_grad(pzcx,pzcy,pz,_parm_mu_z,_parm_mu_zy)
		mean_grad_pzcx /= beta
		ss_pzcx = gd.validStepSize(pzcx,-mean_grad_pzcx,_ls_init,_bk_beta)
		if ss_pzcx == 0:
			isvalid = False
			break
		arm_ss_pzcx = gd.armijoStepSize(pzcx,-mean_grad_pzcx,ss_pzcx,_bk_beta,1e-4,bayat_pzcx_obj,bayat_pzcx_grad,
									**{'pzcy':pzcy,'pz':pz,'mu_z':_parm_mu_z,'mu_zy':_parm_mu_zy},)
		if arm_ss_pzcx == 0:
			arm_ss_pzcx = ss_pzcx
		new_pzcx = pzcx - arm_ss_pzcx * mean_grad_pzcx
		# step2: pzcy		
		(mean_grad_pzcy,_) = bayat_pzcy_grad(pzcy,new_pzcx,_parm_mu_zy)
		mean_grad_pzcy /= beta
		ss_pzcy = gd.validStepSize(pzcy,-mean_grad_pzcy,_ls_init,_bk_beta)
		if ss_pzcy == 0:
			isvalid =False
			break
		arm_ss_pzcy = gd.armijoStepSize(pzcy,-mean_grad_pzcy,ss_pzcy,_bk_beta,1e-4,bayat_pzcy_obj,bayat_pzcy_grad,
											**{'pzcx':new_pzcx,'mu_zy':_parm_mu_zy})
		if arm_ss_pzcy == 0:
			arm_ss_pzcy = ss_pzcy
		new_pzcy = pzcy - arm_ss_pzcy * mean_grad_pzcy
		
		# step3: pz
		(mean_grad_pz,_) = bayat_pz_grad(pz,new_pzcx,_parm_mu_z)
		mean_grad_pz/= beta
		ss_pz = gd.validStepSize(pz,-mean_grad_pz,_ls_init,_bk_beta)
		if ss_pz  == 0:
			isvalid =False
			break
		arm_ss_pz = gd.armijoStepSize(pz,-mean_grad_pz,ss_pz,_bk_beta,1e-4,bayat_pz_obj,bayat_pz_grad,
										**{'pzcx':new_pzcx,'mu_z':_parm_mu_z})
		if arm_ss_pz == 0:
			arm_ss_pz = ss_pz
		new_pz = pz - arm_ss_pz * mean_grad_pz
		# update
		pz = copy.copy(new_pz)
		pzcx = copy.copy(new_pzcx)
		pzcy = copy.copy(new_pzcy)
		penalty_pz = pz - np.sum(pzcx*px[None,:],axis=1)
		_parm_mu_z +=  _parm_c*(penalty_pz)
		tmp_pzcy = pzcx@pxcy
		penalty_pzcy = pzcy - tmp_pzcy/(np.sum(tmp_pzcy,axis=0)[None,:])
		_parm_mu_zy += _parm_c*(penalty_pzcy)
		# termination condition?
		dtv_z = 0.5* (np.sum(np.fabs(penalty_pz)))
		dtv_zy = 0.5* (np.sum(np.fabs(penalty_pzcy),axis=0)) # a y dimensional vector
		if dtv_z < conv_thres and np.all(dtv_zy < conv_thres):
		#if dtv_z < conv_thres:
			isvalid = True
			break
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
	pen_check = 0.5*np.sum(np.fabs(pz-pzcx@px))
	pen_check_pzcy = 0.5*(np.sum(np.fabs(pzcy-(pzcx@pxcy)/(np.sum(pzcx@pxcy,axis=0)[None,:])),axis=0))
	isvalid = (pen_check<=conv_thres and np.all(pen_check_pzcy<=conv_thres))
	# FIXME: this can unify the convergence condition but will ignore the other penalty 
	#        which is not fair in terms of the design given this is also a penalty method.
	#isvalid = (pen_check<=conv_thres)  
	

	return {'prob_zcx':pzcx,'prob_z':pz,'prob_zcy':pzcy,
			'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':isvalid}

def ib_mv(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	_bk_beta = kwargs['backtracking_beta']
	_ls_init = kwargs['line_search_init']
	# learning rate scheduler
	ls_schedule = ut.getLsSchedule(_ls_init)
	ls_idx = 0
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	sel_idx = rs.permutation(nx)
	pzcx = rs.rand(nz,nx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	pzcx /= np.sum(pzcx,axis=0)[None,:]
	pz = np.sum(pzcx*px[None,:],axis=1)
	pzcy = pzcx@pxcy
	# defined in global variables
	pen_c = kwargs['penalty_coeff']
	# dual variables
	dual_z = np.zeros((nz,))
	dual_zy = np.zeros((nz,ny))
	# using the objects
	grad_pz_obj = ent.getMvGradObjPz(beta,px,pen_c)
	grad_pzcy_obj = ent.getMvGradObjPzcy(py,pxcy,pen_c)
	grad_pzcx_obj = ent.getMvGradObjPzcx(beta,px,pxcy,pen_c)
	func_pz_obj = ent.getMvFuncObjPz(beta,px,pen_c)
	func_pzcy_obj = ent.getMvFuncObjPzcy(py,pxcy,pen_c)
	func_pzcx_obj = ent.getMvFuncObjPzcx(beta,px,pxcy,pen_c)
	# ready to start
	flag_valid = False
	itcnt = 0
	while itcnt< max_iter:
		itcnt+=1
		'''
		if itcnt == ls_schedule[ls_idx][0]:
			_ls_init = ls_schedule[ls_idx][1]
			ls_idx += 1
			if ls_idx == len(ls_schedule):
				ls_idx -=1 # stay at the end....
		'''
		# update the str cvx part first
		(grad_x,_) = grad_pzcx_obj(pzcx,pz,pzcy,dual_z,dual_zy)
		ss_x= gd.validStepSize(pzcx,-grad_x,_ls_init,_bk_beta)
		# FIXME: pick the one with better rate of convergence (empirical)
		#ss_x = gd.armijoStepSize(pzcx,-grad_x,_ls_init,_bk_beta,1e-4,func_pzcx_obj,grad_pzcx_obj,**{'pz':pz,'pzcy':pzcy,'mu_z':dual_z,'mu_zy':dual_zy})
		if ss_x == 0:
			break
		new_pzcx = pzcx - grad_x*ss_x
		# update pz and pzcy together, with min stepsize
		(grad_z,_) = grad_pz_obj(pz,new_pzcx,dual_z)
		ss_z = gd.validStepSize(pz,-grad_z,_ls_init,_bk_beta)
		# FIXME: pick the one with better rate of convergence (empirical)
		#ss_z = gd.armijoStepSize(pz,-grad_z,_ls_init,_bk_beta,1e-4,func_pz_obj,grad_pz_obj,**{'pzcx':new_pzcx,'mu_z':dual_z})
		if ss_z ==0:
			break
		# start with this valid stepsize to save time
		(grad_y,_) = grad_pzcy_obj(pzcy,new_pzcx,dual_zy)
		#ss_zy = gd.armijoStepSize(pzcy,-grad_y,ss_z,_bk_beta,1e-4,func_pzcy_obj,grad_pzcy_obj,**{'pzcx':new_pzcx,'mu_zy':dual_zy})
		# FIXME: pick the one with better rate of convergence (empirical)
		ss_zy = gd.validStepSize(pzcy,-grad_y,ss_z,_bk_beta)
		if ss_zy == 0:
			break
		# update augmented vars
		new_pz = pz - grad_z * ss_zy
		new_pzcy = pzcy - grad_y * ss_zy
		# update dual var
		errz = np.sum(new_pzcx*px[None,:],axis=1) - new_pz
		errzy= new_pzcx @ pxcy - new_pzcy
		dual_z += pen_c*errz
		dual_zy+= pen_c*errzy
		# convergence check
		dtv_z = 0.5* np.sum(np.fabs(errz)) < conv_thres
		dtv_zy=0.5 * np.sum(np.fabs(errzy),axis=0) < conv_thres
		flag_valid = np.all(np.array(dtv_z)) and np.all(np.array(dtv_zy))
		if flag_valid:
			break
		else:
			pzcx = new_pzcx
			pz = new_pz
			pzcy = new_pzcy
	'''
	debug_errz = np.sum(new_pzcx*px[None,:],axis=1)-pz
	debug_errzy= new_pzcx@pxcy - pzcy
	print(0.5*np.sum(np.fabs(debug_errz)))
	print(0.5*np.sum(np.fabs(debug_errzy),axis=0))
	'''
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
	return {'prob_zcx':pzcx,'prob_z':pz,'prob_zcy':pzcy,
			'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':flag_valid}

def ib_drs(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	_record_flag= kwargs.get('record',False)
	_bk_beta = kwargs['backtracking_beta']
	_ls_init = kwargs['line_search_init']
	# learning rate scheduler
	ls_schedule = ut.getLsSchedule(_ls_init)
	ls_idx = 0
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	sel_idx = rs.permutation(nx)
	pzcx = rs.rand(nz,nx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	pzcx /= np.sum(pzcx,axis=0)[None,:]
	pz= np.sum(pzcx * px[None,:],axis=1)
	pzcy = pzcx@pxcy
	# defined in global variables
	pen_c = kwargs['penalty_coeff']
	drs_relax_coeff = kwargs['relax_coeff']
	# dual variables
	dual_z = np.zeros((nz,))
	dual_zy = np.zeros((nz,ny))

	dual_drs_z = np.zeros((nz,))
	dual_drs_zy = np.zeros((nz,ny))
	# using the objects
	grad_pz_obj = ent.getMvGradObjPz(beta,px,pen_c)
	grad_pzcy_obj = ent.getMvGradObjPzcy(py,pxcy,pen_c)
	grad_pzcx_obj = ent.getMvGradObjPzcx(beta,px,pxcy,pen_c)
	func_pz_obj = ent.getMvFuncObjPz(beta,px,pen_c)
	func_pzcy_obj = ent.getMvFuncObjPzcy(py,pxcy,pen_c)
	func_pzcx_obj = ent.getMvFuncObjPzcx(beta,px,pxcy,pen_c)
	# ready to start
	flag_valid = False
	itcnt = 0
	record_lval = np.zeros(1)
	if _record_flag:
		record_lval= np.zeros(max_iter)
	gamma = 1/beta
	while itcnt< max_iter:
		itcnt+=1
		# (gamma-1)H(Z)-gamma H(Z|X)+H(Z|Y)
		errz =  np.sum(pzcx*px[None,:],axis=1)-pz
		errzy = pzcx@pxcy - pzcy
		record_lval[itcnt % len(record_lval)] = (1-gamma)*np.sum(pz*np.log(pz)) +gamma *np.sum(pzcx*px[None,:]*np.log(pzcx))\
												-np.sum(pzcy*py[None,:]*np.log(pzcy))+np.sum(dual_z*errz)\
												+0.5*pen_c*(np.linalg.norm(errz)**2) + np.sum(dual_zy*errzy)\
												+0.5*pen_c*(np.linalg.norm(errzy)**2)
		# update the str cvx part first
		(grad_x,_) = grad_pzcx_obj(pzcx,pz,pzcy,dual_z,dual_zy)
		ss_x= gd.validStepSize(pzcx,-grad_x,_ls_init,_bk_beta)
		# FIXME: pick the one with better rate of convergence (empirical)
		#ss_x = gd.armijoStepSize(pzcx,-grad_x,_ls_init,_bk_beta,1e-4,func_pzcx_obj,grad_pzcx_obj,**{'pz':pz,'pzcy':pzcy,'mu_z':dual_z,'mu_zy':dual_zy})
		if ss_x == 0:
			break
		new_pzcx = pzcx - grad_x*ss_x

		# in drs, the intermediate error is computed 
		dual_drs_z = dual_z - pen_c*(1-drs_relax_coeff)*(np.sum(pzcx*px[None,:],axis=1) - pz)
		dual_drs_zy = dual_zy - pen_c*(1-drs_relax_coeff)*(pzcx@pxcy - pzcy)

		# update pz and pzcy together, with min stepsize
		(grad_z,_) = grad_pz_obj(pz,new_pzcx,dual_drs_z)
		ss_z = gd.validStepSize(pz,-grad_z,_ls_init,_bk_beta)
		# FIXME: pick the one with better rate of convergence (empirical)
		#ss_z = gd.armijoStepSize(pz,-grad_z,_ls_init,_bk_beta,1e-4,func_pz_obj,grad_pz_obj,**{'pzcx':new_pzcx,'mu_z':dual_z})
		if ss_z ==0:
			break
		# start with this valid stepsize to save time
		(grad_y,_) = grad_pzcy_obj(pzcy,new_pzcx,dual_drs_zy)
		#ss_zy = gd.armijoStepSize(pzcy,-grad_y,ss_z,_bk_beta,1e-4,func_pzcy_obj,grad_pzcy_obj,**{'pzcx':new_pzcx,'mu_zy':dual_zy})
		# FIXME: pick the one with better rate of convergence (empirical)
		ss_zy = gd.validStepSize(pzcy,-grad_y,ss_z,_bk_beta)
		if ss_zy == 0:
			break
		# update augmented vars
		new_pz = pz - grad_z * ss_zy
		new_pzcy = pzcy - grad_y * ss_zy
		# update dual var
		errz = np.sum(new_pzcx*px[None,:],axis=1) - new_pz
		errzy= new_pzcx @ pxcy - new_pzcy
		# update dual var
		dual_z = dual_drs_z + pen_c*errz
		dual_zy = dual_drs_zy + pen_c*errzy
		# convergence check
		dtv_z = 0.5* np.sum(np.fabs(errz)) < conv_thres
		dtv_zy=0.5 * np.sum(np.fabs(errzy),axis=0) < conv_thres
		flag_valid = np.all(np.array(dtv_z)) and np.all(np.array(dtv_zy))
		if flag_valid:
			break
		else:
			pzcx = new_pzcx
			pz = new_pz
			pzcy = new_pzcy
	'''
	debug_errz = np.sum(new_pzcx*px[None,:],axis=1)-pz
	debug_errzy= new_pzcx@pxcy - pzcy
	print(0.5*np.sum(np.fabs(debug_errz)))
	print(0.5*np.sum(np.fabs(debug_errzy),axis=0))
	'''
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
	outputdict = {'prob_zcx':pzcx,'prob_z':pz,'prob_zcy':pzcy,
			'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':flag_valid}
	if _record_flag:
		outputdict['val_record'] = record_lval[:itcnt]
	return outputdict

def ib_drs_acc(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	_bk_beta = kwargs['backtracking_beta']
	_ls_init = kwargs['line_search_init']
	# learning rate scheduler
	ls_schedule = ut.getLsSchedule(_ls_init)
	ls_idx = 0
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	
	sel_idx = rs.permutation(nx)
	pzcx = rs.rand(nz,nx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	pzcx /= np.sum(pzcx,axis=0)[None,:]
	pz = np.sum(pzcx * px[None,:],axis=1)
	pzcy = pzcx@pxcy
	# defined in global variables
	pen_c = kwargs['penalty_coeff']
	drs_relax_coeff = kwargs['relax_coeff']
	hv_damp = 5e-2
	# dual variables
	dual_z = np.zeros((nz,))
	dual_zy = np.zeros((nz,ny))
	dual_drs_z = np.zeros((nz,))
	dual_drs_zy = np.zeros((nz,ny))
	# delay registers
	# for acceleration
	delay_pzcx = copy.copy(pzcx)
	delay_pz = copy.copy(pz)
	delay_pzcy = copy.copy(pzcy)
	# using the objects
	grad_pz_obj = ent.getMvGradObjPz(beta,px,pen_c)
	grad_pzcy_obj = ent.getMvGradObjPzcy(py,pxcy,pen_c)
	grad_pzcx_obj = ent.getMvGradObjPzcx(beta,px,pxcy,pen_c)
	func_pz_obj = ent.getMvFuncObjPz(beta,px,pen_c)
	func_pzcy_obj = ent.getMvFuncObjPzcy(py,pxcy,pen_c)
	func_pzcx_obj = ent.getMvFuncObjPzcx(beta,px,pxcy,pen_c)
	# ready to start
	flag_valid = False
	itcnt = 0
	while itcnt< max_iter:
		itcnt+=1
		'''
		if itcnt == ls_schedule[ls_idx][0]:
			_ls_init = ls_schedule[ls_idx][1]
			ls_idx += 1
			if ls_idx == len(ls_schedule):
				ls_idx -=1 # stay at the end....
		'''
		# in drs, the intermediate error is computed first
		dual_drs_z = dual_z - pen_c*(1-drs_relax_coeff)*(np.sum(pzcx*px[None,:],axis=1) - pz)
		dual_drs_zy = dual_zy - pen_c*(1-drs_relax_coeff)*(pzcx@pxcy - pzcy)
		# update the str cvx part first
		(grad_x,_) = grad_pzcx_obj(pzcx,pz,pzcy,dual_drs_z,dual_drs_zy)
		#ss_x= gd.validStepSize(pzcx,-grad_x,_ls_init,_bk_beta)
		ss_x = gd.heavyBallStepSize(pzcx,delay_pzcx,-grad_x,_ls_init,_bk_beta,hv_damp)
		# FIXME: pick the one with better rate of convergence (empirical)
		if ss_x == 0:
			break
		new_pzcx = pzcx - grad_x*ss_x + hv_damp*(pzcx - delay_pzcx)
		# update dual var
		dual_z = dual_drs_z + pen_c*(np.sum(new_pzcx*px[None,:],axis=1)-pz)
		dual_zy = dual_drs_zy + pen_c*(new_pzcx@pxcy - pzcy)

		# update pz and pzcy together, with min stepsize
		(grad_z,_) = grad_pz_obj(pz,new_pzcx,dual_z)
		#ss_z = gd.validStepSize(pz,-grad_z,_ls_init,_bk_beta)
		ss_z = gd.heavyBallStepSize(pz,delay_pz,-grad_z,_ls_init,_bk_beta,hv_damp)
		# FIXME: pick the one with better rate of convergence (empirical)
		if ss_z ==0:
			break
		# start with this valid stepsize to save time
		(grad_y,_) = grad_pzcy_obj(pzcy,new_pzcx,dual_zy)
		# FIXME: pick the one with better rate of convergence (empirical)
		#ss_zy = gd.validStepSize(pzcy,-grad_y,ss_z,_bk_beta)
		ss_zy = gd.heavyBallStepSize(pzcy,delay_pzcy,-grad_y,ss_z,_bk_beta,hv_damp)
		if ss_zy == 0:
			break
		# update augmented vars
		new_pz = pz - grad_z*ss_zy + hv_damp*(pz - delay_pz)
		new_pzcy = pzcy - grad_y*ss_zy + hv_damp*(pzcy - delay_pzcy)
		# update dual var
		errz = np.sum(new_pzcx*px[None,:],axis=1) - new_pz
		errzy= new_pzcx @ pxcy - new_pzcy
		#dual_z += pen_c*errz
		#dual_zy+= pen_c*errzy
		# convergence check
		dtv_z = 0.5* np.sum(np.fabs(errz)) < conv_thres
		dtv_zy=0.5 * np.sum(np.fabs(errzy),axis=0) < conv_thres
		flag_valid = np.all(np.array(dtv_z)) and np.all(np.array(dtv_zy))
		if flag_valid:
			break
		else:
			# for heavy ball acceleration
			delay_pzcx = pzcx
			delay_pz = pz
			delay_pzcy = pzcy
			# standard gradient-descent pass
			pzcx = new_pzcx
			pz = new_pz
			pzcy = new_pzcy
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
	return {'prob_zcx':pzcx,'prob_z':pz,'prob_zcy':pzcy,
			'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':flag_valid}

def ib_drs_mark(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	_record_flag= kwargs.get('record',False)
	_bk_beta = kwargs['backtracking_beta']
	_ls_init = kwargs['line_search_init']
	# learning rate scheduler
	ls_schedule = ut.getLsSchedule(_ls_init)
	ls_idx = 0
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	py = py/np.sum(py)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy/py[None,:]
	# on IB, the initialization matters
	# use random start
	sel_idx = rs.permutation(nx)
	pzcx = rs.rand(nz,nx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	pzcx /= np.sum(pzcx,axis=0)[None,:]
	pz = np.sum(pzcx * px[None,:],axis=1)

	#pzcy = pzcx@pxcy
	# defined in global variables
	pen_c = kwargs['penalty_coeff']
	drs_relax_coeff = kwargs['relax_coeff']
	# dual variables
	dual_z = np.zeros((nz,))
	dual_drs_z = np.zeros((nz,))
	# using the objects
	grad_pz_obj = ent.getDrsmarkGradObjPz(beta,px,pen_c)
	grad_pzcx_obj = ent.getDrsmarkGradObjPzcx(beta,px,py,pxcy,pycx,pen_c)
	func_pz_obj = ent.getDrsmarkFuncObjPz(beta,px,pen_c)
	func_pzcx_obj = ent.getDrsmarkFuncObjPzcx(beta,px,py,pxcy,pycx,pen_c)
	# ready to start
	flag_valid = False
	itcnt = 0
	record_lval = np.zeros(1)
	if _record_flag:
		record_lval = np.zeros(max_iter)
	gamma = 1/beta
	while itcnt < max_iter:
		# l = (gamma-1)H(Z) -gammaH(Z|X) + H(Z|Y)+<nu,err> +0.5|err|^2
		errz = pz - np.sum(pzcx*px[None,:],axis=1) # this is right
		pzcy = pzcx @ pxcy
		record_lval[itcnt % len(record_lval)] = (1-gamma)*np.sum(pz*np.log(pz))+gamma*np.sum(pzcx*px[None,:]*np.log(pzcx))\
											    -np.sum( pzcy*py[None,:]*np.log(pzcy))+np.sum(dual_z*errz)\
											    +0.5*pen_c*(np.linalg.norm(errz)**2)
		itcnt +=1
		# relax step
		dual_drs_z = dual_z - (1-drs_relax_coeff)*pen_c*(errz)
		# grad z
		(grad_z,_) = grad_pz_obj(pz,pzcx,dual_drs_z)
		ss_z = gd.validStepSize(pz,-grad_z,_ls_init,_bk_beta)
		if ss_z == 0:
			break
		#arm_ss_z = gd.armijoStepSize(pz,-grad_z,ss_z,_bk_beta,1e-4,func_pz_obj,grad_pz_obj,
		#							**{'pzcx':pzcx,'mu_z':dual_drs_z},)
		new_pz = pz - grad_z * ss_z
		#if arm_ss_z == 0:
		#	arm_ss_z = ss_z
		#new_pz = pz - grad_z * arm_ss_z
		# dual ascend
		dual_z = dual_drs_z + pen_c * (new_pz - np.sum(pzcx*px[None,:],axis=1))
		# grad_x
		(grad_x,_) = grad_pzcx_obj(pzcx,pz,dual_z)
		ss_x = gd.validStepSize(pzcx,-grad_x,_ls_init,_bk_beta)
		if ss_x == 0:
			break
		new_pzcx = pzcx - ss_x * grad_x
		#arm_ss_x = gd.armijoStepSize(pzcx,-grad_x,ss_x,_bk_beta,1e-4,func_pzcx_obj,grad_pzcx_obj,
		#							**{'pz':new_pz,'mu_z':dual_z})
		#if arm_ss_x == 0:
		#	arm_ss_x = ss_x
		#new_pzcx = pzcx - arm_ss_x * grad_x
		# convergence conditions
		errz = new_pz - np.sum(new_pzcx * px[None,:],axis=1)
		dtv=  0.5*np.sum(np.fabs(errz))
		if np.all(np.array(dtv < conv_thres)):
			flag_valid = True
			break
		else:
			pzcx = new_pzcx
			pz = new_pz
	pzcy = pzcx @ pxcy
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
	outputdict = {'prob_zcx':pzcx,'prob_z':pz,'prob_zcy':pzcy,
			'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':flag_valid}
	if _record_flag:
		outputdict['val_record'] = record_lval[:itcnt]
	return outputdict