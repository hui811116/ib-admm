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
	#sel_idx = np.random.permutation(nx)
	sel_idx = rs.permutation(nx)
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	#pzcx = np.random.rand(nz,nx)
	pzcx = rs.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx[:nz,:] = pycx
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

def ib_alm_breg(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):	
	_bk_beta = kwargs['backtracking_beta']
	_ls_init = kwargs['line_search_init']
	#_bk_alpha = kwargs['backtracking_alpha']
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
	pzcx = pzcx / np.sum(pzcx,axis=0)[None,:]
	pzcx[:nz,:] = pycx
	pzcy = pzcx@pxcy
	pzcy = pzcy / np.sum(pzcy,axis=0)[None,:]
	# ready to start
	itcnt = 0
	# gradient descent control
	#_step_size = 1.0
	#ss_pzcx = copy.copy(_step_size)
	#ss_pz = copy.copy(_step_size)
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
		pen_z = pz- pzcx@px
		'''
		# stable version
		mean_grad_pzcx = pzcx_grad_obj(pzcx,pz,_parm_mu_z)
		ss_pzcx = gd.validStepSize(pzcx,-mean_grad_pzcx, ss_pzcx,_bk_beta)
		if ss_pzcx == 0:
			break
		new_pzcx = pzcx - ss_pzcx * mean_grad_pzcx
		mean_grad_pz = pz_grad_obj(pz,new_pzcx,_parm_mu_z,pz_delay)
		ss_pz = gd.validStepSize(pz,-mean_grad_pz,ss_pz,_bk_beta)
		if ss_pz == 0:
			break
		new_pz   = pz   - ss_pz * mean_grad_pz
		'''
		## developing
		
		(mean_grad_pzcx,_) = pzcx_grad_obj(pzcx,pz,_parm_mu_z)
		# calculate update norm
		# unit step size for every update
		#ss_pzcx = gd.validStepSize(pzcx,-mean_grad_pzcx, ss_pzcx,_bk_beta)
		ss_pzcx = gd.validStepSize(pzcx,-mean_grad_pzcx, 0.25,_bk_beta)
		if ss_pzcx == 0:
			break
		arm_ss_pzcx = gd.armijoStepSize(pzcx,-mean_grad_pzcx,ss_pzcx,_bk_beta,1e-4,pzcx_func_obj,pzcx_grad_obj,
									**{'pz':pz,'mu_z':_parm_mu_z},)
		if arm_ss_pzcx==0:
			arm_ss_pzcx = ss_pzcx
		new_pzcx = pzcx - arm_ss_pzcx * mean_grad_pzcx
		
		(mean_grad_pz,_) = pz_grad_obj(pz,new_pzcx,_parm_mu_z,pz_delay)
		#ss_pz = gd.validStepSize(pz,-mean_grad_pz,ss_pz,_bk_beta)
		ss_pz = gd.validStepSize(pz,-mean_grad_pz,0.1,_bk_beta)
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
		if dtv_pen < conv_thres:
			break
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
	pen_check = 0.5*np.sum(np.fabs(pz-pzcx@px))
	isvalid = (pen_check<=conv_thres)
	if not isvalid:
		mixz = 0
		miyz = 0
	return {'prob_zcx':pzcx,'prob_z':pz,'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':isvalid}

def ib_alm_dev(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	_bk_beta = kwargs['backtracking_beta']
	_ls_init = kwargs['line_search_init']
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	# DEBUG
	# FIXME: tuning the bregman regularization for pzcx, make it a parameter once complete debugging
	debug_breg_o2 = 0.0
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
	#sel_idx = np.random.permutation(nx)
	sel_idx = rs.permutation(nx)
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	pz_delay = copy.copy(pz)
	pzcx = rs.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx_delay = copy.copy(pzcx)
	pzcx[:nz,:] = pycx
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
	pzcx_func_obj = ent.devPzcxFuncObj(beta,px,py,pxcy,pycx,_parm_c,debug_breg_o2)
	pzcx_grad_obj = ent.devPzcxGradObj(beta,px,pxcy,pycx,_parm_c,debug_breg_o2)
	while itcnt < max_iter:
		itcnt+=1
		pen_z = pz- pzcx@px		
		(mean_grad_pzcx,_) = pzcx_grad_obj(pzcx,pz,_parm_mu_z,pzcx_delay)
		mean_grad_pzcx/=beta
		# calculate update norm
		# unit step size for every update
		#ss_pzcx = gd.validStepSize(pzcx,-mean_grad_pzcx, ss_pzcx,_bk_beta)
		ss_pzcx = gd.validStepSize(pzcx,-mean_grad_pzcx, _ls_init,_bk_beta)
		if ss_pzcx == 0:
			break
		arm_ss_pzcx = gd.armijoStepSize(pzcx,-mean_grad_pzcx,ss_pzcx,_bk_beta,1e-4,pzcx_func_obj,pzcx_grad_obj,
									**{'pz':pz,'mu_z':_parm_mu_z,'pzcx_delay':pzcx_delay},)
		if arm_ss_pzcx==0:
			arm_ss_pzcx = ss_pzcx
		new_pzcx = pzcx - arm_ss_pzcx * mean_grad_pzcx
		
		# TODO
		# implement the over-relaxed ADMM to update convex primal variable

		(mean_grad_pz,_) = pz_grad_obj(pz,new_pzcx,_parm_mu_z,pz_delay)
		mean_grad_pz/=beta
		#ss_pz = gd.validStepSize(pz,-mean_grad_pz,ss_pz,_bk_beta)
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
		pzcx_delay = copy.copy(pzcx)
		pzcy = new_pzcx @ pxcy
		# FIXME debugging
		#pzcy = pzcy/ np.sum(pzcy,axis=0)[:,None]

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
	rs = RandomState(MT19937(SeedSequence(kwargs['rand_seed'])))
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	'''
	pz = pzcx @ px
	pzcy = pzcx @ pxcy
	pycz = np.transpose((1/pz)[:,None]*pzcy*py[None,:])
	'''
	#sel_idx = np.random.permutation(nx)
	sel_idx = rs.permutation(nx)
	pycz = pycx[:,sel_idx[:qlevel]]
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	
	#pzcx = np.random.rand(nz,nx)
	pzcx = rs.rand(nz,nx)
	pzcx = pzcx /np.sum(pzcx,axis=0)[None,:]
	pzcx[:nz,:] = pycx
	pycz = pycx@ np.transpose(1/pz[:,None]*pzcx*px[None,:])
	pycz = pycz /np.sum(pycz,axis=0)[None,:]
	# naive method
	grad_obj = ent.getLibGDGradObj(beta,px,py,pxcy,pycx)
	func_obj = ent.getLibGDFuncObj(beta,px,py,pxcy,pycx)
	# ready to start
	itcnt = 0
	# gradient descent control
	while (itcnt < max_iter):
		itcnt+=1
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
		
		# theoretically, lambda >= 0. sometimes this is violated
		# ignore at the start of the gradient descent
		# in order to make sure the implementation is correct,
		# you should use naive method to calculate the steepest descent coefficient
		
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
	_ls_init = kwargs['line_search_init']
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
	#sel_idx = np.random.permutation(nx)
	sel_idx = rs.permutation(nx)
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	pzcx = rs.rand(nz,nx)
	#pzcx = np.random.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx[:nz,:] = pycx
	pzcy = pzcx@pxcy
	pzcy = pzcy * (1./np.sum(pzcy,axis=0))[None,:]
	'''
	# use random start instead
	pz = np.random.rand(nz)
	pz /= np.sum(pz)
	
	pzcx = np.random.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	#pzcx[:nz,:] = pycx
	#pzcy = pzcx@pxcy
	pzcy = np.random.rand(nz,ny)
	pzcy = pzcy * (1./np.sum(pzcy,axis=0))[None,:]
	'''
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
		#arm_ss_pzcx = ss_pzcx
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
		#arm_ss_pzcy = ss_pzcy
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
		#arm_ss_pz = ss_pz
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
			isvalid = True
			break
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
	pen_check = 0.5*np.sum(np.fabs(pz-pzcx@px))
	pen_check_pzcy = 0.5*(np.sum(np.fabs(pzcy-(pzcx@pxcy)/(np.sum(pzcx@pxcy,axis=0)[None,:])),axis=0))
	isvalid = (pen_check<=conv_thres and np.all(pen_check_pzcy<=conv_thres))
	

	return {'prob_zcx':pzcx,'prob_z':pz,'prob_zcy':pzcy,
			'niter':itcnt,'IXZ':mixz,'IYZ':miyz,'valid':isvalid}


'''
# KEPT FOR PAPER REVIEW
def ib_alm_breg(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):	
	_bk_beta = kwargs['backtracking_beta']
	_bk_alpha = kwargs['backtracking_alpha']
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
	_step_size = 1.0
	ss_pzcx = copy.copy(_step_size)
	ss_pz = copy.copy(_step_size)
	# defined in global variables
	isvalid = True
	_parm_c = kwargs['penalty_coeff']
	_parm_mu_z = np.zeros((nz,))
	# bregman divergence parameter
	dbd_omega = kwargs['breg_omega']
	# stepsize control
	pz_func_obj = ent.getPzFuncObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pz_grad_obj = ent.getPzGradObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pzcx_func_obj = ent.getPzcxFuncObj(beta,px,py,pxcy,pycx,_parm_c)
	
	while itcnt < max_iter:
		itcnt+=1
		pen_z = pz- pzcx@px
		grad_pzcx = (np.log(pzcx)+1-beta-beta*np.transpose(pycx.T@np.log(pzcy.T))
					-_parm_mu_z[:,np.newaxis]-_parm_c*pen_z[:,np.newaxis])*px[None,:]
		mean_grad_pzcx = grad_pzcx - np.mean(grad_pzcx,axis=0) # uniform mean, could be expectation from pzcx?
		# calculate update norm
		ss_pzcx = gd.validStepSize(pzcx,-1 * mean_grad_pzcx, ss_pzcx,_bk_beta)
		if ss_pzcx == 0:
			isvalid = False
			break
		new_pzcx = pzcx - ss_pzcx * mean_grad_pzcx
		# DEBUG: recording the improvement
		tmp_cvx_val = pz_func_obj(pz,new_pzcx,_parm_mu_z,pz_delay)
		mean_grad_pz = pz_grad_obj(pz,new_pzcx,_parm_mu_z,pz_delay)
		tmp_dict = {
			'pzcx': new_pzcx,
			'mu_z': _parm_mu_z,
			'pz_last': pz_delay,
		}
		ss_pz = gd.validStepSize(pz,-mean_grad_pz,ss_pz,_bk_beta)
		# update probabilities
		if ss_pz == 0:
			isvalid = False
			break
		new_pz   = pz   - ss_pz * mean_grad_pz
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
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pzcy,py)
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