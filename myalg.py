import numpy as np
import sys
import myutils as ut
import myentropy as ent
import graddescent as gd
import copy

# MACRO

def selectAlg(method,**kwargs):
	if method =='orig':
		return ib_orig
	elif method == 'gd':
		return ib_gd
	elif method == 'alm':
		return ib_alm_breg
	return None

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
	# this version, compared to previous ones is that it uses bregman divergence
	# which in our case, is the KL divergence on the pz updates.
	# This introduces a hyperparameter on the Bregman divergence
	# system parameters
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
	pzcx_delay = np.ones((nz,nx))/nz
	pzcy = pzcx@pxcy
	# ready to start
	itcnt = 0
	# gradient descent control
	_step_size = 1.0
	ss_pzcx = copy.copy(_step_size)
	# defined in global variables
	isvalid = True
	_parm_c = kwargs['penalty_coeff']
	_parm_mu_z = np.zeros((nz,))
	# bregman divergence parameter
	dbd_omega = kwargs['breg_omega']
	_breg_kappa = 0
	# stepsize control
	pz_func_obj = ent.getPzFuncObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pz_grad_obj = ent.getPzGradObj(beta,px,_parm_c,dbd_omega,use_breg=True)
	pzcx_func_obj = ent.getPzcxFuncObj(beta,px,py,pxcy,pycx,_parm_c,_breg_kappa,False)
	pzcx_grad_obj = ent.getPzcxGradObj(beta,px,py,pxcy,pycx,_parm_c,_breg_kappa,False)


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
		#grad_pzcx = (np.log(pzcx)+1-beta-beta*np.transpose(pycx.T@np.log(pzcy.T))
		#			-_parm_mu_z[:,np.newaxis]-_parm_c*pen_z[:,np.newaxis])*px[None,:]
		mean_grad_pzcx = pzcx_grad_obj(pz,pzcx,_parm_mu_z)
		# calculate update norm
		# FIXME
		ss_pzcx = gd.validStepSize(pzcx,-1 * mean_grad_pzcx, ss_pzcx,_bk_beta)
		'''
		tmp_dict = {
			'pz': pz,
			'mu_z': _parm_mu_z,
			'pzcx_delay': pzcx_delay,
		}
		ss_pzcx = gd.backtrackStepSize(pzcx,-mean_grad_pzcx,1.0,_bk_alpha,_bk_beta,pzcx_func_obj,pzcx_grad_obj,**tmp_dict)
		'''
		if ss_pzcx == 0:
			isvalid = False
			break
		pzcx_delay = copy.copy(pzcx)
		new_pzcx = pzcx - ss_pzcx * mean_grad_pzcx
		mean_grad_pz = pz_grad_obj(pz,new_pzcx,_parm_mu_z,pz_delay)
		# gradient mean
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
		ss_pz = gd.backtrackStepSize(pz,-mean_grad_pz,1.0,_bk_alpha,_bk_beta,pz_func_obj,pz_grad_obj,**tmp_dict)
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



def ib_gd(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	# system
	_bk_beta = kwargs['backtracking_beta']
	(nx,ny) = pxy.shape
	nz = qlevel
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pycx = np.transpose((1./px)[:,None]*pxy)
	pxcy = pxy*(1./py)[None,:]
	# on IB, the initialization matters
	# use random start
	#pzcx = np.random.rand(nz,nx)
	#pzcx = (1/np.sum(pzcx,axis=0))[None,:]*pzcx
	'''
	pz = pzcx @ px
	pzcy = pzcx @ pxcy
	pycz = np.transpose((1/pz)[:,None]*pzcy*py[None,:])
	'''
	sel_idx = np.random.permutation(nx)
	pycz = pycx[:,sel_idx[:qlevel]]
	pz = px[sel_idx[:qlevel]]
	pz /= np.sum(pz)
	
	pzcx = np.random.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx[:nz,:] = pycx
	pycz = pycx@ np.transpose(1/pz[:,None]*pzcx*px[None,:])


	# naive method
	_step_size = 1.0
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
		tmp_step = gd.validStepSize(pzcx,-grad,tmp_step,_bk_beta)
		if tmp_step == 0:
			flag_valid = False
			break
		new_pzcx = pzcx - tmp_step * zm_grad
		#grad_norm = np.linalg.norm(zm_grad,axis=0)
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
	mixz = ut.calc_mi(pzcx,px)
	miyz = ut.calc_mi(pycz,pz)
	if not flag_valid:
		mixz = 0
		miyz = 0
	return {'prob_zcx':pzcx,'prob_ycz':pycz,'niter':itcnt,'valid':flag_valid,'IXZ':mixz,'IYZ':miyz}