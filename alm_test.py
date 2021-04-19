import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import myutils as ut
import argparse
import mydata as dt
import myalg as alg
import myentropy as ent
import graddescent as gd
import pprint
import copy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

parser = argparse.ArgumentParser()
parser.add_argument("method",type=str,choices=['orig','dev'],help="select the method")
parser.add_argument('-dataset',type=str,choices=['synWu'],default='synWu',help='select the dataset')
parser.add_argument("-beta",type=float,help='the IB beta',default=1.0)
#parser.add_argument('-ntime',type=int,help='run how many times per beta',default=25)
parser.add_argument('-penalty',type=float,help='penalty coefficient',default=4.0)
parser.add_argument('-omega',type=float,help='Bregman Regularization coefficient',default=0.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=1e-5)
parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('-maxiter',type=int,help='The maximum number of iteration per run',default=10000)
parser.add_argument("-v",'--verbose',help='printing the log and parameters along the execution',action='count',default=0)



# the goal of this is to plot the transcient behavior of the iteration
# copy from myalg.py
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
	pz = px[sel_idx[:nz]]
	pz /= np.sum(pz)
	pzcx = rs.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	
	pycz = pycx@ np.transpose(1/pz[:,None]*pzcx*px[None,:])
	
	# record min pzcx over x\in X and z\in Z?
	rec_pzcx_min = np.zeros(max_iter,)
	rec_pz_min   = np.zeros(max_iter,)
	rec_pzcy_min = np.zeros(max_iter,)
	# ready to start
	itcnt = 0
	while itcnt<max_iter:
		# recording
		rec_pzcx_min[itcnt] = np.amin(pzcx)
		rec_pz_min[itcnt]   = np.amin(pz)
		rec_pzcy_min[itcnt] = np.amin(pzcx@pxcy)
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
	return {'prob_zcx':pzcx,'prob_ycz':pycz,'niter':itcnt,
			'IXZ':mixz,'IYZ':miyz,'valid':True,
			'pzcx_min':rec_pzcx_min[:itcnt],
			'pz_min'  :rec_pz_min[:itcnt],
			'pzcy_min':rec_pzcy_min[:itcnt],
			}

def ib_alm_dev(pxy,qlevel,conv_thres,beta,max_iter,**kwargs):
	_bk_beta = kwargs['backtracking_beta']
	#_ls_init = kwargs['line_search_init']
	_ls_init = 0.05
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
	pz = px[sel_idx[:nz]]
	pz /= np.sum(pz)
	pz_delay = copy.copy(pz)
	pzcx = rs.rand(nz,nx)
	pzcx = pzcx * (1./np.sum(pzcx,axis=0))[None,:]
	pzcx_delay = copy.copy(pzcx)
	pzcx[:nz,:] = pycx[:,sel_idx[:nz]]
	pzcy = pzcx@pxcy
	pzcy = pzcy * (1./np.sum(pzcx,axis=0))[None,:]
	# ready to start
	# record min pzcx over x\in X and z\in Z?
	rec_pzcx_min = np.zeros(max_iter,)
	rec_pz_min   = np.zeros(max_iter,)
	rec_pzcy_min = np.zeros(max_iter,)

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
		# RECORD
		rec_pz_min[itcnt] = np.amin(pz)
		rec_pzcx_min[itcnt] = np.amin(pzcx)
		rec_pzcy_min[itcnt] = np.amin(pzcy)
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
		
		# FIXME debugging
		#pzcy = pzcy/ np.sum(pzcy,axis=0)[:,None]


		# probability update
		pzcx_delay = copy.copy(pzcx)
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
	
	return {'prob_zcx':pzcx,'prob_z':pz,'niter':itcnt,
			'IXZ':mixz,'IYZ':miyz,'valid':isvalid,
			'pzcx_min':rec_pzcx_min[:itcnt],
			'pz_min'  :rec_pz_min[:itcnt],
			'pzcy_min':rec_pzcy_min[:itcnt],}

# ------------------------------------------------------------------------------------------------------
# main loop
# -------------------------------------------------------------------------

args = parser.parse_args()
argdict = vars(args)

# fixed parameters
_sys_parms = {
	#'backtracking_alpha': 0.45,
	#'backtracking_beta' : args.sscale,
	'backtracking_beta' : 0.8,
	'line_search_init'  : 0.125,
	'max_iter'          : args.maxiter,
	'conv_thres'        : args.thres,
	'penalty_coeff'     : args.penalty,
	'breg_omega'        : args.omega,
	'rand_seed'         : args.seed,
}

if args.verbose:
	pprint.pprint(_sys_parms)


d_beta = args.beta
d_pxy_info = dt.datasetSelect(args.dataset)

if args.method == "orig":
	d_alg = ib_orig
elif args.method == "dev":
	d_alg =ib_alm_dev 
else:
	sys.exit("method {} no supported".format(args.method))

d_pxy = d_pxy_info['pxy']
px = np.sum(d_pxy,axis=1)
py = np.sum(d_pxy,axis=0)
pycx = np.transpose(np.diag(1./px)@d_pxy)
pxcy = d_pxy@np.diag(1./py)
# for clarity, only use single run
algargs = {'beta':d_beta,'qlevel':d_pxy.shape[1],**_sys_parms,}
ut.checkAlgArgs(**algargs)
ib_res = d_alg(**{'pxy':d_pxy,**algargs})
print('trial:beta={beta:>6.2f}, IXZ={IXZ:<8.4f}, IYZ={IYZ:<8.4f}, niter={niter:<5d}, converge={valid:<5}'.format(
		**{'beta':d_beta,**ib_res}))

# plotting the results

xx = np.arange(0,ib_res['niter'])
#plt.plot(xx,ib_res['pzcx_min'],'-rx',label=r"min $p_{z|x}$")
plt.plot(xx,ib_res['pz_min'],'-b+',label=r"min $p_z,Bp_{z|x}$")
plt.plot(xx,ib_res['pzcy_min'],'-d',color='gray',label=r"min $p_{z|y}$")
plt.plot(xx,np.repeat(np.amin(pxcy),ib_res['niter']),'--m',label=r"min $p_{x|y}$")
plt.plot(xx,np.repeat(np.amin(pycx),ib_res['niter']),'-.k',label=r"min $p_{y|x}$")
plt.plot(xx,np.repeat(np.amin(px),ib_res['niter']),'--c',label=r"min $p_x$")
plt.plot(xx,np.repeat(np.amin(py),ib_res['niter']),':g',label=r"min $p_y$")
plt.plot(xx,np.repeat(np.amin(pycx/py[:,None]),ib_res['niter']),'-.y',label=r"min $p_{y|x}/p_{y}$")
plt.yscale('log')
plt.legend(fontsize=14)
plt.grid()
title_tex =r"Minimum Probability over Iterations, $\beta={:.2f}$, iter={:3}".format(d_beta,ib_res['niter'])
plt.title(title_tex)
plt.xlabel("Iteration")
plt.ylabel("Min probability")
plt.show()


