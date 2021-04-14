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

parser = argparse.ArgumentParser()
#parser.add_argument("method",type=str,choices=['orig'],help="select the method")
parser.add_argument('-dataset',type=str,choices=['synWu'],default='synWu',help='select the dataset')
parser.add_argument("-beta",type=float,help='the IB beta',default=1.0)
#parser.add_argument('-ntime',type=int,help='run how many times per beta',default=25)
parser.add_argument('-penalty',type=float,help='penalty coefficient',default=4.0)
parser.add_argument('-omega',type=float,help='Bregman Regularization coefficient',default=0.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=1e-5)
#parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=123)
parser.add_argument("-v",'--verbose',help='printing the log and parameters along the execution',action='count',default=0)



# the goal of this is to plot the transcient behavior of the iteration
# copy from myalg.py
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

# ------------------------------------------------------------------------------------------------------
# main loop
# -------------------------------------------------------------------------

args = parser.parse_args()
argdict = vars(args)

# fixed parameters
_sys_parms = {
	'backtracking_alpha': 0.45,
	'backtracking_beta' : 0.5,
	'max_iter'          : 10000,
	'conv_thres'        : args.thres,
	'penalty_coeff'     : args.penalty,
	'breg_omega'        : args.omega,
}

if args.verbose:
	pprint.pprint(_sys_parms)


d_beta = args.beta
d_pxy_info = dt.datasetSelect(args.dataset)
#d_alg = alg.selectAlg(args.method)
d_alg = ib_orig
#print('Method:{}'.format(args.method))

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
plt.plot(xx,ib_res['pzcx_min'],'-rx',label=r"min $p_{z|x}$")
plt.plot(xx,ib_res['pz_min'],'-b+',label=r"min $p_{z}$")
plt.plot(xx,ib_res['pzcy_min'],'--d',color='gray',label=r"min $p_{z|y}$")
plt.plot(xx,np.repeat(np.amin(pxcy),ib_res['niter']),'-m^',label=r"min $p_{x|y}$")
plt.plot(xx,np.repeat(np.amin(pycx),ib_res['niter']),'-ks',label=r"min $p_{y|x}$")
plt.plot(xx,np.repeat(np.amin(px),ib_res['niter']),'--co',label=r"min $p_x$")
plt.plot(xx,np.repeat(np.amin(py),ib_res['niter']),':g*',label=r"min $p_y$")
plt.yscale('log')
plt.legend()
plt.grid()
title_tex =r"Minimum Probability over Iterations, $\beta={:.2f}$, iter={:3}".format(d_beta,ib_res['niter'])
plt.title(title_tex)
plt.xlabel("Iteration")
plt.ylabel("Min probability")
plt.show()


