import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import sys
import os
import pickle
import copy


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
		diff = 0.5* np.sum(np.fabs(new_pzcx-pzcx))
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