import numpy as np
import sys


def calc_mi(pzcx,px):
	pz = pzcx@px
	inner_ker = (1./pz)[:,None]*pzcx
	maps = (inner_ker == 0)
	ker_val = np.where(maps,0.0,np.log(inner_ker+1e-7))
	return np.sum(pzcx*px[None,:]*ker_val) # avoiding overflow

def checkAlgArgs(**kwargs):
	arglist = ['qlevel',
				'conv_thres',
				'beta',
				'max_iter',
				]
	ordered_args={}
	for sid, item in enumerate(arglist):
		if kwargs.get(item,False):
			ordered_args[item] = kwargs[item]
		else:
			sys.exit('ERROR: the argument {} required by ib algorithms is missing'.format(item))
	return True

def pxy2allprob(pxy):
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	pxcy = pxy*(1/py)[None,:]
	pycx = np.transpose((1/px)[:,None]*pxy)
	return {'px':px,'py':py,'pxcy':pxcy,'pycx':pycx}

def genOutName(**kwargs):
	method = kwargs['method']
	if method == 'orig':
		return 'orig_{}_result'.format(kwargs['dataset'])
	elif method == 'gd':
		return 'gd_{}_result'.format(kwargs['dataset'])
	elif method == 'alm':
		return 'alm_{}_result'.format(kwargs['dataset'])
	elif method == 'sec':
		return 'sec_{}_result'.format(kwargs['dataset'])
	elif method == 'dev':
		return 'dev_{}_result'.format(kwargs['dataset'])
	elif method == 'bayat':
		return 'bayat_{}_result'.format(kwargs['dataset'])
	else:
		sys.exit('undefined method {}'.format(method))

def getFigLabel(**kwargs):
	method = kwargs['method']
	if method == 'orig':
		return 'IB-orig'
	elif method == 'gd':
		return 'IB-gd'
	elif method == 'bayat':
		return r"bayat, $c={:}$".format(kwargs['penalty'])
	elif method == 'dev':
		return r"ours, $c={:}, \omega={:}$".format(kwargs['penalty'],kwargs['omega'])
