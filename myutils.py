import numpy as np
import sys


def getAlgList(mode='all'):
	if mode == 'all':
		return ['orig','gd','sec','dev','bayat']
	elif mode == 'penalty':
		return ['dev','bayat']
	elif mode == 'trans':
		return ['orig','dev']
	else:
		sys.exit('Undefined mode {}'.format(mode))

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

def genExpName(**kwargs):
	method = kwargs['method']
	dataset = kwargs['dataset']
	if method in ['alm','sec','dev']:
		penalty = '{:.2f}'.format(kwargs['penalty'])
		omega = '{:.2f}'.format(kwargs['omega'])
		return '{}_{}_o{}_c{}'.format(method,dataset,str(omega),str(penalty))
	elif method == 'bayat':
		penalty = '{:.2f}'.format(kwargs['penalty'])
		return '{}_{}_c{}'.format(method,dataset,str(penalty))
	elif method in ['gd','orig']:
		return '{}_{}_exp'.format(method,dataset)
	else:
		sys.exit('undefined method {} for experiment'.format(method))

def genStatus(**kwargs):
	method = kwargs['method']
	dataset = kwargs['dataset']
	if method in ['alm','sec','dev']:
		penalty = '{:.2f}'.format(kwargs['penalty'])
		omega = '{:.2f}'.format(kwargs['omega'])
		return 'method:'+method+'---'+'beta,{beta:>6.3f}, penalty, {penalty_coeff:>6.2f}, omega, {breg_omega:>6.2f}, Progress:'
	elif method == 'bayat':
		penalty = '{:.2f}'.format(kwargs['penalty'])
		return 'method:'+method+'---'+'beta,{beta:>6.3f}, penalty, {penalty_coeff:>6.2f}, Progress:'
	elif method in ['gd','orig']:
		return 'method:'+method+'---'+'beta,{beta:>6.3f}, Progress:'
	else:
		sys.exit('undefined method {} for experiment'.format(method))

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
