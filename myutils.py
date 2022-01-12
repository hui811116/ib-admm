import numpy as np
import sys


def getAlgList(mode='all'):
	if mode == 'all':
		return ['orig','gd','sec','dev','bayat','mv','drs','acc_drs','drs_mark']
	elif mode == 'penalty':
		return ['dev','bayat','mv','drs','acc_drs','drs_mark']
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
	return '{}_{}_result'.format(kwargs['method'],kwargs['dataset'])

def genExpName(**kwargs):
	method = kwargs['method']
	dataset = kwargs['dataset']
	if method in ['alm','sec','dev']:
		penalty = '{:.2f}'.format(kwargs['penalty'])
		omega = '{:.2f}'.format(kwargs['omega'])
		return '{}_{}_o{}_c{}'.format(method,dataset,str(omega),str(penalty))
	elif method in ['bayat','mv']:
		penalty = '{:.2f}'.format(kwargs['penalty'])
		return '{}_{}_c{}'.format(method,dataset,str(penalty))
	elif method in ['drs','drs_acc','drs_mark']:
		penalty = '{:.2f}'.format(kwargs['penalty'])
		relax   = '{:.2f}'.format(kwargs['relax'])
		return '{}_{}_r{}_c{}'.format(method,dataset,str(relax),str(penalty))
	elif method in ['gd','orig']:
		return '{}_{}_exp'.format(method,dataset)
	else:
		sys.exit('undefined method {} for experiment'.format(method))

def genStatus(**kwargs):
	method = kwargs['method']
	dataset = kwargs['dataset']
	if method in ['alm','sec','dev']:
		#penalty = '{:.2f}'.format(kwargs['penalty'])
		#omega = '{:.2f}'.format(kwargs['omega'])
		return 'method:'+method+'---'+'beta,{beta:>6.3f}, penalty, {penalty_coeff:>6.2f}, omega, {breg_omega:>6.2f}, Progress:'
	elif method in ['bayat','mv']:
		#penalty = '{:.2f}'.format(kwargs['penalty'])
		return 'method:'+method+'---'+'beta,{beta:>6.3f}, penalty, {penalty_coeff:>6.2f}, Progress:'
	elif method in ['drs','drs_mark','drs_acc']:
		#penalty = '{:.2f}'.format(kwargs['penalty'])
		#relax   = '{:.2f}'.format(kwargs['relax'])
		return 'method:'+method+'---'+'beta,{beta:>6.3f}, penalty {penalty_coeff:>6.2f}, relax, {relax_coeff:>6.2f}, Progress:'
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
	elif method in ['bayat','mv']:
		return r"{}, $c={:}$".format(method,kwargs['penalty'])
	elif method in ['drs','drs_acc','drs_mark']:
		return r"{}, $c={:}, \alpha={:}$".format(method,kwargs['penalty'],kwargs['relax'])
	elif method == 'dev':
		return r"ours, $c={:}, \omega={:}$".format(kwargs['penalty'],kwargs['omega'])

def getLsSchedule(ls_init):
	# TODO: design a better learning rate scheduler
	#       for now, it suffice to compare all methods with this naive implementation.
	return [(500,0.5*ls_init),(1000,0.1*ls_init),(5000,0.05*ls_init)]
