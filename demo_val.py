import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import sys
import os
import pickle
import copy
import argparse
import mydata as dt
import myalg as alg
import myutils as ut
import myentropy as ent
import graddescent as gd
import pprint
from scipy.io import savemat


d_base = os.getcwd()

available_algs = ut.getAlgList()
datasetlist    = dt.getDatasetList()

parser = argparse.ArgumentParser()
parser.add_argument("method",type=str,choices=available_algs,help="select the method")
parser.add_argument('-dataset',type=str,choices=datasetlist,default='uciHeart',help='select the dataset')
parser.add_argument("-beta",type=float,help='the IB beta',default=3.0)
parser.add_argument('-ntime',type=int,help='run how many times per beta',default=25)
parser.add_argument('-penalty',type=float,help='penalty coefficient',default=64.0)
parser.add_argument('-omega',type=float,help='Bregman Regularization coefficient',default=0.0)
parser.add_argument('-relax',type=float,help='Relaxation parameter for DRS',default=1.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=5e-4)
parser.add_argument('-sinit',type=float,help='initial step size',default=1e-2)
parser.add_argument('-sscale',type=float,help='Scaling of step size',default=0.25)
parser.add_argument('-maxiter',type=int,help='Maximum number of iterations',default=50000)
parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument("-v",'--verbose',help='printing the log and parameters along the execution',action='count',default=0)


args = parser.parse_args()
argdict = vars(args)

# fixed parameters
_sys_parms = {
	'backtracking_alpha': 0.45,
	'backtracking_beta' : args.sscale,
	'line_search_init'  : args.sinit,
	'max_iter'          : args.maxiter,
	'conv_thres'        : args.thres,
	'penalty_coeff'     : args.penalty,
	'breg_omega'        : args.omega,
	'rand_seed'         : args.seed,
	'relax_coeff'       : args.relax,
}

if args.verbose:
	pprint.pprint(_sys_parms)


d_beta = args.beta
d_pxy_info = dt.datasetSelect(args.dataset)
d_alg = alg.selectAlg(args.method)
print('Method:{}'.format(args.method))

d_pxy = d_pxy_info['pxy']
best_IYZ = 0
best_result = None
for ni in range(args.ntime):
	algargs = {'beta':d_beta,'qlevel':d_pxy.shape[1],'record':True,**_sys_parms,}
	ut.checkAlgArgs(**algargs)
	ib_res = d_alg(**{'pxy':d_pxy,**algargs})
	if ib_res['valid'] and ib_res['IYZ']>best_IYZ:
		best_IYZ = ib_res['IYZ']
		best_result = ib_res['val_record']

	print('{nt:4d} trial:beta={beta:>6.2f}, IXZ={IXZ:<8.4f}, IYZ={IYZ:<8.4f}, niter={niter:<5d}, converge={valid:<5}'.format(
		**{'nt':ni,'beta':d_beta,**ib_res}))
filename = 'record_{method:}_r{relax:}_c{penalty:}_si{sinit:}_beta{beta:}'.format(**argdict)
matlab_mat = {'beta':argdict['beta'],'penalty':argdict['penalty'],'relax':argdict['relax'],'data':best_result}
savepath = os.path.join(d_base,filename+'.mat')
savemat(savepath,matlab_mat)
print('demo complete, result saved as:{}'.format(savepath))


