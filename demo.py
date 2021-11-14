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

d_base = os.getcwd()

available_algs = ut.getAlgList()
datasetlist    = dt.getDatasetList()

parser = argparse.ArgumentParser()
parser.add_argument("method",type=str,choices=available_algs,help="select the method")
parser.add_argument('-dataset',type=str,choices=datasetlist,default='synWu',help='select the dataset')
parser.add_argument("-beta",type=float,help='the IB beta',default=1.0)
parser.add_argument('-ntime',type=int,help='run how many times per beta',default=25)
parser.add_argument('-penalty',type=float,help='penalty coefficient',default=4.0)
parser.add_argument('-omega',type=float,help='Bregman Regularization coefficient',default=0.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=1e-5)
parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument("-v",'--verbose',help='printing the log and parameters along the execution',action='count',default=0)


args = parser.parse_args()
argdict = vars(args)

# fixed parameters
_sys_parms = {
	'backtracking_alpha': 0.45,
	'backtracking_beta' : 0.5,
	'line_search_init'  : 0.05,
	'max_iter'          : 50000,
	'conv_thres'        : args.thres,
	'penalty_coeff'     : args.penalty,
	'breg_omega'        : args.omega,
	'rand_seed'         : args.seed,
}

if args.verbose:
	pprint.pprint(_sys_parms)


d_beta = args.beta
d_pxy_info = dt.datasetSelect(args.dataset)
d_alg = alg.selectAlg(args.method)
print('Method:{}'.format(args.method))

d_pxy = d_pxy_info['pxy']
for ni in range(args.ntime):
	algargs = {'beta':d_beta,'qlevel':d_pxy.shape[1],**_sys_parms,}
	ut.checkAlgArgs(**algargs)
	ib_res = d_alg(**{'pxy':d_pxy,**algargs})
	print('{nt:4d} trial:beta={beta:>6.2f}, IXZ={IXZ:<8.4f}, IYZ={IYZ:<8.4f}, niter={niter:<5d}, converge={valid:<5}'.format(
		**{'nt':ni,'beta':d_beta,**ib_res}))
print('demo complete')