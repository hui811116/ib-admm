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


parser = argparse.ArgumentParser()
parser.add_argument("method",type=str,choices=['orig','gd','alm','sec'],help="select the method")
parser.add_argument('output',type=str,help='specify the name of the directory to save results')
parser.add_argument('-dataset',type=str,choices=['synWu'],default='synWu',help='select the dataset')
parser.add_argument("-minbeta",type=float,help='the minimum beta to sweep',default=1.0)
parser.add_argument("-maxbeta",type=float,help='the maximum beta to sweep',default=10.0)
parser.add_argument('-numbeta',type=int,help='the geometric spacing between beta_min and beta_max',default=16)
parser.add_argument('-ntime',type=int,help='run how many times per beta',default=25)
parser.add_argument('-penalty',type=float,help='penalty coefficient',default=4.0)
parser.add_argument('-omega',type=float,help='Bregman Regularization coefficient',default=0.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=1e-5)
parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=123)
parser.add_argument("-v",'--verbose',help='printing the log and parameters along the execution',action='count',default=0)


args = parser.parse_args()
argdict = vars(args)

# fixed parameters
_sys_parms = {
	'backtracking_alpha': 0.45,
	'backtracking_beta' : 0.5,
	'max_iter'          : 25000,
	'conv_thres'        : args.thres,
	'penalty_coeff'     : args.penalty,
	'breg_omega'        : args.omega,
}

if args.verbose:
	pprint.pprint(_sys_parms)


d_beta_range = np.geomspace(args.minbeta,args.maxbeta,num=args.numbeta)
d_pxy_info = dt.datasetSelect(args.dataset)
d_alg = alg.selectAlg(args.method)
print('Method:{}'.format(args.method))

def runSim(ibalg,betarange,pxy,nrun,**kwargs):
	result_all = []
	for bidx,beta in enumerate(betarange):
		print('beta:{:<50.4f}'.format(beta))
		algargs = {'beta':beta,'qlevel':pxy.shape[0],**kwargs,}
		ut.checkAlgArgs(**algargs)
		tmp_result = []
		for it in range(nrun):
			print('\rCurrent progress: {:4.2f}% ({:>5}/{:>5} iterations)'.format(100*it/nrun,it,nrun),end='',flush=True)
			ib_res = ibalg(**{'pxy':pxy,**algargs})
			tmp_result.append(ib_res)
		print('{:<50}'.format(''),end='\r',flush=True)
		result_all.append({'beta':beta,'result':tmp_result})
	return result_all
# main algorithm
result_all = runSim(d_alg,d_beta_range,d_pxy_info['pxy'],args.ntime,**_sys_parms)

d_file_name = ut.genOutName(**argdict) + '.pkl'
d_save_dir = os.path.join(d_base,args.output)
if not os.path.isdir(d_save_dir):
	os.makedirs(d_save_dir,exist_ok=True)

print('saving the result to:{}/{}'.format(d_save_dir,d_file_name))
with open(os.path.join(d_save_dir,d_file_name),'wb') as fid:
	pickle.dump(result_all,fid)
with open(os.path.join(d_save_dir,'arguments.pkl'),'wb') as fid:
	pickle.dump(argdict,fid)
with open(os.path.join(d_save_dir,'sysParams.pkl'),'wb') as fid:
	pickle.dump(_sys_parms,fid)
