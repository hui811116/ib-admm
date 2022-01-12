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
parser.add_argument('output',type=str,help='specify the name of the directory to save results')
parser.add_argument('-dataset',type=str,choices=datasetlist,default='synMy',help='select the dataset')
parser.add_argument("-minbeta",type=float,help='the minimum beta to sweep',default=1.0)
parser.add_argument("-maxbeta",type=float,help='the maximum beta to sweep',default=10.0)
parser.add_argument('-numbeta',type=int,help='the geometric spacing between beta_min and beta_max',default=16)
parser.add_argument('-ntime',type=int,help='run how many times per beta',default=100)
parser.add_argument('-penalty',type=float,help='penalty coefficient',default=4.0)
parser.add_argument('-omega',type=float,help='Bregman Regularization coefficient',default=2.0)
parser.add_argument('-relax',type=float,help='Relaxation parameter for DRS',default=1.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('-sinit',type=float,help='Initial step size for line search',default=0.5)
parser.add_argument('-sscale',type=float,help='Step size line search scaling',default=0.25)
parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('-maxiter',type=int,help='The maximum number of iteration per run',default=25000)
parser.add_argument("-v",'--verbose',help='printing the log and parameters along the execution',action='count',default=0)
parser.add_argument("-s",'--sound',help='remind the end of program by a sinewave sound',action='count',default=0)


args = parser.parse_args()
argdict = vars(args)

# fixed parameters
_sys_parms = {
	#'backtracking_alpha': 0.45,
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


d_beta_range = np.geomspace(args.minbeta,args.maxbeta,num=args.numbeta)
d_pxy_info = dt.datasetSelect(args.dataset)
d_alg = alg.selectAlg(args.method)
print('Method:{}'.format(args.method))

def runSim(ibalg,betarange,pxy,nrun,**kwargs):
	status_template = kwargs['status_tex']
	result_all = []
	time_start = time.time()
	for bidx,beta in enumerate(betarange):
		algargs = {'beta':beta,'qlevel':pxy.shape[1],**kwargs,}
		ut.checkAlgArgs(**algargs)
		tmp_status = status_template.format(**algargs)
		tmp_result = []
		time_per_run = time.time()
		conv_cnt = 0
		for it in range(nrun):
			print('\rCurrent status: {}{:>6.2f}% ({:>5}/{:>5} iterations)'.format(
				tmp_status,100*it/nrun,it,nrun),end='',flush=True)
			ib_res = ibalg(**{'pxy':pxy,**algargs})
			conv_cnt += int(ib_res['valid'])
			tmp_result.append(ib_res)
		time_per_run = (time.time()-time_per_run)/nrun
		avg_conv_rate= conv_cnt / nrun
		print('\r{:<200}'.format(tmp_status+'complete'),end='\r',flush=True)
		result_all.append({'beta':beta,'result':tmp_result,'avg_time':time_per_run,'avg_conv':avg_conv_rate}) 
	print(' '*200+'\r',end='',flush=True)
	print('time elapsed: {:>16.8f} seconds'.format(time.time()-time_start))
	return result_all
# main algorithm
tmp_status   = ut.genStatus(**argdict)
result_all = runSim(d_alg,d_beta_range,d_pxy_info['pxy'],args.ntime,**{'status_tex':tmp_status,**_sys_parms})

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

if args.sound:
	duration = 1 # seconds
	freq= 440 #khz
	os.system('play -nq -t alsa synth {} sine {}'.format(duration,freq))
