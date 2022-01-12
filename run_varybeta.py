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
available_algs = ut.getAlgList(mode='penalty')
datasetlist    = dt.getDatasetList()

parser = argparse.ArgumentParser()
parser.add_argument("method",type=str,choices=available_algs,help="select the method")
parser.add_argument('output',type=str,help='specify the name of the directory to save results')
parser.add_argument('-dataset',type=str,choices=datasetlist,default='synWu',help='select the dataset')
parser.add_argument("-minbeta",type=float,help='the minimum beta to sweep',default=1.0)
parser.add_argument("-maxbeta",type=float,help='the maximum beta to sweep',default=10.0)
parser.add_argument('-numbeta',type=int,help='the geometric spacing between beta_min and beta_max',default=16)
parser.add_argument('-ntime',type=int,help='run how many times per beta',default=100)
parser.add_argument('-minpenalty',type=float,help='min penalty coefficient',default=4.0)
parser.add_argument('-maxpenalty',type=float,help='max penalty coefficient',default=128.0)
parser.add_argument('-steppenalty',type=float,help="the penalty range between penalty_min and penalty_max",default=8.0)
parser.add_argument('-relax',type=float,help='Relaxation parameter for DRS',default=1.0)
parser.add_argument('-omega',type=float,help='Bregman Regularization coefficient',default=2.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('-sinit',type=float,help='Initial step size for line search',default=0.5)
parser.add_argument('-sscale',type=float,help='Step size line search scaling',default=0.25)
parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('-maxiter',type=int,help='The maximum number of iteration per run',default=25000)
parser.add_argument('-convrate',type=float,help='target convergence rate',default=0.95)
parser.add_argument("-v",'--verbose',help='printing the log and parameters along the execution',action='count',default=0)


args = parser.parse_args()
argdict = vars(args)

# fixed parameters
_sys_parms = {
	'backtracking_beta' : args.sscale,
	'line_search_init'  : args.sinit,
	'max_iter'          : args.maxiter,
	'conv_thres'        : args.thres,
	'breg_omega'        : args.omega,
	'rand_seed'         : args.seed,
	'relax_coeff'       : args.relax,
	'conv_rate'         : args.convrate,
}

if args.verbose:
	pprint.pprint(_sys_parms)


d_beta_range = np.geomspace(args.minbeta,args.maxbeta,num=args.numbeta)
d_penalty_range = np.arange(args.minpenalty,args.maxpenalty,args.steppenalty).astype('float32')
d_pxy_info = dt.datasetSelect(args.dataset)
d_alg = alg.selectAlg(args.method)

def runSim(ibalg,betarange,penalty_range,pxy,nrun,**kwargs):
	status_template = kwargs['status_tex']
	rate_thres = kwargs['conv_rate']
	result_all = []
	time_start = time.time()
	# start with a small penalty coefficient, stop until reaching the first case with rate > thres
	for bidx,beta in enumerate(betarange):
		pen_idx_cnt = 0
		tmp_penalty = penalty_range[pen_idx_cnt]
		algargs = {'beta':beta,'qlevel':pxy.shape[1],'penalty_coeff':tmp_penalty,**kwargs,}
		ut.checkAlgArgs(**algargs)
		tmp_status = status_template.format(**algargs)
		tmp_result = []
		time_per_run = time.time()
		conv_cnt = 0
		run_cnt = 0
		early_stop_limit = int(nrun*( 1-rate_thres)) # if accumulated x tiems ... consider failed
		itcnt = 0
		while True:
			itcnt += 1
			print('\rCurrent status: {}{:>5}/{:>5}, {:>5} runs)'.format(
				tmp_status,conv_cnt,nrun,itcnt),end='',flush=True)
			ib_res = ibalg(**{'pxy':pxy,'penalty_coeff':tmp_penalty,**algargs})
			run_cnt +=1 
			conv_cnt += int(ib_res['valid'])
			if (run_cnt - conv_cnt) >early_stop_limit:
				tmp_result = []
				conv_cnt = 0
				run_cnt = 0
				pen_idx_cnt+=1
				if pen_idx_cnt<len(penalty_range):
					tmp_penalty = penalty_range[pen_idx_cnt]
					algargs['penalty_coeff'] = tmp_penalty
					tmp_status = status_template.format(**algargs)
				else:
					break
			if ib_res['valid']:
				tmp_result.append(ib_res)
			if run_cnt == nrun:
				# successfully reach the targeted iterations
				break
		if conv_cnt/nrun >= rate_thres:
			print('\r{:<200}'.format(tmp_status+'complete, min_pen:{:>5.4f}'.format(tmp_penalty)),end='\r',flush=True)	
			result_all.append({'beta':beta,'result':tmp_result,'min_penc':tmp_penalty}) 
		else:
			print('\r{:<200}'.format(tmp_status+'failed'),end='\r',flush=True)
	print(' '*os.get_terminal_size().columns+'\r',end='',flush=True)
	print('time elapsed: {:>16.8f} seconds'.format(time.time()-time_start))
	return result_all
# main loop

d_exp_dir = os.path.join(d_base,args.output)# this is the main experiment folder
# avoid duplicate 



# prepare the directory
# the folder must not exist
#tmp_save_dir = os.path.join(d_exp_dir,tmp_dir_name)
# create the corresponding experiment result folder
#os.makedirs(tmp_save_dir,exist_ok=True)
tmp_status   = ut.genStatus(**argdict)
tmp_sys_dict = {'status_tex':tmp_status, **_sys_parms}
result_all = runSim(d_alg,d_beta_range,d_penalty_range,d_pxy_info['pxy'],args.ntime,**tmp_sys_dict)
print('beta,min_penc')
for item in result_all:
	print('{},{}'.format(item['beta'],item['min_penc']))

try:
	os.makedirs(d_exp_dir,exist_ok=False)
except:
	sys.exit("The folder already exists, please change the output name:{}".format(d_exp_dir))
# saving the results to experiment folder
d_file_name = ut.genOutName(**argdict) + '.pkl'
print('saving the result to:{}/{}\n'.format(tmp_exp_dir,d_file_name))
with open(os.path.join(tmp_exp_dir,d_file_name),'wb') as fid:
	pickle.dump(result_all,fid)
with open(os.path.join(tmp_exp_dir,'arguments.pkl'),'wb') as fid:
	pickle.dump(argdict,fid)
with open(os.path.join(tmp_exp_dir,'sysParams.pkl'),'wb') as fid:
	pickle.dump(_sys_parms,fid)
