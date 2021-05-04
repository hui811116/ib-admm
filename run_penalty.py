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
#parser.add_argument('output',type=str,help='specify the MATLAB .mat filename')
parser.add_argument('-dataset',type=str,choices=datasetlist,default='synWu',help='select the dataset')
#parser.add_argument("-minbeta",type=float,help='the minimum beta to sweep',default=1.0)
#parser.add_argument("-maxbeta",type=float,help='the maximum beta to sweep',default=10.0)
#parser.add_argument('-numbeta',type=int,help='the geometric spacing between beta_min and beta_max',default=16)
parser.add_argument("-beta",type=float,help="the trade-off parameter beta to run",default=8.0)
parser.add_argument('-ntime',type=int,help='run how many times per beta',default=100)
parser.add_argument('-minpenalty',type=float,help='min penalty coefficient',default=4.0)
parser.add_argument('-maxpenalty',type=float,help='max penalty coefficient',default=128.0)
parser.add_argument('-steppenalty',type=float,help="the penalty range between penalty_min and penalty_max",default=4.0)

parser.add_argument('-omega',type=float,help='Bregman Regularization coefficient',default=2.0)
parser.add_argument('-thres',type=float,help='convergence threshold',default=1e-6)
parser.add_argument('-sinit',type=float,help='Initial step size for line search',default=0.5)
parser.add_argument('-sscale',type=float,help='Step size line search scaling',default=0.25)
parser.add_argument('-seed',type=int,help='Random seed for reproduction',default=None)
parser.add_argument('-maxiter',type=int,help='The maximum number of iteration per run',default=50000)
parser.add_argument('-mat',type=str,help="Save the results to MATLAB .mat file as specified",default="")

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
}

if args.verbose:
	pprint.pprint(_sys_parms)


#d_beta_range = np.geomspace(args.minbeta,args.maxbeta,num=args.numbeta)
d_beta = args.beta
d_penalty_range = np.arange(args.minpenalty,args.maxpenalty,args.steppenalty)
d_pxy_info = dt.datasetSelect(args.dataset)
d_alg = alg.selectAlg(args.method)

def runSimSingleBeta(ibalg,beta,pxy,nrun,**kwargs):
	status_template = kwargs['status_tex']
	#result_all = []
	time_start = time.time()
	# now only have one beta
	#for bidx,beta in enumerate(betarange):
	algargs = {'beta':beta,'qlevel':pxy.shape[1],**kwargs,}
	ut.checkAlgArgs(**algargs)
	tmp_status = status_template.format(**algargs)
	#tmp_result = []
	time_per_run = time.time()
	conv_cnt = 0
	result_all = np.zeros((nrun,2))
	for it in range(nrun):
		print('\rCurrent status: {}{:>6.2f}% ({:>5}/{:>5} iterations)'.format(
			tmp_status,100*it/nrun,it,nrun),end='',flush=True)
		tmp_time_start = time.time()
		ib_res = ibalg(**{'pxy':pxy,**algargs})
		result_all[it,:] = [ib_res['niter'],time.time()-tmp_time_start]
		conv_cnt += int(ib_res['valid'])
		#tmp_result.append(ib_res)
	time_per_run = (time.time()-time_per_run)/nrun
	avg_conv_rate= conv_cnt / nrun
	print('\r{:<200}'.format(tmp_status+'complete'),end='\r',flush=True)
	#result_all.append({'beta':beta,'result':tmp_result,'avg_time':time_per_run,'avg_conv':avg_conv_rate}) 
	print(' '*200+'\r',end='',flush=True)
	print('time elapsed: {:>16.8f} seconds'.format(time.time()-time_start))
	return {'beta':beta,
			'avg_time':np.mean(result_all[:,1]),'var_time':np.std(result_all[:,1]),
			'avg_conv':avg_conv_rate,
			'avg_iter':np.mean(result_all[:,2]),'var_iter':np.std(result_all[:,2])}
# main loop

#d_exp_dir = os.path.join(d_base,args.output)# this is the main experiment folder
# avoid duplicate 
#try:
#	os.makedirs(d_exp_dir,exist_ok=False)
#except:
#	sys.exit("The folder already exists, please change the output name:{}".format(d_exp_dir))

hdr_tex = 'penalty,avg_conv,avg_time,std_time,avg_iter,std_iter'
print(hdr_tex)
pt_template = '{penalty:5.2f},{avg_conv:10.6f},{avg_time:10.6f},{std_time:10.6f},{avg_iter:10.6f},{std_iter:10.6f}'
pen_rec = np.zeros((len(d_penalty_range),6))

for pidx, pen in enumerate(d_penalty_range):
	# prepare the directory
	#tmp_dir_name = ut.genExpName(**{'penalty':pen,**argdict})
	tmp_status   = ut.genStatus(**{'penalty':pen,**argdict})
	# the folder must not exist
	#tmp_save_dir = os.path.join(d_exp_dir,tmp_dir_name)
	# create the corresponding experiment result folder
	#os.makedirs(tmp_save_dir,exist_ok=True)
	tmp_sys_dict = {'penalty_coeff':pen,'status_tex':tmp_status, **_sys_parms}
	result_all = runSimSingleBeta(d_alg,d_beta,d_pxy_info['pxy'],args.ntime,**tmp_sys_dict)
	pen_rec[pidx,:] = [pen,result_all['avg_conv'],result_all['avg_time']]
	print(pt_template.format(**{penalty:pen,**result_all}))
	# saving the results to experiment folder
	#d_file_name = ut.genOutName(**argdict) + '.pkl'
	#print('saving the result to:{}/{}\n'.format(tmp_save_dir,d_file_name))
	#with open(os.path.join(tmp_save_dir,d_file_name),'wb') as fid:
	#	pickle.dump(result_all,fid)
	#with open(os.path.join(tmp_save_dir,'arguments.pkl'),'wb') as fid:
	#	pickle.dump(argdict,fid)
	#with open(os.path.join(tmp_save_dir,'sysParams.pkl'),'wb') as fid:
	#	pickle.dump({'penalty_coeff':pen, **_sys_parms},fid)

if not args.mat:
	# a file name is specified
	nametex = 'method,{:},omega,{:},beta,{:.2f}'.format(args.method,args.omega,d_beta)
	savemat(args.mat+'.mat',{'label':hdr_tex,nametex:pen_rec})
	print('saving MATLAB .mat file: {:}'.format(args.mat))