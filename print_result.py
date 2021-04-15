import pickle
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pprint
import mydata as dt
d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("filedir",type=str,help='specify the directory to be converted')
parser.add_argument('-output',type=str,help='specify the output file name',default='converted')
parser.add_argument('-draw',type=str,choices=["ib","mi","mixz","miyz","niter",""],default="")

args = parser.parse_args()
filedir = os.path.join(d_base,args.filedir)

for files in os.listdir(filedir):
	if files == 'arguments.pkl':
		with open(os.path.join(filedir,'arguments.pkl'),'rb') as fid:
			arguments = pickle.load(fid)
	elif files == 'sysParams.pkl':
		with open(os.path.join(filedir,'sysParams.pkl'),'rb') as fid:
			sysParams = pickle.load(fid)
	elif '.pkl' in files:
		with open(os.path.join(filedir,files),'rb') as fid:
			results = pickle.load(fid)

beta_range = np.geomspace(arguments['minbeta'],arguments['maxbeta'],num=arguments['numbeta'])
print('arguments summary:')
pprint.pprint(arguments)
print('-'*50)
print('Results:')
print('-'*50)

d_pxy_info = dt.datasetSelect(arguments['dataset'])
# compute the H(Y) and I(X;Y)
d_pxy = d_pxy_info['pxy']
py = np.sum(d_pxy,axis=0)
px = np.sum(d_pxy,axis=1)
enthy = np.sum(-py*np.log(py)) # nats, convert later
mixy = np.sum(d_pxy*np.log(d_pxy/py[None,:]/px[:,None]))


res_hdr = ['IXZ','IYZ','niter','valid']
collect_all = []
for bidx, item_beta in enumerate(results):
	beta = item_beta['beta']
	beta_result = item_beta['result']
	ncnt = len(beta_result)
	nvalid = 0
	for nidx, nresult in enumerate(beta_result):
		nvalid += int(nresult['valid'])
		tmp_row = [beta]
		for ele in res_hdr:
			tmp_row.append(float(nresult[ele]))
		collect_all.append(tmp_row)
	print('beta={:6.2f}; convergence rate--{:10.4f}'.format(beta,nvalid/ncnt))
npresult = np.array(collect_all)
# plotting (scatter)
# 1. IB curve: I(Y;Z) versus I(X;Z)
# 2. MI:       I(Y;Z), I(X;Z) versus beta
# 3. niter:    niter versus beta
fig, (ax1,ax2,ax3) = plt.subplots(1,3)
fig.set_size_inches(12,6)
title_tex = r'Method:{},Name:{},Data:{},Conv=${:.3e}$'.format(
	arguments['method'],arguments['output'],arguments['dataset'],arguments['thres'])
fig.suptitle(title_tex)
# SUBFIGURE 1
ax1.grid()
ax1.scatter(npresult[:,1],npresult[:,2])
ax1.set_xlabel(r'$I(X;Z)$')
ax1.set_ylabel(r'$I(Y;Z)$')
# SUBFIGURE 2
ax2.grid()
ax2.set_xlabel(r'$\beta$')
ax2.set_ylabel(r'$I(Y;Z), I(X;Z)$')
ax2.scatter(npresult[:,0],npresult[:,2])
# SUBFIGURE3
ax3.grid()
ax3.set_xlabel(r'$\beta$')
ax3.set_ylabel('Number of Iterations')
ax3.scatter(npresult[:,0],npresult[:,3])
ax3.set_yscale('log')
plt.tight_layout()
plt.show()



'''
# if some specific plot is needed
'''

if args.draw == "ib":
	titletex = r'Information Plane, $|Z|={:}$, thres={:.3e}'.format(d_pxy.shape[1],arguments['thres'])
	fig = plt.figure()
	plt.grid()
	plt.title(r"Inforamtion Plane, thres={:}, $|Z|=3$".format(1e-5),fontsize=18)
	plt.xlabel(r"$I(X;Z)$ (bits)",fontsize=16)
	plt.ylabel(r"$I(Y;Z)$ (bits)",fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.scatter(npresult[:,1]*np.log2(np.exp(1)),npresult[:,2]*np.log2(np.exp(1)),label="BA")
	plt.tight_layout()

	plt.axhline(y=enthy*np.log2(np.exp(1)),xmin=0,xmax=np.amax(npresult[:,1]*np.log2(np.exp(1))),
				linewidth=2,color='b',linestyle="--",label=r"$H(Y)\geq I(Y;Z)$")
	plt.axline((0,0),(np.amax(npresult[:,2]*np.log2(np.exp(1))),np.amax(npresult[:,2]*np.log2(np.exp(1)))),
		linewidth=2,color='m',linestyle=":",label=r"$I(X;Y)\geq I(Y;Z)$")
	plt.legend(fontsize=12, loc='center left')
	plt.show()
#elif args.draw == "mixz":
#elif args.draw == "miyz":
#elif args.draw == "mi":
#elif args.draw == "niter":
else:
	sys.exit()