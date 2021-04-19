import pickle
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pprint
import mydata as dt
import myutils as ut
d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("filedir",type=str,help='specify the directory to be converted')
#parser.add_argument('-output',type=str,help='specify the output file name',default='converted')
#parser.add_argument('-draw',type=str,choices=["ib","mi","mixz","miyz","niter",""],default="")
parser.add_argument('-save',help='printing the log and parameters along the execution',action='count',default=0)

args = parser.parse_args()

inputdir = os.path.join(d_base,args.filedir)

##
def readResult(filedir=".",savedir=".",collect=False):
	enddir = False
	for files in os.listdir(filedir):
		if files == 'arguments.pkl':
			with open(os.path.join(filedir,'arguments.pkl'),'rb') as fid:
				arguments = pickle.load(fid)
				enddir = True
		elif files == 'sysParams.pkl':
			with open(os.path.join(filedir,'sysParams.pkl'),'rb') as fid:
				sysParams = pickle.load(fid)
		elif '.pkl' in files:
			with open(os.path.join(filedir,files),'rb') as fid:
				results = pickle.load(fid)
	if not enddir:
		return


	beta_range = np.geomspace(arguments['minbeta'],arguments['maxbeta'],num=arguments['numbeta'])
	#print('arguments summary:')
	#pprint.pprint(arguments)
	#print('-'*50)
	#print('Results:')
	#print('-'*50)

	d_pxy_info = dt.datasetSelect(arguments['dataset'])
	# compute the H(Y) and I(X;Y)
	d_pxy = d_pxy_info['pxy']
	py = np.sum(d_pxy,axis=0)
	px = np.sum(d_pxy,axis=1)
	enthy = np.sum(-py*np.log(py)) # nats, convert later
	mixy = np.sum(d_pxy*np.log(d_pxy/py[None,:]/px[:,None]))
	print('Estimated H(Y):{:.6f}, I(X:Y):{:.6f}'.format(enthy,mixy))

	res_hdr = ['IXZ','IYZ','niter','valid']
	collect_all = []
	nconv_rate = []
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
		#print('beta={:6.2f}; convergence rate--{:10.4f}'.format(beta,nvalid/ncnt))
		nconv_rate.append([beta,nvalid/ncnt])
	npresult = np.array(collect_all)
	nconv_res =  np.array(nconv_rate)
	#print(nconv_res)

	fig_label = ut.getFigLabel(**arguments)
	# plotting (scatter)
	# 1. IB curve: I(Y;Z) versus I(X;Z)
	# 2. MI:       I(Y;Z), I(X;Z) versus beta
	# 3. niter:    niter versus beta
	fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
	fig.set_size_inches(16,6)
	title_tex = r'Method:{},Name:{},Data:{},Conv=${:.3e}$'.format(
		arguments['method'],arguments['output'],arguments['dataset'],arguments['thres'])
	fig.suptitle(title_tex)
	# SUBFIGURE 1
	ax1.grid()
	ax1.scatter(npresult[:,1],npresult[:,2],label=fig_label)
	ax1.set_xlabel(r'$I(X;Z)$')
	ax1.set_ylabel(r'$I(Y;Z)$')
	#print(npresult[:,1])
	#print(np.amax(npresult[:,1]))
	ax1.axhline(y=mixy,xmin=0,xmax=np.amax(npresult[:,1]+1.0),
					linewidth=2,color='b',linestyle="--",label=r"$I(Z;Y)=I(X;Y)$")
	ax1.axline((0,0),(mixy,mixy),
			linewidth=2,color='m',linestyle=":",label=r"$I(Z;Y)\leq I(Z;X)$")
	ax1.legend()

	# SUBFIGURE 2
	ax2.grid()
	ax2.set_xlabel(r'$\beta$')
	ax2.set_ylabel(r'$I(Y;Z), I(X;Z)$')
	ax2.scatter(npresult[:,0],npresult[:,1],label=r"$I(X;Z)$")
	ax2.scatter(npresult[:,0],npresult[:,2],label=r"$I(Y;Z)$")
	ax2.legend()
	# SUBFIGURE3
	ax3.grid()
	#twin_v = ax3.twinx()
	#twin_v.set_ylabel("Percentage of Convergence")
	ax3.set_xlabel(r'$\beta$')
	#fig3_p2 = twin_v.plot(nconv_res[0],nconv_res[1],":k*",label=r"conv. ($%$)")
	ax3.set_ylabel('Number of Iterations')
	ax3.scatter(npresult[:,0],npresult[:,3],label=r"# iterations")
	ax3.set_yscale('log')
	#ax3.legend()

	# SUBFIGURE4
	ax4.grid()
	ax4.set_xlabel(r'$\beta$')
	ax4.set_ylabel(r'convergence (\%)')
	ax4.set_ylim(0.0,100.0)
	ax4.plot(nconv_res[:,0],100*nconv_res[:,1],'-k*',label=r"conv.")
	#ax4.legend()

	plt.tight_layout()
	

	savename = os.path.join(filedir,"infoplane.eps")
	plt.savefig(savename,dpi=150)
	print("saving the figure to:{:}".format(savename))

	# collectively
	# mind the file_name...
	# use fir_dir .eps
	if collect:
		pathlist = os.path.split(filedir)
		collectpath = os.path.join(savedir,"collect_figs")
		os.makedirs(collectpath,exist_ok=True)
		plt.savefig(os.path.join(collectpath,pathlist[-1]+'.eps'),dpi=150)
		print("save collected figures to:{}".format(collectpath))
		#print("trying to save to {}".format(os.path.join(savedir,pathlist[-1])))

	plt.close()


# could be single example
readResult(inputdir,inputdir,False)

# or experiments
for files in os.listdir(inputdir):
	fullname = os.path.join(inputdir,files)
	#print(os.path.join(d_base,fullname))
	#print(os.path.isdir(os.path.join(d_base,fullname)))
	hir_dir  = os.path.join(d_base,fullname)
	if os.path.isdir(hir_dir):
		readResult(hir_dir,inputdir,args.save)





'''

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
'''

#
'''
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
print('Estimated H(Y):{:.6f}, I(X:Y):{:.6f}'.format(enthy,mixy))

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

fig_label = ut.getFigLabel(**arguments)
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
ax1.scatter(npresult[:,1],npresult[:,2],label=fig_label)
ax1.set_xlabel(r'$I(X;Z)$')
ax1.set_ylabel(r'$I(Y;Z)$')
ax1.axhline(y=mixy*np.log2(np.exp(1)),xmin=0,xmax=np.amax(npresult[:,1]*np.log2(np.exp(1))),
				linewidth=2,color='b',linestyle="--",label=r"$I(Z;Y)=I(X;Y)$")
ax1.axline((0,0),(mixy*np.log2(np.exp(1)),mixy*np.log2(np.exp(1))),
		linewidth=2,color='m',linestyle=":",label=r"$I(Z;Y)\leq I(Z;X)$")
ax1.legend()

# SUBFIGURE 2
ax2.grid()
ax2.set_xlabel(r'$\beta$')
ax2.set_ylabel(r'$I(Y;Z), I(X;Z)$')
ax2.scatter(npresult[:,0],npresult[:,1],label=r"$I(X;Z)$")
ax2.scatter(npresult[:,0],npresult[:,2],label=r"$I(Y;Z)$")
ax2.legend()
# SUBFIGURE3
ax3.grid()
ax3.set_xlabel(r'$\beta$')
ax3.set_ylabel('Number of Iterations')
ax3.scatter(npresult[:,0],npresult[:,3])
ax3.set_yscale('log')
plt.tight_layout()
if args.save:
	savename = os.path.join(filedir,"infoplane.eps")
	plt.savefig(savename,dpi=150)
	print("saving the figure to:{:}".format(savename))
plt.show()

'''


'''
# if some specific plot is needed
'''
'''
if args.draw == "ib":
	titletex = r'Information Plane, $|Z|={:}$, thres={:.2e}'.format(d_pxy.shape[1],arguments['thres'])
	fig = plt.figure()
	plt.grid()
	plt.title(titletex,fontsize=18)
	plt.xlabel(r"$I(Z;X)$ (bits)",fontsize=16)
	plt.ylabel(r"$I(Z;Y)$ (bits)",fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.scatter(npresult[:,1]*np.log2(np.exp(1)),npresult[:,2]*np.log2(np.exp(1)),label=arguments['method'])
	plt.tight_layout()

	plt.axhline(y=mixy*np.log2(np.exp(1)),xmin=0,xmax=np.amax(npresult[:,1]*np.log2(np.exp(1))),
				linewidth=2,color='b',linestyle="--",label=r"$I(Z;Y)=I(X;Y)$")
	plt.axline((0,0),(mixy*np.log2(np.exp(1)),mixy*np.log2(np.exp(1))),
		linewidth=2,color='m',linestyle=":",label=r"$I(Z;Y)\leq I(Z;X)$")
	plt.legend(fontsize=12, loc='best')
	plt.show()
#elif args.draw == "mixz":
elif args.draw == "miyz":
	titletex = r'$I(Z;Y)$ versus $\beta$, $|Z|={:}$, thres={:.2e}'.format(d_pxy.shape[1],arguments['thres'])
	fig = plt.figure()
	plt.grid()
	plt.title(titletex,fontsize=18)
	plt.xlabel(r"$\beta$",fontsize=16)
	plt.ylabel(r"$I(Z;Y)$ (bits)",fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.scatter(npresult[:,0],npresult[:,2]*np.log2(np.exp(1)),label=arguments['method'])
	plt.tight_layout()
	plt.axhline(y=mixy*np.log2(np.exp(1)),xmin=0,xmax=np.amax(beta_range),
		label=r"I(X;Y)",linewidth=2.5,linestyle=":",
		color="m")
	#plt.axhline(y=enthy*np.log2(np.exp(1)),xmin=0,xmax=np.amax(npresult[:,1]*np.log2(np.exp(1))),
	#			linewidth=2,color='b',linestyle="--",label=r"$H(Y)\geq I(Y;Z)$")
	#plt.axline((0,0),(np.amax(npresult[:,2]*np.log2(np.exp(1))),np.amax(npresult[:,2]*np.log2(np.exp(1)))),
	#	linewidth=2,color='m',linestyle=":",label=r"$I(X;Y)\geq I(Y;Z)$")
	plt.legend(fontsize=12, loc='best')
	plt.show()
#elif args.draw == "mi":
elif args.draw == "niter":
	titletex = r'Convergence Time, $|Z|={:}$, thres={:.2e}'.format(d_pxy.shape[1],arguments['thres'])
	fig = plt.figure()
	plt.grid()
	plt.title(titletex,fontsize=18)
	plt.xlabel(r"$\beta$",fontsize=16)
	plt.ylabel(r"Number of Iterations",fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.scatter(npresult[:,0],npresult[:,3]*np.log2(np.exp(1)),label=arguments['method'])
	plt.tight_layout()
	plt.yscale("log")
	#plt.axhline(y=mixy*np.log2(np.exp(1)),xmin=0,xmax=np.amax(beta_range),
	#	label=r"I(X;Y)",linewidth=2.5,linestyle=":",
	#	color="m")
	#plt.axhline(y=enthy*np.log2(np.exp(1)),xmin=0,xmax=np.amax(npresult[:,1]*np.log2(np.exp(1))),
	#			linewidth=2,color='b',linestyle="--",label=r"$H(Y)\geq I(Y;Z)$")
	#plt.axline((0,0),(np.amax(npresult[:,2]*np.log2(np.exp(1))),np.amax(npresult[:,2]*np.log2(np.exp(1)))),
	#	linewidth=2,color='m',linestyle=":",label=r"$I(X;Y)\geq I(Y;Z)$")
	plt.legend(fontsize=12, loc='best')
	plt.show()	
else:
	sys.exit()
'''