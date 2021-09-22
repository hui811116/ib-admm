import pickle
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pprint
import mydata as dt
import myutils as ut
from scipy.io import savemat

d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("filedir",type=str,help='specify the directory to be converted')
parser.add_argument('-save',help='printing the log and parameters along the execution',action='count',default=0)
parser.add_argument('-conv',help='Exporting figures for presentation',action='count',default=0)
parser.add_argument('-mi',help='Plotting the Mutual Information and Information Plane',action='count',default=0)
parser.add_argument('-omega',help='Display the surface of omega versus penalty of ALM',action='count',default=0)
#parser.add_argument('-mat',help='Save a matlab-compatible .mat file',action='count',default=0)
args = parser.parse_args()

inputdir = os.path.join(d_base,args.filedir)

mk_sets = ["^","s","+",".","x","o"]
ls_sets = [":","--","-.","-"]
color_sets = ["g","m","k","r","b","c",'y']

fs_s = {
	'label_fs': 16,
	'title_fs': 18,
	'tick_fs' : 14,
	'leg_fs'  : 14,
}

penalty_methods = ['dev','sec','bayat']

def extractData(results):
	res_hdr = ['IXZ','IYZ','niter','valid']
	collect_all = []
	nconv_rate = []
	for bidx, item_beta in enumerate(results):
		beta = item_beta['beta']
		beta_result = item_beta['result']
		if item_beta.get('avg_conv',False):
			print('beta,{:5.3f}, avg_conv,{:5.3f}'.format(beta,item_beta['avg_conv']))
		ncnt = len(beta_result)
		nvalid = 0
		for nidx, nresult in enumerate(beta_result):
			nvalid += int(nresult['valid'])
			tmp_row = [beta]
			for ele in res_hdr:
				tmp_row.append(float(nresult[ele]))
			collect_all.append(tmp_row)
		#print('beta={:6.2f}; convergence rate--{:10.4f}'.format(beta,nvalid/ncnt))
		#nconv_rate.append([beta,nvalid/ncnt])
		nconv_rate.append([beta,item_beta['avg_conv'],item_beta['avg_time']])
	npresult = np.array(collect_all)
	nconv_res =  np.array(nconv_rate)
	return (npresult, nconv_res)

def readFolder(filedir):
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
		print('found no arguments')
		return
	if not arguments.get('penalty',False):
		# attempt to collect from file name
		flist = os.path.split(filedir)
		tmp_fname =flist[-1] 
		if "_c" in tmp_fname:
			cstr_idx = tmp_fname.find("c")
			penalty = float(tmp_fname[cstr_idx+1:tmp_fname.find("_",cstr_idx+1)])
		else:
			sys.exit('Fatal error, no penalty argument found')
	else:
		penalty = arguments['penalty']

	beta_range = np.geomspace(arguments['minbeta'],arguments['maxbeta'],num=arguments['numbeta'])
	d_pxy_info = dt.datasetSelect(arguments['dataset'])
	# compute the H(Y) and I(X;Y)
	d_pxy = d_pxy_info['pxy']
	py = np.sum(d_pxy,axis=0)
	px = np.sum(d_pxy,axis=1)
	enthy = np.sum(-py*np.log(py)) # nats, convert later
	mixy = np.sum(d_pxy*np.log(d_pxy/py[None,:]/px[:,None]))
	tmpdict = {
		'data':results, 'pxy':d_pxy,'mixy':mixy,'beta_range':beta_range,
		'method':arguments['method'],'ntime':arguments['ntime'],
		'penalty':penalty,'omega':arguments['omega'],
		'thres':arguments['thres'],'dataset':arguments['dataset'],'output':arguments['output'],
	}
	return tmpdict

## for plotting mi
def miResult(filedir):
	readout = readFolder(filedir)
	res_hdr = ['IXZ','IYZ','niter','valid']
	(npresult,nconv_res) = extractData(readout['data'])
	method = readout['method']
	penalty = readout['penalty']
	if method == 'dev':
		labeltex = r"alm, $c={:},\omega={:}$".format(int(penalty),int(readout['omega']))
	elif method=='bayat':
		labeltex = r"bayat, $c={:}$".format(int(penalty))
	elif method=='gd':
		labeltex = r"IB-gd"
	elif method == 'orig':
		labeltex = r"IB-BA"
	else:
		sys.exit("method {:} undefined".format(method))
	
	outdict = {'data':npresult,'hdr':['beta']+res_hdr,'method':method,'label':labeltex,'mixy':mixy}
	return outdict

## for plotting convergence specifically
def convResult(filedir):
	readout = readFolder(filedir)
	print('method,{},penalty,{:5.3f},omega,{:5.3f}'.format(readout['method'],readout['penalty'],readout['omega']))
	results = readout['data']
	res_hdr = ['IXZ','IYZ','niter','valid']
	(npresult,nconv_res) = extractData(results)
	# FIXME: auto saving a .mat file for further processing
	#pathlist = os.path.split(filedir)
	#print('DEBUG(convResult),sizeof concate data:')
	#matdict_npres = {'label':'beta,'+','.join(res_hdr),pathlist[-1]+'_np':npresult}
	#matdict_conv  = {'label':'beta,avg_prob,avg_time',pathlist[-1]+'_pt':nconv_res}
	#savemat
	#savemat(os.path.join(filedir,pathlist[-1]+'_np.mat'),matdict_npres)
	#savemat(os.path.join(filedir,pathlist[-1]+'_pt.mat'),matdict_conv)
	saveMat(filedir,(npresult,nconv_res))
	method = readout['method']
	penalty = readout['penalty']
	out_dict = {'beta':nconv_res[:,0],'percent':nconv_res,'method':method,'penalty':penalty}
	if method == 'dev':
		out_dict['omega'] = readout['omega']
	elif method in penalty_methods:
		pass
	else:
		sys.exit("fatal error, the method does not belong to penalty methods")
	return out_dict

def saveMat(filedir,extracted):
	res_hdr = ['IXZ','IYZ','niter','valid']
	(npresult,nconv_res) = extracted
	pathlist = os.path.split(filedir)
	matdict_npres = {'label':'beta,'+','.join(res_hdr),pathlist[-1].replace('.','f')+'_np':npresult}
	matdict_conv  = {'label':'beta,avg_prob,avg_time',pathlist[-1].replace('.','f')+'_pt':nconv_res}
	#savemat
	savemat(os.path.join(filedir,pathlist[-1]+'_np.mat'),matdict_npres)
	savemat(os.path.join(filedir,pathlist[-1]+'_pt.mat'),matdict_conv)
	print('Saving .mat MATLAB file to:{}\n'.format(filedir))
	return
##
def readResult(filedir=".",savedir=".",collect=False):
	readout = readFolder(filedir)
	mixy = readout['mixy']
	results = readout['data']
	print('Estimated I(X:Y):{:.6f}'.format(readout['mixy']))

	res_hdr = ['IXZ','IYZ','niter','valid']
	
	(npresult,nconv_res) = extractData(results)
	# save a MATLAB .mat file for further processing
	saveMat(filedir,(npresult,nconv_res))
	fig_label = ut.getFigLabel(**readout)
	# plotting (scatter)
	# 1. IB curve: I(Y;Z) versus I(X;Z)
	# 2. MI:       I(Y;Z), I(X;Z) versus beta
	# 3. niter:    niter versus beta
	# 4. convergence ratio: over ntrials, not the rate at which error decreases

	sel_conv = npresult[:,4]!=0 # selecting those converged
	fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
	fig.set_size_inches(16,6)
	title_tex = r'Method:{},Name:{},Data:{},Conv=${:.3e}$'.format(
		readout['method'],readout['output'],readout['dataset'],readout['thres'])
	fig.suptitle(title_tex)
	# SUBFIGURE 1
	ax1.grid()
	# exclude the points that diverge
	ax1.scatter(npresult[sel_conv,1],npresult[sel_conv,2],label=fig_label)
	ax1.set_xlabel(r'$I(X;Z)$')
	ax1.set_ylabel(r'$I(Y;Z)$')

	ax1.axhline(y=mixy,xmin=0,xmax=np.amax(npresult[:,1])+1.0,
					linewidth=2,color='b',linestyle="--",label=r"$I(Z;Y)=I(X;Y)$")
	ax1.axline((0,0),(mixy,mixy),
			linewidth=2,color='m',linestyle=":",label=r"$I(Z;Y)\leq I(Z;X)$")
	ax1.legend()

	# SUBFIGURE 2
	ax2.grid()
	ax2.set_xlabel(r'$\beta$')
	ax2.set_ylabel(r'$I(Y;Z), I(X;Z)$')
	ax2.scatter(npresult[sel_conv,0],npresult[sel_conv,1],label=r"$I(X;Z)$")
	ax2.scatter(npresult[sel_conv,0],npresult[sel_conv,2],label=r"$I(Y;Z)$")
	ax2.legend()
	# SUBFIGURE3
	ax3.grid()
	ax3.set_xlabel(r'$\beta$')
	ax3.set_ylabel('Number of Iterations')
	ax3.scatter(npresult[:,0],npresult[:,3],label=r"# iterations")
	ax3.set_yscale('log')

	# SUBFIGURE4
	ax4.grid()
	ax4.set_xlabel(r'$\beta$')
	ax4.set_ylabel(r'convergence (\%)')
	ax4.set_ylim(0.0,100.0)
	ax4.plot(nconv_res[:,0],100*nconv_res[:,1],'-k*',label=r"conv.")

	plt.tight_layout()

	savename = os.path.join(filedir,"infoplane.eps")
	plt.savefig(savename,dpi=300)
	print("saving the figure to:{:}".format(savename))

	# collectively
	# mind the file_name...
	# use fir_dir .eps
	if collect:
		pathlist = os.path.split(filedir)
		collectpath = os.path.join(savedir,"collect_figs")
		os.makedirs(collectpath,exist_ok=True)
		plt.savefig(os.path.join(collectpath,pathlist[-1]+'.eps'),dpi=300)
		print("save collected figures to:{}".format(collectpath))
		#print("trying to save to {}".format(os.path.join(savedir,pathlist[-1])))

	plt.close()

if args.conv:
	# experiment mode
	# the directory has various method in single parameter 
	# currently support convergence rate comparison
	conv_collect = {}
	# structure:
	#		method
	#		beta
	#		percentage of convergent trials

	for files in os.listdir(inputdir):
		if files == 'collect_figs':
			continue
		fullname = os.path.join(inputdir,files)
		if os.path.isdir(fullname):
			one_conv = convResult(fullname)
			method = one_conv['method']
			betas  = one_conv['beta']
			percent= one_conv['percent']
			penalty= one_conv['penalty']
			#print(one_conv)
			#{'beta': array([5.5, 6. ]), 'percent': array([[5.5, 0. ],[6. , 0. ]]), 'method': 'bayat', 'penalty': 80.0}
			if not conv_collect.get(method,False):
				conv_collect[method] = {}
			if method == 'dev':
				# handle omega
				omega = str(int(one_conv['omega']))
				if not conv_collect['dev'].get(omega,False):
					conv_collect['dev'][omega] = {}
			for tmp_idx in range(len(betas)):
				if method == 'dev':
					ref_dict = conv_collect[method][omega]
				else:
					ref_dict = conv_collect[method]
				str_beta = '{:.2f}'.format(betas[tmp_idx])
				if not ref_dict.get(str_beta,False):
					ref_dict[str_beta] = []
				ref_dict[str_beta].append([penalty, percent[tmp_idx][1]])
			ref_dict['beta'] = betas
	mk_idx = 0
	ls_idx = 0
	color_idx = 0
	fig = plt.figure()
	plt.grid()
	
	for nk,nv in conv_collect.items():
		#method, #dict
		if nk == 'dev':
			# each omega
			for ik,iv in nv.items():
				label_tex = r'alm,$\omega={:}$'.format(float(ik))
				# each beta in text, and beta values
				for iik,iiv in iv.items():
					if iik != 'beta':
						tmp_conv = np.sort(np.array(iiv),axis=0)
						plt.plot(tmp_conv[:,0],100.0*tmp_conv[:,1],label=label_tex+r",$\beta={:.2f}$".format(float(iik)),
							linewidth=2.0,
							marker=mk_sets[mk_idx],
							linestyle=ls_sets[ls_idx],
							color=color_sets[color_idx])
						mk_idx +=1
						ls_idx+=1
						color_idx+=1
			
		else:
			for ik,iv in nv.items():
				if ik != 'beta':
					label_tex = r'{},$\beta={:.2f}$'.format(nk,float(ik)) 
					tmp_conv = np.sort(np.array(iv),axis=0)
					plt.plot(tmp_conv[:,0],100.0*tmp_conv[:,1],label=label_tex,linewidth=2.0,
							marker=mk_sets[mk_idx],
							linestyle=ls_sets[ls_idx],
							color=color_sets[color_idx])
					mk_idx +=1
					ls_idx+=1
					color_idx+=1
	plt.legend(fontsize=14)
	plt.title(r"$|X|=|Y|=|Z|=3,n={:},conv={:.2e}$".format(100,1e-6),fontsize=18)
	plt.xlabel(r"penalty coeff., $c$",fontsize=16)
	plt.ylabel(r"Convergent Trials (%)",fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.tight_layout()
	#plt.show()
	
	savefile = os.path.join( inputdir , args.filedir+"_conv.eps")
	plt.savefig(savefile,dpi=300)
	print("saving convergence figure to: {:}".format(savefile))
	plt.close()
		
elif args.mi:
	results_list = []
	for files in os.listdir(inputdir):
		fullname = os.path.join(inputdir,files)
		if os.path.isdir(fullname):
			results_list.append(miResult(fullname))
	results_list.sort(key=lambda x: x['method'])
	mk_idx = 0
	ls_idx = 0
	color_idx = 0
	fig = plt.figure()
	plt.grid()
	xmax = 0
	for data in results_list:
		sel_idx = data['data'][:,4] !=0
		plt.scatter(data['data'][sel_idx,1]*np.log2(np.exp(1)),data['data'][sel_idx,2]*np.log2(np.exp(1)),
			label=data['label'],color=color_sets[color_idx],marker=mk_sets[mk_idx])
		mk_idx += 1
		mk_idx = mk_idx % len(mk_sets)
		color_idx += 1
		color_idx = color_idx % len(color_sets)
		mixy = data['mixy']
		tmpxmax = np.amax(data['data'][:,1])
		xmax = np.maximum(tmpxmax,xmax)
	plt.axhline(y=mixy*np.log2(np.exp(1)),xmin=0,xmax=np.amax(xmax+1.0)*np.log2(np.exp(1)),
					linewidth=2,linestyle="--",label=r"$I(Z;Y)=I(X;Y)$")
	plt.axline((0,0),(mixy,mixy),
			linewidth=2,linestyle=":",label=r"$I(Z;Y)\leq I(Z;X)$")
	plt.xlabel(r"$I(Z;X)$ (bits)",fontsize=14)
	plt.ylabel(r"$I(Z;Y)$ (bits)",fontsize=14)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.title(r"Information Plane,$|Z|=3$,conv=${:}$".format(1e-5),fontsize=18)
	plt.legend(fontsize=12)
	plt.show()
	plt.close()
	

elif args.omega:
	# this will only be one method,
	# the goal is to extract each avg conv. prob. / avg time
	# Note: there are previous version data that doesn't store penalty_coeff in system_param
	all_result = []
	for files in os.listdir(inputdir):
		fullname = os.path.join(inputdir,files)
		if os.path.isdir(fullname):
			oneresult = convResult(fullname)
			all_result.append([oneresult['penalty'],np.array(oneresult['percent'])])
	all_result.sort(key=lambda x:x[0])
	beta_range = np.array(oneresult['beta'])
	penalty_range =[]
	for item in all_result:
		penalty_range.append(item[0])
	penalty_range = np.array(penalty_range)
	cputime_conv = np.zeros((len(beta_range),len(penalty_range)))
	prob_conv = np.zeros((len(beta_range),len(penalty_range)))
	for bi in range(len(beta_range)):
		for pi in range(len(penalty_range)):
			prob_conv[bi,pi] = all_result[pi][1][bi,1]
			cputime_conv[bi,pi] = all_result[pi][1][bi,2] # FIXME: easier but confusing
	# write the avg_prob and avg_time to .mat format
	#print('DEBUG(omega), attempting to store files to:{}'.format(inputdir)) # this is correct
	pathlist = os.path.split(inputdir)
	bdict = {'label':'beta,axis_0',pathlist[-1]+'_beta':beta_range}
	pdict = {'label':'penalty,axis_1',pathlist[-1]+'_penalty':penalty_range}
	datadict = {'label':'(convprob,cputime),beta,penalty',pathlist[-1]+'_ct':
					np.concatenate((prob_conv[np.newaxis,...],cputime_conv[np.newaxis,...]),axis=0)}
	savemat(os.path.join(inputdir,pathlist[-1]+'_beta.mat'),bdict)
	savemat(os.path.join(inputdir,pathlist[-1]+'_penalty.mat'),pdict)
	savemat(os.path.join(inputdir,pathlist[-1]+'_omega.mat'),datadict)
	print('saving MATLAB .mat to {}'.format(inputdir))
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	s_x=[]
	s_y=[]
	s_z=[]
	for bi in range(len(beta_range)):
		for pi in range(len(penalty_range)):
			s_x.append(beta_range[bi])
			s_y.append(penalty_range[pi])
			s_z.append(prob_conv[bi,pi])
	ax.scatter3D(np.array(s_x),np.array(s_y),100*np.array(s_z),c=np.array(s_z))
	#X,Y = np.meshgrid(beta_range,penalty_range)
	#surf = ax.plot_surface(X,Y,100*prob_conv.T,cmap='viridis',edgecolor='none')
	ax.set_xlabel(r"$\beta$",fontsize=fs_s['label_fs'])
	ax.set_ylabel(r"penalty coeff. $c$",fontsize=fs_s['label_fs'])
	ax.set_zlabel(r"convergent cases (%)",fontsize=fs_s['label_fs'])
	if oneresult.get('omega',False):
		titletex = r"Method:{},$\omega$={:.2f}".format(oneresult['method'],oneresult['omega'])
	else:
		titletex = r"Method:{}".format(oneresult['method'])

	ax.set_title(titletex,fontsize=fs_s['title_fs'])
	plt.tight_layout()
	plt.show()
	plt.close()


else:
	# could be single example
	readResult(inputdir,inputdir,False)
	# or experiments
	for files in os.listdir(inputdir):
		fullname = os.path.join(inputdir,files)
		if os.path.isdir(fullname):
			readResult(fullname,inputdir,args.save)



