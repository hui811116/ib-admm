import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#result_file = 'ibgd_exp_first_attempt.pkl'
result_file = 'ibgd_exp_class_3.pkl'

with open(result_file,'rb') as fid:
	result_ib = pickle.load(fid)

'''
for k,v in result_ib.items():
	print(k)
print('------------inside')
for k,v in result_ib['res_gd'].items():
	print(k)
sys.exit()
'''

beta_range = result_ib['beta']
methods = result_ib['methods']
pxy = result_ib['prob_xy']
avg_num = result_ib['avg_num']

fullname= {
	'orig':'IB-orig',
	'gd':'IB-grad',
	'alm':'IB-ADMM',
	'alm_2w':'ALM-Experiment',
	'alm_exp':'ADMM exp-grad',
	'alm_dbg':'IB-ADMM',
	'alm_breg':'IB-ADMM-Breg',
}

# plotting beta versus mi
fig, ax = plt.subplots()
plt.grid('on')
colors = ['blue','red','green','black','aqua','purple']
msty = ['x','^','.','*','+','d']
msize = 8.0
tit_fs = 24
lab_fs = 20
leg_fs = 18
tik_fs = [16,14]
betavsmi_data = np.zeros((len(methods),len(beta_range)*avg_num,3))
for im,md in enumerate(methods):
	mn = fullname[md]
	md_mixz = result_ib['res_'+md]['IXZ']
	md_miyz = result_ib['res_'+md]['IYZ']
	for ib,beta in enumerate(beta_range):
		# convert to bits
		# NOTE: should exclude invalid points
		betavsmi_data[im,ib*avg_num:(ib+1)*avg_num,1] = np.log2(np.exp(1))* md_mixz[ib]
		betavsmi_data[im,ib*avg_num:(ib+1)*avg_num,2] = np.log2(np.exp(1))* md_miyz[ib]
		betavsmi_data[im,ib*avg_num:(ib+1)*avg_num,0] = beta
	plt.scatter(betavsmi_data[im,:,1],betavsmi_data[im,:,2],color=colors[im],marker=msty[im],label=mn)
	#plt.scatter(betavsmi_data[im,:,1],betavsmi_data[im,:,2],color=colors[2*im+1],marker=msty[2*im+1])
plt.legend(fontsize=leg_fs)
plt.title(r'Information Plane'.format(pxy.shape[1]),fontsize=tit_fs)
plt.xlabel(r'$I(X;Z)$ (bits)',fontsize=lab_fs)
plt.ylabel(r'$I(Y;Z)$ (bits)',fontsize=lab_fs)
ax.tick_params(axis='both',which='major',labelsize=tik_fs[0])
ax.tick_params(axis='both',which='minor',labelsize=tik_fs[1])
plt.show()

# show the MI over beta

mi_plot_sets = []
for im, md in enumerate(methods):
	md_mixz = result_ib['res_'+md]['IXZ']
	md_miyz = result_ib['res_'+md]['IYZ']
	tmp_mixz = np.zeros((len(beta_range)*len(md_mixz[0]),))
	tmp_miyz = np.zeros((len(beta_range)*len(md_miyz[0]),))
	copy_beta = np.zeros((len(beta_range)*len(md_mixz[0]),))
	for ib,beta in enumerate(beta_range):
		tmp_mixz[ib*len(md_mixz[ib]):(ib+1)*len(md_mixz[ib])] = np.log2(np.exp(1)) * md_mixz[ib]
		tmp_miyz[ib*len(md_miyz[ib]):(ib+1)*len(md_miyz[ib])] = np.log2(np.exp(1)) * md_miyz[ib]
		copy_beta[ib*len(md_miyz[ib]):(ib+1)*len(md_miyz[ib])] = beta* np.ones((len(md_miyz[ib]),))
	mi_plot_sets.append({
			'IXZ':tmp_mixz,
			'IYZ':tmp_miyz,
			'BETA':copy_beta,
			'method':fullname[md],
		})
fig, ax = plt.subplots()
ax.grid('on')
for it,item in enumerate(mi_plot_sets):
	ax.scatter(item['BETA'],item['IXZ'],color=colors[2*it],marker=msty[2*it],label=item['method'])
ax.tick_params(axis='both',which='major',labelsize=tik_fs[0])
ax.tick_params(axis='both',which='minor',labelsize=tik_fs[1])
ax.set_title(r'$I(X;Z)$ versus $\beta$',fontsize=tit_fs)
ax.set_ylabel(r'$I(X;Z)$ bits',fontsize=lab_fs)
ax.set_xlabel(r'$\beta$',fontsize=lab_fs)
#ax.set_xlim([0,6])
ax.legend(fontsize=leg_fs,loc=2)
plt.show()

fig, ax = plt.subplots()
ax.grid('on')
for it,item in enumerate(mi_plot_sets):
	ax.scatter(item['BETA'],item['IYZ'],color=colors[2*it+1],marker=msty[2*it+1],label=item['method'])
ax.tick_params(axis='both',which='major',labelsize=tik_fs[0])
ax.tick_params(axis='both',which='minor',labelsize=tik_fs[1])
ax.set_title(r'$I(Y;Z)$ versus $\beta$',fontsize=tit_fs)
ax.set_ylabel(r'$I(Y;Z)$ bits',fontsize=lab_fs)
ax.set_xlabel(r'$\beta$',fontsize=lab_fs)
#ax.set_xlim([0,6])
ax.legend(fontsize=leg_fs,loc=2)
plt.show()


#####
# PLOT: SHOW the probability transition for pz, pzcx