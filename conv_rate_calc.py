import numpy as np
import os
import sys
import pickle

args = sys.argv

d_base = os.getcwd()
conv_result =[]
print('method,beta,conv_rate')
for item in os.listdir(os.path.join(d_base,args[1])):
	inner_path = os.path.join(d_base,args[1],item)
	for inner_file in os.listdir(inner_path):
		if inner_file == 'arguments.pkl':
			with open(os.path.join(inner_path,'arguments.pkl'),'rb') as fid:
				tmp_args = pickle.load(fid)
		elif inner_file == 'sysParams.pkl':
			with open(os.path.join(inner_path,'sysParams.pkl'),'rb') as fid:
				tmp_sysparams = pickle.load(fid)
		elif '.pkl' in inner_file:
			with open(os.path.join(inner_path,inner_file),'rb') as fid:
				tmp_data = pickle.load(fid)
		else:
			# could be "converted.csv"
			pass
	# ideally, the three ingredients are prepared for processing...
	tmp_beta_range = np.geomspace(tmp_args['minbeta'],tmp_args['maxbeta'],num=tmp_args['numbeta'])
	navg = tmp_args['ntime']
	method = tmp_args['method']
	if method == 'orig':
		method_text = 'IB-orig'
	elif method == 'gd':
		method_text = 'IB-grad'
	elif method == 'alm':
		method_text = 'ALM;c={};$\\omega$={}'.format(int(tmp_args['penalty']),int(tmp_args['omega']))
	tmp_conv = {'method':method_text,'beta':tmp_beta_range,'conv':np.zeros((len(tmp_beta_range),))}
	
	for bidx, item_beta in enumerate(tmp_data):
		beta = item_beta['beta']
		beta_result = item_beta['result']
		tmp_valid_cnt = 0
		for nidx , nresult in enumerate(beta_result):
			tmp_valid_cnt += int(nresult['valid'])
		print('{},{:.4f},{:.5f}'.format(method_text,beta,tmp_valid_cnt/navg))
		tmp_conv['conv'][bidx] = tmp_valid_cnt/navg
