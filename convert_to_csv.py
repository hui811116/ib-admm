import pickle
import numpy as np
import os
import sys
import argparse

d_base = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("filedir",type=str,help='specify the directory to be converted')
parser.add_argument('-output',type=str,help='specify the output file name',default='converted')

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

# header to parse
csv_hdr = ['IXZ','IYZ','niter','valid']

collect_csv = []
for bidx, item_beta in enumerate(results):
	beta = item_beta['beta']
	beta_result = item_beta['result']
	for nidx , nresult in enumerate(beta_result):
		tmp_csv = [str(beta)]
		for ele in csv_hdr:
			tmp_csv.append(str(float(nresult[ele])))
		collect_csv.append(','.join(tmp_csv))

# writing csv file
output_path = os.path.join(filedir,args.output+'.csv')
with open(output_path,'w') as fid:
	fid.write('beta,'+','.join(csv_hdr)+'\n')
	for line in collect_csv:
		fid.write(line+'\n')
print('Convertion complete, write to {}'.format(output_path))
