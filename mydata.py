import numpy as np
import sys

def getDatasetList():
	return ['synWu','synMy']

def datasetSelect(name,**kwargs):
	if name == 'synWu':
		return synICLR2020_WU()
	elif name == 'synMy':
		return synMy()
	else:
		return None
def synICLR2020_WU():
	gbl_pycx = np.array([[0.7,0.3,0.075],[0.15,0.50,0.025],[0.15,0.20,0.90]])
	gbl_px = np.ones(3,)/3
	gbl_pxy = (gbl_pycx*gbl_px[None,:]).T
	return {'pxy':gbl_pxy,'nx':3,'ny':3}

def synMy():
	gbl_pycx = np.array([[0.80,0.20,0.05],[0.15,0.65,0.05],[0.05,0.15,0.90]])
	gbl_px = np.ones(3,)/3
	gbl_pxy = (gbl_pycx*gbl_px[None,:]).T
	return {'pxy':gbl_pxy,'nx':3, 'ny':3}
