import numpy as np
import sys

# macro definitions
_NAMEOFDATASET=['synWu']


def datasetSelect(name,**kwargs):
	if name in _NAMEOFDATASET:
		return synICLR2020_WU()
	else:
		return None
def synICLR2020_WU():
	gbl_pycx = np.array([[0.7,0.3,0.075],[0.15,0.50,0.025],[0.15,0.20,0.90]])
	gbl_px = np.ones(3,)/3
	gbl_pxy = gbl_pycx*gbl_px[:,None].T
	return {'pxy':gbl_pxy,'nx':3,'ny':3}

