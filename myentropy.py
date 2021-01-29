import numpy as np
def getPzFuncObj(beta,px,pen_c,b_omega,use_breg=False):
	def val_obj(pz,pzcx,mu_z):
		pen_z =  pz-pzcx@px
		return (beta-1)*np.sum(pz*np.log(pz))+np.sum(mu_z*(pen_z))+pen_c/2*(np.sum(pen_z**2)) # ignore G, not relevent
	def val_obj_breg(pz,pzcx,mu_z,pz_last):
		pen_z = pz -pzcx@px
		return (beta-1)*np.sum(pz*np.log(pz))+np.sum(mu_z*(pen_z))+pen_c/2*(np.sum(pen_z**2)) \
				+b_omega*np.sum(pz*np.log(pz/pz_last))
	if use_breg:
		return val_obj_breg
	else:
		return val_obj

def getPzGradObj(beta,px, pen_c,b_omega,use_breg=False):
	def grad_obj(pz,pzcx,mu_z):
		pen_z = pz - pzcx@px
		raw_grad = (beta-1)*(np.log(pz)+1)+mu_z+pen_c*pen_z
		return raw_grad-np.mean(raw_grad)
	def grad_obj_breg(pz,pzcx,mu_z,pz_last):
		pen_z = pz - pzcx@px
		raw_grad = (beta-1)*(np.log(pz)+1)+mu_z+pen_c*pen_z+b_omega*(np.log(pz/pz_last))
		return raw_grad-np.mean(raw_grad)
	if use_breg:
		return grad_obj_breg
	else:
		return grad_obj
'''
def getPzcxFuncObj(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pz,pzcx,mu_z):
		pen_z = pz-pzcx@px
		pzcy = pzcx@pxcy
		return np.sum(pzcx*np.log(pzcx)*px[None,:])-beta*np.sum(pzcy*np.log(pzcy)*py[None,:])\
				+np.sum(mu_z*pen_z)+pen_c*np.sum(pen_z**2)
	return val_obj
		
def getPzcxGradObj(beta,px,pxcy,pycx,pen_c):
	def grad_obj(pz,pzcx,mu_z):
		pen_z = pz-pzcx@px
		pzcy = pzcx@pxcy
		raw_grad = (np.log(pzcx)+1-beta+beta*np.transpose(pycx.T*np.log(1./pzcy.T))\
					-mu_z[:,None]-(pen_c*pen_z)[:,None] )*px[None,:]
		return raw_grad - np.mean(raw_grad,axis=0)
	return grad_obj
'''
def getPzcxFuncObj(beta,px,py,pxcy,pycx,pen_c,b_kappa,use_breg=False):
	def val_obj(pzcx,pz,mu_z):
		pen_z = pz-pzcx@px
		pzcy = pzcx@pxcy
		return np.sum(pzcx*np.log(pzcx)*px[None,:])-beta*np.sum(pzcy*np.log(pzcy)*py[None,:])\
				+np.sum(mu_z*pen_z)+pen_c*np.sum(pen_z**2)
	def val_obj_breg(pzcx,pz,mu_z,pzcx_delay):
		pen_z = pz - pzcx@px
		pzcy = pzcx@pxcy
		return np.sum(pzcx*np.log(pzcx)*px[None,:])-beta*np.sum(pzcy*np.log(pzcy)*py[None,:])\
				+np.sum(mu_z*pen_z)+pen_c*np.sum(pen_z**2)+b_kappa*np.sum( (pzcx*px[None,:])*(np.log(pzcx)-np.log(pzcx_delay)))
	if use_breg:
		return val_obj_breg
	else:
		return val_obj
		
def getPzcxGradObj(beta,px,py,pxcy,pycx,pen_c,b_kappa,use_breg=False):
	def grad_obj(pzcx,pz,mu_z):
		pen_z = pz-pzcx@px
		pzcy = pzcx@pxcy
		raw_grad = (np.log(pzcx)+1-beta+beta*np.transpose(pycx.T@np.log(1./pzcy.T))\
					-mu_z[:,None]-(pen_c*pen_z)[:,None] )*px[None,:]
		return raw_grad - np.mean(raw_grad,axis=0)
	def grad_obj_breg(pzcx,pz,mu_z,pzcx_delay):
		pen_z = pz -pzcx@px
		pzcy = pzcx @ pxcy
		raw_grad = (np.log(pzcx)+1-beta+beta*np.transpose(pycx.T@np.log(1./pzcy.T))\
					-mu_z[:,None]-(pen_c*pen_z)[:,None] + b_kappa*(np.log(pzcx)-np.log(pzcx_delay)+1) )*px[None,:]
		return raw_grad - np.mean(raw_grad,axis=0)
	if use_breg:
		return grad_obj_breg
	else:
		return grad_obj