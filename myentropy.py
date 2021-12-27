import numpy as np
def getPzFuncObj(beta,px,pen_c,b_omega,use_breg=False):
	def val_obj(pz,pzcx,mu_z):
		pen_z =  pz-pzcx@px
		return (beta-1)*np.sum(pz*np.log(pz))+np.sum(mu_z*(pen_z))+0.5*pen_c*(np.sum(np.fabs(pen_z))**2) # ignore G, not relevent
	def val_obj_breg(pz,pzcx,mu_z,pz_last):
		pen_z = pz -pzcx@px
		return (beta-1+b_omega)*np.sum(pz*np.log(pz))+np.sum(mu_z*(pen_z))+0.5*pen_c*(np.sum(np.fabs(pen_z))**2) \
				-b_omega*np.sum(pz*np.log(pz_last))
	if use_breg:
		return val_obj_breg
	else:
		return val_obj

def getPzGradObj(beta,px, pen_c,b_omega,use_breg=False):
	def grad_obj(pz,pzcx,mu_z):
		pen_z = pz - pzcx@px
		raw_grad = (beta-1)*(np.log(pz)+1)+mu_z+pen_c*pen_z
		first_lambda = np.mean(raw_grad)
		return (raw_grad-first_lambda,first_lambda)
	def grad_obj_breg(pz,pzcx,mu_z,pz_last):
		pen_z = pz - pzcx@px
		raw_grad = (beta-1)*(np.log(pz)+1)+mu_z+pen_c*pen_z+b_omega*(np.log(pz/pz_last)+1)
		first_lambda = np.mean(raw_grad)
		return (raw_grad - first_lambda, first_lambda)
	if use_breg:
		return grad_obj_breg
	else:
		return grad_obj

def getPzNewtonObj(beta,px,pen_c,b_omega,use_breg=False):
	def hess_obj(pz,pzcx,mu_z):
		# standard gradient
		pen_z = pz - pzcx@px
		raw_grad = (beta-1)*(np.log(pz)+1)+mu_z + pen_c * pen_z
		# hessian inverse is simply a precondition matrix
		hess_inv_vec = pz/(beta+pen_c*pz-1)
		hinvgrad = hess_inv_vec*raw_grad # intermediate step
		sec_lambda = np.sum(hinvgrad)/np.sum(hess_inv_vec)
		newton_update = hinvgrad - sec_lambda*hinvgrad
		return (newton_update,sec_lambda) # lambda_z for valid probability constraint
	def hess_obj_breg(pz,pzcx,mu_z,pz_last):
		# standard gradient
		pen_z = pz - pzcx@px
		raw_grad = (beta+b_omega-1)*np.log(pz)+beta-1+mu_z + pen_c * pen_z -b_omega*np.log(pz_last)
		# hessian inverse is simply a precondition matrix
		hess_inv_vec = pz/(beta+b_omega+pen_c*pz-1)
		hinvgrad = raw_grad*hess_inv_vec
		sec_lambda = np.sum(hinvgrad)/np.sum(hess_inv_vec)
		newton_update = hinvgrad-hess_inv_vec*sec_lambda
		return (newton_update,sec_lambda)  # lambda_z for valid probability constraint
	if use_breg:
		return hess_obj_breg
	else:
		return hess_obj

def getPzcxFuncObj(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pzcx,pz,mu_z):
		pen_z = pz-pzcx@px
		pzcy = pzcx@pxcy
		return np.sum(pzcx*np.log(pzcx)*px[None,:])-beta*np.sum(pzcy*np.log(pzcy)*py[None,:])\
				+np.sum(mu_z*pen_z)+0.5*pen_c*(np.sum(np.fabs(pen_z))**2)
	return val_obj
		
def getPzcxGradObj(beta,px,pxcy,pycx,pen_c):
	def grad_obj(pzcx,pz,mu_z):
		pen_z = pz-pzcx@px
		pzcy = pzcx@pxcy
		raw_grad = (np.log(pzcx)+1-beta-beta*np.transpose(pycx.T@np.log(pzcy.T))\
					-mu_z[:,None]-pen_c*pen_z[:,None] )*px[None,:]
		first_lambda = np.mean(raw_grad,axis=0)
		return (raw_grad - first_lambda , first_lambda)
	return grad_obj

# --------------------------------------------------
#                      DEV
# --------------------------------------------------
def devPzcxFuncObj(beta,px,py,pxcy,pycx,pen_c,b_o2):
	def val_obj(pzcx,pz,mu_z,pzcx_delay):
		pen_z = pz-np.sum(pzcx*px[None,:],axis=1)
		pzcy = pzcx@pxcy
		return np.sum(pzcx*np.log(pzcx)*px[None,:])-beta*np.sum(pzcy*np.log(pzcy)*py[None,:])\
				+np.sum(mu_z*pen_z)+0.5*pen_c*(np.sum(pen_z**2))+b_o2*np.sum(pzcx*np.log(pzcx/pzcx_delay)*px[None,:])
	return val_obj
		
def devPzcxGradObj(beta,px,pxcy,pycx,pen_c,b_o2):
	def grad_obj(pzcx,pz,mu_z,pzcx_delay):
		pen_z = pz-np.sum(pzcx*px[None,:],axis=1)
		pzcy = pzcx@pxcy
		raw_grad = (np.log(pzcx)+1-beta-beta*np.transpose(pycx.T@np.log(pzcy.T))\
					-mu_z[:,None]-pen_c*pen_z[:,None]+ b_o2*(np.log(pzcx)-np.log(pzcx_delay)+1)   )*px[None,:]
		first_lambda = np.mean(raw_grad,axis=0)
		return (raw_grad - first_lambda , first_lambda)
	return grad_obj

# --------------------------------------------------

def getPzcxNewtonObj(beta,px,pxcy,pycx,pen_c):
	def hess_obj(pzcx,pz,mu_z):
		pz_hat = pzcx@px
		pzcy = pzcx@pxcy
		pen_z = pz- pz_hat
		raw_grad = (np.log(pzcx)+1-beta-beta*np.transpose(pycx.T@np.log(pzcy.T))\
					-mu_z[:,None]-pen_c*pen_z[:,None] )*px[None,:]
		(nz,nx) = pzcx.shape
		tmp_hess_inv = np.zeros((nz,nx,nx))
		tmp_dir = np.zeros((nz,nx));
		for iz in range(nz):
			pzcy_vec = pzcy[iz,:]
			pzcx_vec = pzcx[iz,:]
			tmp_hess = -beta* px[:,None]*pycx.T/(pzcy_vec[None,:])@pxcy.T+pen_c*px[:,None]@px[None,:]
			tmp_hess[np.diag_indices(nx)] += (px/pzcx_vec) # FIXME: this part may explode...
			tmp_hess_inv[iz,...] = np.linalg.inv(tmp_hess) # FIXME: Could approximate the inverse with BFGS
			tmp_dir[iz,:] = tmp_hess_inv[iz,...]@raw_grad[iz,:]
		sum_hess_inv = np.linalg.inv(np.sum(tmp_hess_inv,axis=0))@np.sum(tmp_dir,axis=0) # FIXME: Could approximate the inverse with BFGS
		for iz in range(nz):
			tmp_dir[iz,:] -= tmp_hess_inv[iz,...]@sum_hess_inv
		return (tmp_dir,sum_hess_inv)
	return hess_obj


'''
Oringial IB in gradient descent

'''
def getLibGDGradObj(beta,px,py,pxcy,pycx):
	def grad_obj(pzcx,pz,pycz):
		nabla_iyz = np.transpose(np.log((1/py)[:,None]*pycz))@pycx
		grad = (np.log((1./pz)[:,None]*pzcx)-beta*nabla_iyz )*px[None,:]
		mean_grad = np.mean(grad,axis=0)
		return (grad-mean_grad,mean_grad)
	return grad_obj
def getLibGDFuncObj(beta,px,py,pxcy,pycx):
	def val_obj(pzcx,pz,pycz):
		return np.sum(pzcx*px[None,:]*np.log(pzcx/pz[:,None]))\
				-beta*np.sum(pycz*pz[None,:]*np.log(pycz/py[:,None]))
	return val_obj

'''
---------------------------------------------
Compare method: Bayat's ADMM-IB

F. Bayat and S. Wei, "Information Bottleneck Problem Revisited,
" 2019 57th Annual Allerton Conference on Communication, Control, 
and Computing (Allerton), Monticello, IL, USA, 2019, 
pp. 40-47, doi: 10.1109/ALLERTON.2019.8919752.
---------------------------------------------
'''


def getBayatGradObjPzcx(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pzcx,pzcy,pz,mu_z,mu_zy):
		ky = np.sum(pzcx@pxcy,axis=0) # dim=ny
		tmp_term =  (ky[None,:]-pzcx@pxcy)/((ky**2)[None,:])  # dim = nz * nx
		grad = (np.log(pzcx)+1)*px[None,:]-pen_c*(pz - np.sum(pzcx*px[None,:],axis=1)+mu_z/pen_c)[:,None]*px[None,:] \
						-pen_c*(tmp_term)*(pzcy-pzcx@pxcy/ky[None,:]+mu_zy/pen_c)@ pxcy.T
		mean_grad = np.mean(grad,axis=0)
		return (grad-mean_grad, mean_grad)
	return val_obj
def getBayatGradObjPzcy(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pzcy,pzcx,mu_zy):
		grad =-beta*(np.log(pzcy)+1)*py[None,:]+mu_zy+pen_c*(pzcy-pzcx@pxcy/(np.sum(pzcx@pxcy,axis=0)[None,:]))
		mean_grad = np.mean(grad,axis=0)
		return (grad-mean_grad,mean_grad)
	return val_obj
def getBayatGradObjPz(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pz,pzcx,mu_z):
		grad = (beta-1)*(np.log(pz)+1)+mu_z+pen_c*(pz-np.sum(pzcx*px[None,:],axis=1))
		mean_grad = np.mean(grad)
		return (grad-mean_grad,mean_grad)
	return val_obj

def getBayatFuncObjPz(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pz,pzcx,mu_z):
		return (beta-1)*(np.sum(pz*np.log(pz)))+pen_c/2*np.sum((pz-np.sum(pzcx*px[None,:],axis=1)+mu_z/pen_c)**2)
	return val_obj
def getBayatFuncObjPzcx(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pzcx,pz,pzcy,mu_z,mu_zy):
		return np.sum(pzcx*px[None,:]*np.log(pzcx))+pen_c/2*np.sum((pz-np.sum(pzcx*px[None,:],axis=1)+mu_z/pen_c)**2)\
				+pen_c/2*np.sum( (pzcy-pzcx@pxcy/np.sum(pzcx@pxcy,axis=0)[None,:]+mu_zy/pen_c)**2)
	return val_obj

def getBayatFuncObjPzcy(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pzcy,pzcx,mu_zy):
		return -beta*np.sum(pzcy*py[None,:]*np.log(pzcy))\
				+pen_c/2*np.sum((pzcy-pzcx@pxcy/np.sum(pzcx@pxcy,axis=0)[None,:]+mu_zy/pen_c)**2)
	return val_obj


# basic multiview formulation
def getMvFuncObjPz(beta,px,pen_c):
	def val_obj(pz,pzcx,mu_z):
		gamma = 1/beta
		errz = np.sum(pzcx*px[None,:],axis=1)-pz
		return (1-gamma)*(np.sum(pz*np.log(pz)))+np.sum(mu_z*errz)+pen_c/2*(np.linalg.norm(errz)**2)
	return val_obj
def getMvFuncObjPzcy(py,pxcy,pen_c):
	def val_obj(pzcy,pzcx,mu_zy):
		errzy = pzcx@pxcy - pzcy
		return -np.sum(pzcy*py[None,:]*np.log(pzcy))+np.sum(mu_zy*errzy)+pen_c/2*(np.linalg.norm(errzy)**2)
	return val_obj
def getMvFuncObjPzcx(beta,px,pxcy,pen_c):
	def val_obj(pzcx,pz,pzcy,mu_z,mu_zy):
		gamma = 1/beta
		errz = np.sum(pzcx*px[None,:],axis=1)-pz
		errzy = pzcx@pxcy-pzcy
		return gamma*np.sum(pzcx*px[None,:]*np.log(pzcx))+np.sum(mu_z*errz)+pen_c/2*(np.linalg.norm(errz)**2)\
														+np.sum(mu_zy*errzy)+pen_c/2*(np.linalg.norm(errzy)**2)
	return val_obj

def getMvGradObjPz(beta,px,pen_c):
	def grad_obj(pz,pzcx,mu_z):
		gamma = 1/beta
		errz = np.sum(pzcx*px[None,:],axis=1)-pz
		grad= (1-gamma)*(np.log(pz)+1)-mu_z-pen_c*errz
		mean_grad= np.mean(grad)
		return (grad-mean_grad,mean_grad)
	return grad_obj
def getMvGradObjPzcy(py,pxcy,pen_c):
	def grad_obj(pzcy,pzcx,mu_zy):
		errzy = pzcx@pxcy - pzcy
		grad= -(np.log(pzcy)+1)*py[None,:]-mu_zy-pen_c*errzy
		mean_grad = np.mean(grad,axis=0)
		return (grad-mean_grad,mean_grad)
	return grad_obj
def getMvGradObjPzcx(beta,px,pxcy,pen_c):
	def grad_obj(pzcx,pz,pzcy,mu_z,mu_zy):
		gamma = 1/beta
		errz = np.sum(pzcx*px[None,:],axis=1) - pz
		errzy= pzcx@pxcy - pzcy
		grad= gamma*px[None,:]*(np.log(pzcx)+1)+(mu_z+pen_c*errz)[:,None]*px[None,:]+(mu_zy+pen_c*errzy)@pxcy.T
		mean_grad = np.mean(grad,axis=0)
		return (grad-mean_grad,mean_grad)
	return grad_obj


## DRS-IB TYPE I
def getDrsmarkFuncObjPz(beta,px,pen_c):
	def val_obj(pz,pzcx,mu_z):
		gamma = 1/beta
		errz = pz - np.sum(pzcx * px[None,:],axis=1)
		# (gamma-1)H(Z) + <mu_z,>
		return (1-gamma)*np.sum(pz * np.log(pz)) + np.sum(mu_z * errz) + 0.5*pen_c*(np.linalg.norm(errz)**2)
	return val_obj
def getDrsmarkFuncObjPzcx(beta,px,py,pxcy,pycx,pen_c):
	def val_obj(pzcx,pz,mu_z):
		gamma = 1/beta
		errz = pz - np.sum(pzcx * px[None,:],axis=1)
		pzcy = pzcx @ pxcy
		# -gamma H(Z|X) + H(Z|Y)
		return gamma * (np.sum(pzcx*px[None,:]*np.log(pzcx))) - (np.sum(pzcy*py[None,:]*np.log(pzcy))) \
				+np.sum(mu_z*errz)+ 0.5*pen_c*(np.linalg.norm(errz)**2)
	return val_obj

def getDrsmarkGradObjPz(beta,px,pen_c):
	def grad_obj(pz,pzcx,mu_z):
		gamma = 1/ beta
		errz = pz - np.sum(pzcx*px[None,:],axis=1)
		grad = (1-gamma) * (np.log(pz)+1)+ mu_z + pen_c*errz
		mean_grad = np.mean(grad)
		return (grad-mean_grad,mean_grad)
	return grad_obj
def getDrsmarkGradObjPzcx(beta,px,py,pxcy,pycx,pen_c):
	def grad_obj(pzcx,pz,mu_z):
		gamma = 1/beta
		errz = pz - np.sum(pzcx*px[None,:],axis=1)
		pzcy = pzcx @ pxcy
		grad = (gamma * (np.log(pzcx)+1) -(np.log(pzcy)+1)@pycx-(mu_z+pen_c*errz)[:,None])*px[None,:]
		mean_grad = np.mean(grad,axis=0)
		return (grad-mean_grad[None,:],mean_grad)
	return grad_obj

