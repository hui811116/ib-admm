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
		raw_grad = (beta-1)*(np.log(pz)+1)+mu_z+pen_c*pen_z+b_omega*(np.log(pz/pz_last))
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
