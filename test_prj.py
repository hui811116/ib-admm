import numpy as np

y0 = np.array([1.5931,1.7712,-1.8788,-0.0614]).T

nz = len(y0)
nx = 1
y_srt = np.flip(np.sort(y0)) # along each pzcx given a x # flip to make it descending
y_cum = np.cumsum(y_srt)
_tmp_arr = 1./np.arange(1,nz+1)
dbg_cond = y_srt + _tmp_arr * (1.0-y_cum)
print('dbg_cond',dbg_cond)
max_log = (y_srt + _tmp_arr * (1.0-y_cum))>0
print('log',max_log)
rho_idx = np.sum(max_log.astype('int32'))
lamb_all = np.zeros((nx,))
lamb_all = (1.0/rho_idx) * (1.0-y_cum[rho_idx-1])
print('lamba',lamb_all)
new_pzcx = np.maximum(y0+np.repeat(np.expand_dims(lamb_all,axis=0),nz,axis=0),0)
print(new_pzcx)