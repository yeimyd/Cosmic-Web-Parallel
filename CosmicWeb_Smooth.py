import h5py
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage.filters
import numpy.linalg


L_box = 205 # in Mpc/h
Nmesh1=256
L_celda=L_box/Nmesh1

smooth_scale = 20.0
filename = '/Volumes/cosmic_web/DM256/Density256_302.hdf5' #CIC density
f = h5py.File(filename, 'r')
print(f.keys())
data = f['density'][:,:,:]
data_smooth = scipy.ndimage.filters.gaussian_filter(data,smooth_scale)
data = (data - np.mean(data))/np.mean(data)
data_smooth = (data_smooth - np.mean(data_smooth))/np.mean(data_smooth)
print(np.shape(data))
f.close()

print(smooth_scale)

fft_lowres = np.fft.fftn(data_smooth)
fft_lowres[0,0,0]=0.0
back_data = np.fft.ifftn(fft_lowres)

from numba import jit


#@nb.jit(nopython=True,parallel=True)
@nb.jit(nopython=True)

def compute(n_side,L_box,delta):  
    k_2_values = np.ones((n_side, n_side, n_side))    
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                k_i = i
                k_j = j
                k_k = k
                if i > n_side/2:
                    k_i = n_side - i
                if j > n_side/2:
                    k_j = n_side - j
                if k > n_side/2:
                    k_k = n_side - k
            
                k_2_values[i, j, k] = -(2.0*np.pi*k_i/L_box)**2 + -(2.0*np.pi*k_j/L_box)**2 + -(2.0*np.pi*k_k/L_box)**2
    k_2_values[0, 0, 0] = 1.0
    
    return k_2_values

n_side = np.shape(fft_lowres)[0]
print(n_side)
L_box = 205.0 # in Mpc/h
delta = L_box/n_side

k_2_values=compute(n_side,L_box,delta)

fft_potential = fft_lowres/k_2_values
potential = np.fft.ifftn(fft_potential) # este es ahora el potencial reducido
density = scipy.ndimage.filters.laplace(np.real(potential), mode='wrap')/(delta**2) # sanity check

r = data_smooth/density

slice_id = 10


r = density[slice_id,:,:].flatten()
print(np.count_nonzero(density<-1.0))


# https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

h = hessian(potential)


l = list(data_smooth.shape)
l.append(3)
eigenvector1 = np.empty(l, dtype=data_smooth.dtype)
np.shape(eigenvector1)


@nb.jit(nopython=True)

def eigen(l,lambda1,lambda2,lambda3,eigenvector1,eigenvector2,eigenvector3):
    for i in range(n_side):
        print(i)
        for j in range(n_side):
            for k in range(n_side):
                local_hessian = np.real(h[:,:,i,j,k])/(delta**2)
                values, vectors = np.linalg.eig(local_hessian)
                ii = np.argsort(values)
                values = values[ii]
                vectors = vectors[ii]
                eigenvector1[i,j,k,:] = vectors[:,2]
                eigenvector2[i,j,k,:] = vectors[:,1]
                eigenvector3[i,j,k,:] = vectors[:,0]
                lambda1[i, j, k] = values[2]
                lambda2[i, j, k] = values[1]
                lambda3[i, j, k] = values[0]
    return lambda1[i, j, k],lambda2[i, j, k],lambda3[i, j, k],i

l = list(data_smooth.shape)
l.append(3)
l=np.array(l)
lambda1 = data_smooth.copy()
lambda2 = data_smooth.copy()
lambda3 = data_smooth.copy()
eigenvector1 = np.empty(l, dtype=data_smooth.dtype)
eigenvector2 = np.empty(l, dtype=data_smooth.dtype)
eigenvector3 = np.empty(l, dtype=data_smooth.dtype)

print(eigen(l,lambda1,lambda2,lambda3,eigenvector1,eigenvector2,eigenvector3))


filename = './CosmicWeb_smooth_s{:.2f}.hdf5'.format(smooth_scale)
h5f = h5py.File(filename, 'w')

h5f.create_dataset('density_smooth', data=data_smooth)

h5f.create_dataset('lambda1', data=lambda1)
h5f.create_dataset('lambda2', data=lambda2)
h5f.create_dataset('lambda3', data=lambda3)

h5f.create_dataset('potential', data=np.real(potential))

h5f.create_dataset('hessian', data=h)

h5f.create_dataset('eigenvector1', data=eigenvector1)
h5f.create_dataset('eigenvector2', data=eigenvector2)
h5f.create_dataset('eigenvector3', data=eigenvector3)

h5f.close()
