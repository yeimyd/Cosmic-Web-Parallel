# Cosmic-Web-Parallelized

Parallel Version of https://github.com/forero/IllustrisWeb/blob/master/compute_cosmic_web.ipynb using numba (http://numba.pydata.org/)

To use this script require the installation of the following python modules:
     
     h5py 
     numba
     matplotlib
     scipy
     numpy
     
     
Run in a teminal`python CosmicWeb_Smooth.py`

This script use Dark matter density data (in format HDF5) of Illustris TNG300-2 to created the smooth cosmic web.

1. Define: 

  `L_box   #  box Length in Mpc/h`
  
  `Nmesh   #number of mesh e.g. 256, 512, 1024 `
  
  `smooth_scale # Smooth Scale `
  
 
 2. The output file is HDF5 format and contain:
  
  density_smooth
  
  lambda1
  
  lambda2
  
  lambda3   #lambda1, lambda2 and lambda3, are Eigenvalues of Hessian Matrix
  
  potential
  
  hessian 
  
  eigenvector1
  
  eigenvector2
  
  eigenvector2   #eigenvector1, eigenvector2 and eigenvector3, are Eigenvectors of Hessian Matrix
  
  
  
  
  
![Screenshot](https://user-images.githubusercontent.com/10146082/117058545-5eec5680-ace4-11eb-9191-3fff8739c603.png)
