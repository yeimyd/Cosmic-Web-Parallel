# Cosmic-Web-Parallel

Parallel Version of https://github.com/forero/IllustrisWeb/blob/master/compute_cosmic_web.ipynb using numba (http://numba.pydata.org/)

To use this script is neccesary install the following python modules:
     
     h5py 
     numba
     matplotlib
     scipy
     numpy
     
     
Run in a teminal python CosmicWeb_Smooth.py

This script use Dark matter density data of Illustris TNG300-2 to created the smooth cosmic web.

1. Define: 
    L_box = 205 #  box Length Mpc/h
    Nmesh1=256  # Number of mesh e.g. 256 512 or 1024
    The smooth scale  
    
2. The output file is HDF5 format and contain
    
    density_smooth  
    lambda1
    lambda2 
    lambda3 # lambda1,lambda2 and lambda3 are the eigenvalues of hessian matrix
    potential
    hessian 
    eigenvector1 
    eigenvector2
    eigenvector3 #eigenvector1, eigenvector2 and eigenvector3  are the eigenvalues of hessian matrix
    
    ![Screenshot](https://user-images.githubusercontent.com/10146082/116950337-521b2480-ac4a-11eb-9ec1-4feedf8d5b2b.png)

    
    
    
    
