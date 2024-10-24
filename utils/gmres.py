from sinogram import create_sinogram,BP_recon
import numpy as np
import astra
import scipy as sp
from derivatives import laplacian
from tqdm import tqdm

def A_f(f,n_angles,max_angle,n_samples,v,h):
    """Blurs an image using a gaussian filter"""
    vol_geom,proj_geom,projector_id,sinogram_id,sinogram = create_sinogram(f,n_angles,max_angle,n_samples,v,h)
    Af = sinogram.flatten()
    return vol_geom,proj_geom,projector_id,sinogram_id,Af

def AT_y(vol_geom,projector_id,sinogram_id):
    """Blurs an image using a gaussian filter"""
    f_rec = BP_recon(vol_geom,projector_id,sinogram_id,'BP')
    AT_y = f_rec.flatten()

    return AT_y

def ATA(f,n_angles,max_angle,n_samples,vol_geom,v,h,alpha,reg_type):
    _,_,projector_id,sinogram_id,Af = A_f(f,n_angles,max_angle,n_samples,v,h)
    A_Ty = AT_y(vol_geom,projector_id,sinogram_id)
    if reg_type == "tikhonov":
        z = A_Ty.reshape(f.shape) + alpha*f
    elif reg_type == "gradient":
        z = A_Ty + alpha*laplacian(f.reshape((v,h)))
    return z.reshape((-1,1))

def gmres_solver(g_id,projector_id,max_angle,n_angles,n_samples,v,h,alpha,reg_type,callback=None):
    """
    Implements the GMRES solver for normal equations. 
    g is the blurred image to be reconstructed
    alpha is the regularisation parameter
    sigma is the blurring parameter passed to A (imblur(f))
    """
    vol_geom = astra.create_vol_geom(v,h)
    angles = np.linspace(0,max_angle,n_angles,endpoint=False)
    M = v*h
    N = M
    ATg = AT_y(vol_geom,projector_id,g_id)
    A = sp.sparse.linalg.LinearOperator((M,N),
                                        matvec=lambda f: ATA(f,n_angles,max_angle,n_samples,vol_geom,v,h,alpha,reg_type))
    gmresOutput = sp.sparse.linalg.gmres(A, ATg.flatten(),
                                         callback=callback,maxiter=700) #gmres output
    f_alpha = gmresOutput[0]
    return f_alpha.reshape(v,h)


# # Alpha choice

def L_curve(g,g_id,projector_id,alpha,max_angle,n_angles,n_samples,v,h,reg_type):
    fi = gmres_solver(g_id,projector_id,max_angle,n_angles,n_samples,v,h,alpha,reg_type,callback=None)
    Af = create_sinogram(fi,n_angles,max_angle,n_samples,v,h,return_sino=True)
    r = g - Af
    fi_norm = np.linalg.norm(fi)
    r_norm = np.linalg.norm(r)
    return fi_norm,r_norm

def L_array(f,alphas,max_angle,n_angles,n_samples,v,h,reg_type):
    vol_geom,proj_geom,projector_id,g_id,g = create_sinogram(f,n_angles,max_angle,n_samples,v,h)
    gNoisy = astra.functions.add_noise_to_sino(g,10000)
    gNoisy_id = astra.data2d.create('-sino',proj_geom,gNoisy)
    r_norms = np.zeros(len(alphas))
    fi_norms = np.zeros(len(alphas))
    keys = []
    for i, alpha in tqdm(enumerate(alphas)):
        fi_norm,r_norm = L_curve(gNoisy,gNoisy_id,projector_id,alpha,max_angle,n_angles,n_samples,v,h,reg_type)
        fi_norms[i],r_norms[i] = fi_norm,r_norm
        keys.append([fi_norm,r_norm,alpha])
    return fi_norms,r_norms,keys
