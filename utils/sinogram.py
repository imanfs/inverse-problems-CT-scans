import numpy as np
import scipy as sp
import astra

def create_sinogram(f,n_angles,max_angle,n_samples,v,h,return_sino=False):
    vol_geom = astra.create_vol_geom(v,h)
    angles = np.linspace(0,max_angle,n_angles,endpoint=False)
    proj_geom = astra.create_proj_geom('parallel',1.,n_samples,angles)
    projector_id = astra.create_projector('strip', proj_geom, vol_geom)
    sinogram_id, sinogram = astra.create_sino(f.reshape(v,h), projector_id)
    if return_sino:
        return sinogram
    return vol_geom,proj_geom,projector_id,sinogram_id,sinogram

def BP_recon(vol_geom,projector_id,sinogram_id,BP_type):
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)
    # Set up the parameters for a reconstruction via back-projection
    cfg = astra.astra_dict(BP_type)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ProjectorId'] = projector_id
    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    # Run back-projection and get the reconstruction
    astra.algorithm.run(alg_id)
    f_rec = astra.data2d.get(rec_id)
    #return f_rec if BP_type == 'FBP' else normalize_array(f_rec)
    return f_rec 

def explicit_radon(n_angles,max_angle,n_samples,v,h,sparse_svd=True):
    angles = np.linspace(0,max_angle,n_angles,endpoint=False)
    A_exp = []
    vol_geom = astra.create_vol_geom(v,h)
    proj_geom = astra.create_proj_geom('parallel',1.,n_samples,angles)
    projector_id = astra.create_projector('strip', proj_geom, vol_geom)
    row_idx = 0
    for col in range(v): # For each column
        for row in range(h): # For each row in the current column
            dummy_img = np.zeros((v,h))
            dummy_img[row][col] = 1
            # Radon transform (generate sinogram)
            sinogram_id, sinogram = astra.create_sino(dummy_img.T, projector_id)
            A_exp.append(sinogram.reshape(-1,1,order='F'))
        row_idx +=1 
    A_exp = np.hstack(A_exp)
    if sparse_svd:
        _,W,_ = sp.sparse.linalg.svds(A_exp)
    else:
        _,W,_ = sp.linalg.svd(A_exp)
    return A_exp,W
