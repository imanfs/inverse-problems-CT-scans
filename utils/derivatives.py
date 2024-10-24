import numpy as np

def applyDx(u):
    '''Applies the first derivative along the first dimension (rows)'''
    Dx = np.diff(u, axis=0)
    Dx = np.vstack([Dx, np.zeros((1, u.shape[1]))])
    return Dx

def applyDy(u):
    '''Applies the first derivative along the second dimension (columns)'''
    Dy = np.diff(u, axis=1)
    Dy = np.hstack([Dy, np.zeros((u.shape[0], 1))])
    return Dy

def applyDxTrans(u):
    '''Applies the transpose of the first derivative along the first dimension (rows)'''
    DxT = np.vstack([-u[0:1, :], -np.diff(u[:-1, :], axis=0), u[-2:-1, :]])
    return DxT

def applyDyTrans(u):
    '''Applies the transpose of the first derivative along the second dimension (columns)'''
    DyT = np.hstack([-u[:, 0:1], -np.diff(u[:, :-1], axis=1), u[:, -2:-1]])
    return DyT

def laplacian(f,gamma=0):
    if not isinstance(gamma, np.ndarray):
        gamma = np.eye(f.shape[0])
    dxx = np.sqrt(gamma)@ applyDxTrans(np.sqrt(gamma)@applyDx(f))
    dyy = np.sqrt(gamma)@ applyDyTrans(np.sqrt(gamma)@applyDy(f))
    laplace_f = dxx + dyy
    
    return laplace_f.flatten()

def laplacian(f):
    dxx = applyDxTrans(applyDx(f))
    dyy = applyDyTrans(applyDy(f))
    laplace_f = dxx + dyy
    
    return laplace_f.flatten()
