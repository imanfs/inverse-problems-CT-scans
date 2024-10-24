import astra
import matplotlib.pyplot as plt
import numpy as np
import skimage
from utils.gmres import gmres_solver, L_array
from utils.plotting import MSE, format_plot, f_subplots, g_subplots
from utils.sinogram import BP_recon, create_sinogram, explicit_radon


def normalise(f):
    f = np.nan_to_num(f)
    if f.max() == 0:
        return np.zeros_like(f) 
    else:
        return f / f.max()
    
class counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


# Create volume geometries
f = np.load('SLphan.npy')
plt.imshow(f)
format_plot('1a')
v,h = f.shape

n_angles = 180
max_angle = np.pi
n_samples = 128

vol_geom,proj_geom,projector_id,sinogram_id,sinogram = create_sinogram(f,n_angles,max_angle,128,v,h)
_,_,projector1_id,sinogram1_id,sinogram1 = create_sinogram(f,n_angles,max_angle,95,v,h)

fig,ax = plt.subplots(1,2,figsize=(10,6))
ax[0].imshow(sinogram)
ax[1].imshow(sinogram1)
format_plot('1b')


f_rec = BP_recon(vol_geom,projector_id,sinogram_id,'BP')
f_subplots(f,normalise(f_rec),'1c')

g_rec = create_sinogram(f_rec,n_angles,max_angle,128,v,h,return_sino=True)
g_subplots(sinogram,g_rec,'1ci')

f_rec = BP_recon(vol_geom,projector_id,sinogram_id,'FBP')
f_subplots(f,f_rec,'1d')

g_rec = create_sinogram(f_rec,n_angles,max_angle,128,v,h,return_sino=True)
g_subplots(sinogram,g_rec,'1di')

thetas = [30,100,1000,10000]
fig,ax = plt.subplots(2,len(thetas),figsize=(17,8))
ax[0,0].set_ylabel("BP")
ax[1,0].set_ylabel("FBP")
for i,theta in enumerate(thetas):
    gNoisy = astra.functions.add_noise_to_sino(sinogram,theta)
    gNoisy_id = astra.data2d.create('-sino',proj_geom,gNoisy)

    f_rec = BP_recon(vol_geom,projector_id,gNoisy_id,'BP')

    f_rec_FBP = BP_recon(vol_geom,projector_id,gNoisy_id,'FBP')

    ax[0,i].imshow(f_rec)
    ax[1,i].imshow(f_rec_FBP)
    ax[1,i].set_xlabel(r"$\theta$ = " + str(theta))
format_plot('1e',ax=ax)


f_small = skimage.transform.rescale(f,scale=0.5)
v_small,h_small = f_small.shape

### sparse
n_samples = 95
n_angles = 45
max_angle = np.pi

A_sp,W_sp = explicit_radon(n_angles,max_angle,n_samples,v_small,h_small,sparse_svd=False)

### limited
n_angles = 45
max_angle = np.pi/4
angles = np.linspace(0,max_angle,n_angles,endpoint=False)

A_lim,W_lim = explicit_radon(n_angles,max_angle,n_samples,v_small,h_small,sparse_svd=False)

fig,ax = plt.subplots(1,2)
ax[0].imshow(A_sp)
ax[1].imshow(A_lim)
format_plot('2a',ax=ax)

plt.plot(W_sp,label='Sparse angles',lw=0.7)
plt.plot(W_lim,label='Limited angles',lw=0.7)
plt.legend()
format_plot('2b',ax=ax)

_,_,_,_,sinogram = create_sinogram(f_small,45,np.pi,95,v_small,h_small)
g_explicit = A_sp@f_small.flatten()
g_explicit = g_explicit.reshape(n_samples,n_angles)
g_subplots(sinogram.T,g_explicit,'2c')

# %%
n_angles = [45,90,135,180]
max_angle = np.pi
fig_sp,ax_sp = plt.subplots()
for n in n_angles:
  _,W = explicit_radon(n,max_angle,n_samples,v_small,h_small)
  ax_sp.plot(W[::-1],lw=0.7,label=str(n) + " angles")
ax_sp.legend()
format_plot('2i')  

n_angles = 45
max_angles = [np.pi/2,2*np.pi/3, 3*np.pi/4,np.pi]
fig_lim,ax_lim = plt.subplots()
for m in max_angles:
  _,W = explicit_radon(n_angles,m,n_samples,v_small,h_small)
  degrees = int(round((m * 180/np.pi)))
  ax_lim.plot(W[::-1],lw=0.7,label=str(degrees) + " degrees")
ax_lim.legend()
format_plot('2ii')  

## Finding optimal alpha using L-curve residuals

alphas = 10.0**np.linspace(-3,4,15)
n_angles = 45
n_samples = 95
max_angle = np.pi
fi_norms,r_norms,keys = L_array(f_small,alphas,max_angle,n_angles,n_samples,v_small,h_small,"tikhonov")

max_angle = np.pi/4
fi_norms1,r_norms1,keys1 = L_array(f_small,alphas,max_angle,n_angles,n_samples,v_small,h_small,"tikhonov")

fig,ax = plt.subplots(1,2,figsize=(10,6))
ax[0].loglog(fi_norms,r_norms,'.-',lw=0.7,label='L-curve residuals')
idx_sp = 5
ax[0].scatter(fi_norms[idx_sp],r_norms[idx_sp],c='g',marker='x',
              label=r"Optimal $\alpha$ = " + f"{alphas[idx_sp]:.2f}")
ax[0].set_xlabel(f'$||f||$'), ax[1].set_xlabel(f'$||f||$')
ax[0].set_ylabel(f'$||r||$'), ax[1].set_ylabel(f'$||r||$')
ax[0].set_title("L-curve for sparse angle case"),ax[1].set_title("L-curve for limited angle case")
ax[1].loglog(fi_norms1,r_norms1,'.-',lw=0.7,label='L-curve residuals')
idx_lim = 6
ax[1].scatter(fi_norms1[idx_lim],r_norms1[idx_lim],c='g',marker='x',
              label=r"Optimal $\alpha$ = " + f"{alphas[idx_lim]:.2f}")
ax[0].legend(), ax[1].legend()
format_plot('3a')

# ## Sparse angle case

alpha=alphas[idx_sp]
v,h = f.shape
n_angles = [45,90,135,180]
max_angle = np.pi
n_samples = 95
fig,ax = plt.subplots(2,4,figsize=(17,8))
ax[0,0].set_ylabel("0-order Tikhonov")
ax[1,0].set_ylabel("1st order Tikhonov")
for i,n in enumerate(n_angles):
  vol_geom,proj_geom,projector_id,g_id,g = create_sinogram(f,n,max_angle,n_samples,v,h)
  gNoisy = astra.functions.add_noise_to_sino(g,10000)
  gNoisy_id = astra.data2d.create('-sino',proj_geom,gNoisy)
  
  f_gmres_tik = gmres_solver(gNoisy_id,projector_id,max_angle,n,n_samples,v,h,alpha,"tikhonov",callback=None)
  f_gmres_grad = gmres_solver(gNoisy_id,projector_id,max_angle,n,n_samples,v,h,alpha,"gradient",callback=None)

  ax[0,i].imshow(f_gmres_tik)
  ax[0,i].set_title("MSE " + f"{MSE(f_gmres_tik,f):.1e}")

  ax[1,i].imshow(f_gmres_grad)
  ax[1,i].set_title("MSE " + f"{MSE(f_gmres_grad,f):.1e}")

  ax[0,i].set_xlabel(str(n) + " angles")
  ax[1,i].set_xlabel(str(n) + " angles")

format_plot('3i',ax=ax)

# ## Limited angle case

alpha=alphas[idx_lim]
v,h = f.shape
max_angles = [np.pi/4,np.pi/2,3*np.pi/4,np.pi]
n_angles = 45
fig,ax = plt.subplots(2,4,figsize=(17,8))
ax[0,0].set_ylabel("0-order Tikhonov")
ax[1,0].set_ylabel("1st order Tikhonov")
for i,m in enumerate(max_angles):
  vol_geom,proj_geom,projector_id,g_id,g = create_sinogram(f,n_angles,m,n_samples,v,h)
  gNoisy = astra.functions.add_noise_to_sino(g,10000)
  gNoisy_id = astra.data2d.create('-sino',proj_geom,gNoisy)
  
  f_gmres_tik = gmres_solver(gNoisy_id,projector_id,m,n_angles,n_samples,v,h,alpha,"tikhonov",callback=None)
  f_gmres_grad = gmres_solver(gNoisy_id,projector_id,m,n_angles,n_samples,v,h,alpha*2,"gradient",callback=None)
  
  ax[0,i].imshow(f_gmres_tik)
  ax[0,i].set_title("MSE " + f"{MSE(f_gmres_tik,f):.1e}")

  ax[1,i].imshow(f_gmres_grad)
  ax[1,i].set_title("MSE " + f"{MSE(f_gmres_grad,f):.1e}")
  
  degrees = int(round((m * 180/np.pi)))
  ax[0,i].set_xlabel(str(degrees) + " degrees")
  ax[1,i].set_xlabel(str(degrees) + " degrees")

format_plot('3ii',ax=ax)
