import astra
import matplotlib.pyplot as plt
import numpy as np
import pywt
from utils.gmres import AT_y, L_array
from utils.plotting import MSE, format_plot, f_subplots
from utils.sinogram import BP_recon, create_sinogram
import skimage

def normalise(f):
    f = np.nan_to_num(f)
    if f.max() == 0:
        return np.zeros_like(f) 
    else:
        return f / f.max()

# ## Haar Wavelets

f = np.load('SLphan.npy')

v,h = f.shape

n_angles = 180
max_angle = np.pi
n_samples = 128

vol_geom,proj_geom,projector_id,sinogram_id,sinogram = create_sinogram(f,n_angles,max_angle,128,v,h)
_,_,projector1_id,sinogram1_id,sinogram1 = create_sinogram(f,n_angles,max_angle,95,v,h)

f_small = skimage.transform.rescale(f,scale=0.5)
v_small,h_small = f_small.shape

coeffs = pywt.wavedec2(f,'haar',level = 7)
lvls = len(coeffs) - 1
fig,ax = plt.subplots(3,lvls,figsize=(13,5))
ax[0,0].set_ylabel("Horizontal")
ax[1,0].set_ylabel("Vertical")
ax[2,0].set_ylabel("Diagonal")

for i in range(1,len(coeffs)):
    for j in range(3):
        if i == 1:
            vmin = (min(coeffs[i]))
            vmax = (max(coeffs[i]))
            ax[j][i-1].imshow(np.array(coeffs[i][j]),vmin=vmin, vmax=vmax)
        else:
            ax[j][i-1].imshow(np.array(coeffs[i][j]))
    ax[j][i-1].set_xlabel(f"Coefficient {len(coeffs) - i}")
format_plot('4a',ax=ax)

coeffs = pywt.wavedec2(f,'haar',level = 7)
f_rec = pywt.waverec2(coeffs,'haar')
f_subplots(f,f_rec,'4b')
np.allclose(f,f_rec)

def thresholdFunction(coeffs,tRange,tVal):
    for i in tRange:
        coeffs[i+1] = [tuple(pywt.threshold(comp, tVal) for comp in detail) for detail in coeffs[i+1]]
    return coeffs

#f = plt.imread('house.png')
theta = 0.05
f_noisy = f + theta*np.random.randn(f.shape[0],f.shape[1])
coeffs = pywt.wavedec2(f_noisy,'haar',level = 7) # W

## varying threshold parameter
tVals = [0.01,0.05,0.1]
tRange = [0,1,2,3,4,5,6]
fig,ax = plt.subplots(3,3,figsize=(6,6))
for i,tVal in enumerate(tVals):
    coeffsT = thresholdFunction(coeffs,tRange,tVal)
    f_denoise = pywt.waverec2(coeffsT,'haar') # W^-1
    ax[i][0].imshow(f_noisy)
    ax[i][0].set_title("$f_{noisy}$")   
    ax[i][1].imshow(f_denoise)
    ax[i][1].set_title("$f_{denoise}$ (MSE " + f"{MSE(f_denoise,f):.1e})")
    ax[i][2].imshow(f-f_denoise)
    ax[i][2].set_title("$f_{true} - f_{denoise}$")
format_plot('4ci',ax=ax)

theta = 0.1
f_noisy = f + theta*np.random.randn(f.shape[0],f.shape[1])
coeffs = pywt.wavedec2(f_noisy,'haar',level = 7) # W
## varying threshold parameter
tVal = 0.05
tRange = [2,3,4]
fig,ax = plt.subplots(3,3,figsize=(6,6))
for i in range(3):
    coeffsT = thresholdFunction(coeffs,tRange,tVal)
    f_denoise = pywt.waverec2(coeffsT,'haar') # W^-1
    ax[i][0].imshow(f_noisy)
    ax[i][0].set_title("$f_{noisy}$")   
    ax[i][1].imshow(f_denoise)
    ax[i][1].set_title("$f_{denoise}$ (MSE " + f"{MSE(f_denoise,f):.1e})")
    ax[i][2].imshow(f-f_denoise)
    ax[i][2].set_title("$f_{true} - f_{denoise}$")
    tRange = [x + 1 for x in tRange]
format_plot('4cii',ax=ax)

def soft_threshold(f,levels,alpha,lambda_,tRange):
    mu = alpha*lambda_
    Wf = pywt.wavedec2(f,'haar',level = levels)
    S_mu = thresholdFunction(Wf,tRange,mu)
    W_1SWf = pywt.waverec2(S_mu,'haar')
    return W_1SWf

def iterative_thres(f,n_angles,max_angle,n_samples,vol_geom,v,h,alpha,threshold,step_size,tRange,maxIter=5000):
    vol_geom,proj_geom,projector_id,g_id,g = create_sinogram(f,n_angles,max_angle,n_samples,v,h)
    g = astra.functions.add_noise_to_sino(g,10000)
    g_id = astra.data2d.create('-sino',proj_geom,g)
    f_k = normalise(BP_recon(vol_geom,projector_id,g_id,'BP'))  ## create f0
    for _ in range(maxIter):
        vol_geom,_,projector_id,sinogram_id,Af_k = create_sinogram(f_k,n_angles,max_angle,n_samples,v,h)
        # create sinogram id for Afk_g so that it can be passed into ATy
        # (because ATy takes sinogram ID, not sinogram as input)
        Afk_g_id = astra.data2d.create('-sino',proj_geom, Af_k - g)
        ATAfk_g = AT_y(vol_geom,projector_id,Afk_g_id).reshape(v,h)
        f_kplus1 = soft_threshold(f_k - step_size*ATAfk_g,7,alpha,step_size,tRange)
        print(MSE(f_k,f) - MSE(f_kplus1,f))
        if MSE(f_k,f) - MSE(f_kplus1,f) <= threshold:
            break
        else:
            f_k = f_kplus1
    return f_k,MSE(f_k,f)

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

alpha = alphas[idx_sp]

n_angles = 45
max_angle = np.pi
threshold = 0.0001
tRange = [1,2,3,4,5,6]
arr = [1,10,30,35,37,38,39]
step_sizes = [x * 1e-5 for x in arr]
mse_arr = np.zeros((len(alphas),len(step_sizes)))
best_score = float('inf')
for a,alpha in enumerate(alphas):
    for s,step_size in enumerate(step_sizes):
        f_recon,mse =  iterative_thres(f,n_angles,max_angle,n_samples,vol_geom,v,h,alpha,threshold,step_size,tRange)
        if mse < best_score:
                # re-assign best values for MSE, gamma & sigma
                best_score = mse
                best_alpha = alpha
                best_step_size = step_size  
        mse_arr[a,s] = mse

fig,ax = plt.subplots()
xx, yy = np.meshgrid(np.log(step_sizes),np.log(alphas))
ax = fig.add_subplot(projection = '3d')
surf1 = ax.plot_surface(xx, yy, mse_arr,cmap=plt.cm.coolwarm)
plt.xlabel(r"log $\lambda$")
plt.ylabel(r"log $\alpha$")
plt.show()

f_k,mse = iterative_thres(f,n_angles,max_angle,n_samples,vol_geom,v,h,best_alpha,threshold,best_step_size,tRange)
f_subplots(f,f_k,'')

alpha = alphas[idx_lim]
step_size = 30e-5
n_angles = 45
max_angle = np.pi/2
threshold = 0.0001
tRange = [4,5,6]
arr = [1,10,30,35]
step_sizes = [x * 1e-5 for x in arr]
mse_arr = np.zeros((len(alphas),len(step_sizes)))
best_score = float('inf')
for a,alpha in enumerate(alphas):
    for s,step_size in enumerate(step_sizes):
        f_recon,mse =  iterative_thres(f,n_angles,max_angle,n_samples,vol_geom,v,h,alpha,threshold,step_size,tRange)
        if mse < best_score:
                best_score = mse
                best_alpha = alpha
                best_step_size = step_size  
        mse_arr[a,s] = mse

fig,ax = plt.subplots()
xx, yy = np.meshgrid(np.log(step_sizes),np.log(alphas))
ax = fig.add_subplot(projection = '3d')
surf1 = ax.plot_surface(xx, yy, mse_arr,cmap=plt.cm.coolwarm)
plt.show()

threshold = 0.000001
f_recon,mse =  iterative_thres(f,n_angles,max_angle,n_samples,vol_geom,v,h,best_alpha,threshold,1e-6,tRange)
f_subplots(f,f_recon,'')
