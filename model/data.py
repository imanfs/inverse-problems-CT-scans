import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from utils.sinogram import BP_recon, create_sinogram
from utils.plotting import format_plot
import torch
import astra

def draw_ellipses(n,opacity=0.1,plot=False):
    """
    Draws a grayscale image with 5 to 20 random, shaded ellipses.
    """
    image = np.zeros((n, n))

    num_ellipses = np.random.randint(5, 20)
    for _ in range(num_ellipses):
        center_x = np.random.randint(0, n)
        center_y = np.random.randint(0, n)
        axis_length_x = np.random.randint(n//48, n//3)
        axis_length_y = np.random.randint(n//48, n//3)
        orientation = np.random.rand() * np.pi * 2
        rr, cc = ellipse(center_x, center_y, axis_length_x, axis_length_y, rotation=orientation, shape=image.shape)
        image[rr, cc] = image[rr, cc] + opacity
    if plot:
        plt.imshow(image)
        plt.colorbar()
        plt.axis('off')
        plt.show()
    return image/image.max()



def generate_train(n_angles, max_angle, n_samples, theta, v, h, plot=False,grad=False):
    '''
    Generates ellipses and noisy filtered backprojections  
    Returns: tuple containing target (ftrue) and sample (f_recon)
    -- note that order of outputs is generally flipped 
    -- but this was more convenient for plotting 
    '''
    ftrue = draw_ellipses(v)
    vol_geom, proj_geom, projector_id, g_id, g = create_sinogram(ftrue, n_angles, max_angle, n_samples, v, h)
    gNoisy = astra.functions.add_noise_to_sino(g, theta)
    gNoisy_id = astra.data2d.create('-sino', proj_geom, gNoisy)
    f_BP = BP_recon(vol_geom, projector_id, gNoisy_id, 'FBP')
    
    if plot:
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(ftrue)
        ax[1].imshow(f_BP)    
        format_plot('data',ax=ax)
    
    ftrue = torch.as_tensor(ftrue).unsqueeze(0)
    f_BP = torch.as_tensor(f_BP).unsqueeze(0)  
    if grad:
        return (f_BP, gNoisy), ftrue
    else:
        return (ftrue, f_BP),gNoisy