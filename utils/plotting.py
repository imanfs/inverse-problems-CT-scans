import numpy as np
import matplotlib.pyplot as plt

#### FORMATTING FUNCTIONS ####

def format_plot(name,show_axis=False,ax=None):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 144
    plt.rc('font', size=10)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    plt.rc('xtick.major', size=4, width=1)
    plt.rc('ytick.major', size=4, width=1)
    plt.rc('axes', linewidth=1, labelsize='medium', titlesize='medium');
    if show_axis:
        plt.axhline(color='k', lw=0.5)
        plt.axvline(color='k', lw=0.5);
    if ax is not None:
        for axis in ax.ravel():
            axis.set_xticks([])
            axis.set_yticks([]);
    plt.savefig("figs/" + name, bbox_inches='tight');
    plt.show();

def MSE(f_rec,f_true):
    '''
    Returns mean squared error between each iterate reconstruction f_k
    and the original image f_true
    '''
    N = f_true.size
    return (1/N) * np.sum((f_rec - f_true)**2)

def f_subplots(f,f_rec,plot_name):
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(f)
    ax[1].imshow(f_rec)
    ax[2].imshow(f-f_rec)

    ax[0].set_title("$f_{true}$")
    ax[1].set_title("$f_{recon}$ (MSE " + f"{MSE(f_rec,f):.1e})")
    ax[2].set_title("$f_{true} - f_{recon}$")
    plt.colorbar(ax[2].imshow(f-f_rec),ax=ax[2],fraction=0.046, pad=0.04)
    format_plot(plot_name,ax=ax)

def g_subplots(Af,g_rec,plot_name):
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(Af)
    ax[1].imshow(g_rec)
    ax[2].imshow(Af-g_rec)

    ax[0].set_title("$Af_{true}$")
    ax[1].set_title("$g_{recon}$")
    ax[2].set_title("$Af_{true} - g_{recon}$")
    plt.colorbar(ax[2].imshow(Af-g_rec),ax=ax[2],fraction=0.063, pad=0.04)
    format_plot(plot_name,ax=ax)
