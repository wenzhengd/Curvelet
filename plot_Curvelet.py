from ast import Pass
from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import pywt


def plot_2D_curve(curve_data, plt_title ='2D curve'):
    """
    A function to plot the 2d curve:

    input:
    curve_data: (len*2  2d-array) - {{x(t1), y(t1)},...., {x(tN), y(tN)}}
    -------------------------------------------------------------------
    output:
    visualization: the colorfunction is slowly changing by adapting to curve length
    """
    x = curve_data[:, 0]
    y = curve_data[:, 1]
    t = np.arange(len(x)) # Assuming t is just the index of the points

    # Normalize t to range between 0 and 1 for the colormap
    norm = plt.Normalize(t.min(), t.max())

    # Create a colormap
    cmap = plt.get_cmap('plasma')

    # Create colors based on the colormap
    colors = cmap(norm(t))

    # Plot the data
    plt.figure(figsize=(4,4))
    for i in range(len(x)-1):
        plt.plot(x[i:i+2], y[i:i+2], color=colors[i])

    # Create a scatter plot for the colorbar
    sc = plt.scatter(x, y, c=t, cmap='plasma')

    # Add a colorbar
    plt.colorbar(sc, label='Time (t)')

    # Labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(plt_title)

    # Show the plot
    plt.show()



def plot_2D_tantrix(tantrix_data, plt_title = '2D tantrix'):
    """
   A function to plot the 2d curve:

    input:
    curve_data: (len*2  2d-array) - {{x(t1), y(t1)},...., {x(tN), y(tN)}}
    -------------------------------------------------------------------
    output:
    (L): the 2D curve that should on S^2 circle if it is real_tantrix 
    (R): the x(t) and y(t) plt of tantrix r(t)
    """
    x = tantrix_data[:, 0]
    y = tantrix_data[:, 1]
    t = np.arange(len(x)) # Assuming t is just the index of the points

    # Create a figure with two subplots: one on the left and one on the right
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # Plotting data on the left subplot
    norm = plt.Normalize(t.min(), t.max())      ## Normalize t to range between 0 and 1 for the colormap
    cmap = plt.get_cmap('plasma')               # Create a colormap
    colors = cmap(norm(t))                      # Create colors based on the colormap
    sc = ax1.scatter(x, y, c=t, cmap='plasma')
    cbar = plt.colorbar(sc, ax=ax1, label='Time (t)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(plt_title)
    ax1.set_aspect('equal')  # Set the aspect ratio to be equal to make the plot square

    ax2.plot(t, x, label='real_ x(t)', color='g')
    ax2.plot(t, y, label='real_ y(t)', color='b')
    ax2.set_title('x-y projection')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x/y')
    ax2.legend()

# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================


    # Define the Daubechies wavelet
def daubechies_wavelet(n, j, k, t, center=3.5, bandwidth=1.3, level=10):
    """
    Returns the Daubechies wavelet psi_{j,k}(t) for given parameters.
    
    Parameters:
    n (int): Daubechies wavelet order
    j (int): Scale index
    k (int): Shift index
    t (np.ndarray): Support (range of t values)
    level (int): Resolution level for the wavelet function (default: 10)
    
    Returns:
    np.ndarray: Wavelet values psi_{j,k}(t) for the given support t
    """
    # Define the wavelet
    wavelet = pywt.Wavelet(f'db{n}')
    
    # Generate the wavelet function at the specified resolution level
    phi, psi, tt = wavelet.wavefun(level=level)
    
    # Scale and translate the wavelet function
    tt = tt * 2.0 ** bandwidth - center
    tt = tt * 2.0 ** j -k
    psi = psi / np.sqrt(2.0**j)
    
    # Interpolate the wavelet function to match the provided support x
    wavelet_function = interp1d(tt, psi, kind='linear', bounds_error=False, fill_value=0.0)
    return wavelet_function(t)

def db4_wavelet(j, k, t, center=3.5, bandwidth=1.3, level=10):
    return daubechies_wavelet(4, j, k, t, center, bandwidth, level)



def plot_dbWVs(jk_list, T_i,T_f):
    """
    A function to plot the wavelets:

    input:
    jk_list: (len*2  2d-array) - {{j1, k1},...., {jN, kN}}
    -------------------------------------------------------------------
    output:
    plot of wavelets
    """
    plt.figure(figsize=(8,2))
    times  =  np.linspace(T_i, T_f, 500)
    for j,k in jk_list:
        wv =  [db4_wavelet(j,k, t) for t in times]
        plt.plot(times, wv ,label =None)
    #plt.axvline(x = T_i, color = 'gray', label = None)   
    #plt.axvline(x = T_f, color = 'gray', label = None)    
    #plt.legend(None)
    plt.xlabel('t')
    plt.ylabel(r'$\psi$')     
    plt.show()



# Define the ricker wavelet(Mexican hat wavelet)
def ricker_wavelet(j,k,t, Ti=0, Tf=10):
    """
    
    """
    t = (t-(Ti+Tf)/2)/(Tf-Ti)*6.0             # regularizer to match T_i and T_f
    t = (t-k)*2**j
    return (-2.0+4.0*t**2) * np.exp(-t** 2) 

def plot_rickerWVs(jk_list, coeff_list=None, T_i=0,T_f=10, plt_title = 'Ricker WVs'):
    """
    A function to plot the wavelets:

    input:
    jk_list: (len*2  2d-array) - {{j1, k1},...., {jN, kN}}
    -------------------------------------------------------------------
    output:
    plot of wavelets
    """
    times  =  np.linspace(T_i-1, T_f+1, 500)
    if coeff_list is None:
        plt.figure(figsize=(8,2))
        for j,k in jk_list:
            #coef = coeff_list[j][]
            wv =  [ricker_wavelet(j,k, t, T_i,T_f) for t in times]
            plt.plot(times, wv ,label =None) 
        plt.xlabel('t')
        plt.axvline(x=T_i, color='c', linestyle='--', linewidth=1)
        plt.axvline(x=T_f, color='c', linestyle='--', linewidth=1)
        plt.ylabel(r'$\psi$')
        plt.title(plt_title)     
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
        for idx in np.arange(len(jk_list)):
            coeff_x = coeff_list[idx,0]
            wv_x = coeff_x * np.array([ricker_wavelet(jk_list[idx,0],jk_list[idx,1], t, T_i,T_f) for t in times])
            ax1.plot(times, wv_x)
        ax1.set_xlabel('t')
        ax1.set_ylabel('c*wv [x]')
        ax1.axvline(x=T_i, color='c', linestyle='--', linewidth=1)
        ax1.axvline(x=T_f, color='c', linestyle='--', linewidth=1)
        for idx in np.arange(len(jk_list)):
            coeff_y = coeff_list[idx,1]
            wv_y = coeff_y * np.array([ricker_wavelet(jk_list[idx,0],jk_list[idx,1], t, T_i,T_f) for t in times])
            ax2.plot(times, wv_y)
        ax2.set_xlabel('t')
        ax2.set_ylabel('c*wv [x]')
        ax2.axvline(x=T_i, color='c', linestyle='--', linewidth=1)
        ax2.axvline(x=T_f, color='c', linestyle='--', linewidth=1)
        plt.suptitle(plt_title)
        plt.show()      

    

# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================




if __name__ == "__main__":
    pass
    #random_pseudo_curve = np.array([ [np.cos(0.05*t), np.sin(0.14*t)] for t in np.linspace(0,10,1000)])
    #plot_2D_curve(random_pseudo_curve)
    #plt.savefig('2d_curve_with_color_gradient.png')
