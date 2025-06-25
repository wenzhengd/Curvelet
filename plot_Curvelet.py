from ast import Pass
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm

import numpy as np
from scipy.interpolate import interp1d

import pywt

from wavelet_factory import wavelet

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


def plot_3D_curve(curve_data, plt_title= '3D curve'):
    """
    A function to plot the 3d curve:

    input:
    curve_data: (len*3  3d-array) - {{x(t1), y(t1), z(t1)},...., {x(tN), y(tN), y(tN)}}
    -------------------------------------------------------------------
    output:
    visualization: the colorfunction is slowly changing by adapting to curve length
    """
    x = curve_data[:, 0]
    y = curve_data[:, 1]
    z = curve_data[:, 2]
    t = np.arange(len(x)) # Assuming t is just the index of the points
    # Normalize t to [0, 1] for color mapping
    t_n = (t - t.min()) / (t.max() - t.min())
    
    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot with color mapped to t_array
    colors = cm.gist_rainbow(Normalize()(t_n))
    ax.plot(x, y, z, color='black', alpha=0)  # Invisible line for structure

    for i in range(len(x) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=colors[i])

    # Set labels and limits
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_zlabel('z(t)')
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min(), z.max()])

    # Add colorbar
    mappable = cm.ScalarMappable(cmap=cm.gist_rainbow, norm=Normalize(vmin=t.min(), vmax=t.max()))
    mappable.set_array(t)
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label('t')

    plt.show()


def plot_2D_tantrix(tantrix_data, plt_title = '2D tantrix', curve_data =None, curvature_data = None):
    """
    A function to plot the 2d curve:

    input:
    curve_data: (len*2  2d-array) - {{x(t1), y(t1)},...., {x(tN), y(tN)}}
    -------------------------------------------------------------------
    output:
    (L): the 2D curve that should on S^2 circle if it is real_tantrix 
    (R): the x(t) and y(t) plt of tantrix r(t)
    (RR): plot the curvuture
    """
    x = tantrix_data[:, 0]
    y = tantrix_data[:, 1]
    t = np.arange(len(x)) # Assuming t is just the index of the points


    if curve_data is None:
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

    else: # shoud plot curvature 
        # Create a figure with three subplots: 

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
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

        X, Y = curve_data[:, 0], curve_data[:, 1]
        sc = ax3.scatter(X, Y, c=t, cmap='rainbow',  s=8) # Set 's' to a smaller value to reduce thickness
        cbar = plt.colorbar(sc, ax=ax1, label='Time (t)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('real_curve')
        ax3.set_aspect('equal')  # Set the aspect ratio to be equal to make the plot square

        ax4.plot(t, curvature_data, label=r'$\Omega(t)$', color='r')   
        ax4.set_title('curvature plot')
        ax4.set_xlabel('t')
        ax4.set_ylabel(r'$\Omega(t)$')
        ax4.legend()    


def plot_3D_tantrix(tantrix_data, plt_title = '3D tantrix', curve_data =None, curvatureS_data = None):
    """
    A function to plot the 3d curve:

    input:
    curve_data: (len*3  3d-array) - {{x(t1), y(t1), z(t1)},...., {x(tN), y(tN), z(tN)}}
    -------------------------------------------------------------------
    output:
    (L): the 2D curve that should on S^3 circle if it is real_tantrix 
    (M): plot the curvuture
    (R): plot the torsion
    """ 
    x = tantrix_data[:, 0]
    y = tantrix_data[:, 1]
    z = tantrix_data[:, 2]
    t = np.arange(len(x))  
    
    # Normalize t to [0, 1] for color mapping
    t_n = (t - t.min()) / (t.max() - t.min())

    if curvatureS_data is None:
        """
        only plot tantrix
        """
        # Create the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the curve using the color map based on t_array
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=t.min(), vmax=t.max())
        lc = ax.plot(x, y, z, color='black', alpha=0)  # Invisible line for structure
        
        for i in range(len(x) - 1):
            ax.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=cm.gist_rainbow(norm(t[i])))
        
        # Add a background sphere for aesthetic effect
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        sphere_x = np.cos(u) * np.sin(v)
        sphere_y = np.sin(u) * np.sin(v)
        sphere_z = np.cos(v)
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color='lightgray', alpha=0.3)
        
        # Set labels and view angle
        ax.set_xlabel('x(t)')
        ax.set_ylabel('y(t)')
        ax.set_zlabel('z(t)')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        # Optional: Add color bar
        mappable = cm.ScalarMappable(cmap=cm.gist_rainbow, norm=norm)
        mappable.set_array(t)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label('t')
        
        plt.show()

    else:
        """
        plot tantrix, curvature, torsion
        """
        curvature = curvatureS_data[0]
        torsion = curvatureS_data[1]
        
        fig = plt.figure(figsize=(18, 6))
        
        # Plot the 3D tantrix in the first subplot
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Plot the curve using the color map based on t_array
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=t.min(), vmax=t.max())
        
        for i in range(len(x) - 1):
            ax1.plot(x[i:i + 2], y[i:i + 2], z[i:i + 2], color=cm.gist_rainbow(norm(t[i])))
        
        # Add a background sphere for aesthetic effect
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        sphere_x = np.cos(u) * np.sin(v)
        sphere_y = np.sin(u) * np.sin(v)
        sphere_z = np.cos(v)
        ax1.plot_surface(sphere_x, sphere_y, sphere_z, color='lightgray', alpha=0.3)
        
        # Set labels and view angle
        ax1.set_xlabel('x(t)')
        ax1.set_ylabel('y(t)')
        ax1.set_zlabel('z(t)')
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        ax1.set_title('3D Tantrix')

        # Plot the curvature in the second subplot
        ax2 = fig.add_subplot(132)
        ax2.plot(t, curvature, color='blue', lw=2)
        ax2.set_xlabel('t')
        ax2.set_ylabel('Curvature')
        ax2.set_title('Curvature vs t')

        # Plot the torsion in the third subplot
        ax3 = fig.add_subplot(133)
        ax3.plot(t, torsion, color='green', lw=2)
        ax3.set_xlabel('t')
        ax3.set_ylabel('Torsion')
        ax3.set_title('Torsion vs t')

        plt.tight_layout()
        plt.show()


# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================




#def plot_dbWVs(jk_list, T_i,T_f):
#    """
#    A function to plot the wavelets:
#
#    input:
#    jk_list: (len*2  2d-array) - {{j1, k1},...., {jN, kN}}
#    -------------------------------------------------------------------
#    output:
#    plot of wavelets
#    """
#    plt.figure(figsize=(8,2))
#    times  =  np.linspace(T_i, T_f, 500)
#    for j,k in jk_list:
#        wv =  [db4_wavelet(j,k, t) for t in times]
#        plt.plot(times, wv ,label =None)
#    #plt.axvline(x = T_i, color = 'gray', label = None)   
#    #plt.axvline(x = T_f, color = 'gray', label = None)    
#    #plt.legend(None)
#    plt.xlabel('t')
#    plt.ylabel(r'$\psi$')     
#    plt.show()






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
            wv =  [wavelet(j,k, t, T_i,T_f) for t in times]
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
            wv_x = coeff_x * np.array([wavelet(jk_list[idx,0],jk_list[idx,1], t, T_i,T_f) for t in times])
            ax1.plot(times, wv_x)
        ax1.set_xlabel('t')
        ax1.set_ylabel('c*wv [x]')
        ax1.axvline(x=T_i, color='c', linestyle='--', linewidth=1)
        ax1.axvline(x=T_f, color='c', linestyle='--', linewidth=1)
        for idx in np.arange(len(jk_list)):
            coeff_y = coeff_list[idx,1]
            wv_y = coeff_y * np.array([wavelet(jk_list[idx,0],jk_list[idx,1], t, T_i,T_f) for t in times])
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




if __name__ == 'main':
    pass
    #random_pseudo_curve = np.array([ [np.cos(0.05*t), np.sin(0.14*t)] for t in np.linspace(0,10,1000)])
    #plot_2D_curve(random_pseudo_curve)
    #plt.savefig('2d_curve_with_color_gradient.png')
