"""
produce different orders of ricker wavelets 
"""
import numpy as np 
from scipy.interpolate import interp1d
import pywt 


############################################################
###       Ricker Wavelet functions  
############################################################

# Define the ricker wavelet(Mexican hat wavelet)
def ricker_wavelet_2(j,k,t, Ti=0, Tf=10):
    """
    """
    t = (t-(Ti+Tf)/2)/(Tf-Ti)*6.0             # regularizer to match T_i and T_f.  
                                              # SEE @Mathematica //Ricker_WV_property.nb  2nd
    t = (t-k)*2**j
    return (-2.0+4.0*t**2) * np.exp(-t** 2)   # Ricker-2

def ricker_wavelet_4(j,k,t, Ti=0, Tf=10):
    """
    """
    t = (t-(Ti+Tf)/2)/(Tf-Ti)*6.0             # regularizer to match T_i and T_f.  
                                              # SEE @Mathematica //Ricker_WV_property.nb  4th
    t = (t-k)*2**j
    return (6-24*t**2 + 8*t**4)*np.exp(-t**4)/np.pi   # Ricker-4


# choose the wavelet 
def wavelet(j,k,t, Ti=0, Tf=10):
    return ricker_wavelet_4(j,k,t, Ti, Tf)   # use Ricker-2  wavelet
    #return ricker_wavelet_(j,k,t, Ti, Tf)    # use Ricker-4  wavelet




############################################################
    # Define the Daubechies wavelet
############################################################

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





if __name__ == "__main__":
    print(wavelet(j=-0.2,k=0.0,t=2, Ti=0, Tf=10))       # test = -0.0035988331724544353