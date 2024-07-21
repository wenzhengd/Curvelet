import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.integrate import quad, IntegrationWarning
import scipy.optimize as opt
from scipy.optimize import least_squares, curve_fit

from joblib import Parallel, delayed
import time
import warnings
from functools import wraps



from plot_Curvelet import plot_2D_curve, plot_dbWVs

ansatz_curve = np.array(" the tantrix of a closed curve") 

# Define the Gaussian wavelet(Mexican hat wavelet)
def gaussian_wavelet(j,k,t, center=0, bandwidth=1):
    """
    THE WAVELET SHOULD BE EXTENDED TO DAUBECHIES !!!
    先让cos sin 的code 测试一下daubechies 看看收敛的效果 ！！
    the function should be normalized and need bandwidth_param to match T_f-t_i 
    """
    t = (t-center)/bandwidth                     # regularizer to match T_i and T_f
    return (1 - ((t - k) ** 2) / (j ** 2)) * np.exp(-((t - k) ** 2) / (2 * j ** 2))

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
    tt = tt * 2.0 ** bandwidth - center             # the modified \psi_{0,0} to fit the T_i=0 and T_f=10
    tt = tt * 2.0 ** j -k                           
    psi = psi / np.sqrt(2.0**j)
    
    # Interpolate the wavelet function to match the provided support x
    wavelet_function = interp1d(tt, psi, kind='linear', bounds_error=False, fill_value=0.0)
    return wavelet_function(t)

def db4_wavelet(j, k, t, center=3.5, bandwidth=1.3, level=10):
    return daubechies_wavelet(4, j, k, t, center, bandwidth, level)

# Parameterizer to return arc-length curve based on input curve
def curve_parameterization(pseudo_curve, curve_len =1):
    """
    input:  pseudo_curve (\tilde{r}(t))
    curve_len (float):  the total_length of the returned real_curve
    output: real_curve (r(t)) with arclength (Lenght ==1 ) parameterization done
    """ 
    # follows from GPT:
    x , y = pseudo_curve[:,0], pseudo_curve[:,1]
    # Step 1: Calculate the distances between consecutive points
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    # Step 2: Compute the cumulative arc length
    arc_length = np.concatenate(([0], np.cumsum(distances)))

    # Step 3: Interpolate x and y with respect to the arc length
    interp_x = interp1d(arc_length, x, kind='linear')
    interp_y = interp1d(arc_length, y, kind='linear')

    # Define a new set of arc length values for interpolation
    s_new = np.linspace(0, arc_length[-1], num=1000)

    # Get the interpolated x and y values
    x_new = interp_x(s_new)
    y_new = interp_y(s_new)      
    return np.dstack((x_new,y_new))[0]/arc_length[-1]* curve_len          # make the len(r(t)) == curve_len  !

def get_curve_len(real_curve):
    """
    calculate the total length of a real_curve r(t)
    -------------
    """
    pass                            # useless for now [in curve_parameterization() it handles]
    return None

# Decompose the curve into the WVs by jk_pool
def curve_fit(real_curve, jk_pool, cxy_pool, T_i, T_f, cxy_bound_lo = None, cxy_bound_hi = None ):
    """
    Given real_curve & pool_jk as wvs and pool_Cxy as init_guess
    Return: optimzized_coffe,  that can update pool_Cxy
    -------------------------------------------------------
    Parameters:
    real_curve    (times*2 2d-darray): the parameterized real curve
    jk_pool         (size*2 2d-array): the (j,k) pairs in the pool
    cxy_pool        (size*2 2d-array): the coeff_Cxy of each (j,k)
    cxy_bound       (size*2 2d-array): the the bound of each cxy
    T_i             (float):            inital time of real_curve
    T_f             (float):            final  time of real_curve
    """
    # Define the target function
    times = np.linspace(T_i, T_f,  len(real_curve))
    target_x, target_y = real_curve[:,0], real_curve[:,1]         
    
    # Define the linear combination of given coeffs on db4 
    def approximation_function(t, coeffs, j_k_):
        approx = np.zeros_like(t)
        for c, [j,k] in zip(coeffs, j_k_):
            approx += c * daubechies_wavelet(n=4, j=j,k=k, t=t)
        return approx

    # Define the residuals function for least-squares optimization
    def residuals(coeffs, t, x_y_, j_k_):
        # the x_y_ can be either x(t) or y(t) for 2d-curve
        return x_y_ - approximation_function(t, coeffs, j_k_)
    
    # % % % Set the scales and translations:   t_k_ = jk_pool
    
    # % % % Initialize the coefficients    initial_cx = cxy_pool[:,0] initial_cy = cxy_pool[:,1]

    # Optimize the coefficients
    init_guess_x,   init_guess_y  = cxy_pool[:,0], cxy_pool[:,1]
    if cxy_bound_lo == None : cxy_bound_lo = [-10]*len(jk_pool)
    if cxy_bound_hi == None : cxy_bound_hi = [ 10]*len(jk_pool)
    # ((internal_debug)) print(init_guess_x ,init_guess_y) #print(init_guess_x.shape,times.shape,target_x.shape, jk_pool.shape) #print(cxy_bound_lo) #print(residuals(init_guess_x, times, target_x, jk_pool))
    optimized_cx = least_squares(residuals, x0= init_guess_x,  bounds=(cxy_bound_lo, cxy_bound_hi), args=(times, target_x, jk_pool) ).x
    optimized_cy = least_squares(residuals, x0= init_guess_y,  bounds=(cxy_bound_lo, cxy_bound_hi), args=(times, target_y, jk_pool) ).x

    return np.array([optimized_cx, optimized_cy])


def cache_method(func):
    @wraps(func)
    def wrapper(self, t1, t2):
        # Ensure that the instance has a cache attribute
        if not hasattr(self, '_cache'):
            self._cache = {}
        cache = self._cache
        
        # Check if the result is in the cache
        if (t1, t2) in cache:
            return cache[(t1, t2)]
        else:
            result = func(self, t1, t2)
            cache[(t1, t2)] = result
            return result
    
    return wrapper



# The Curve to Wavelet-Manifold optimization formalism
class Curvelet():
    """
    Construct the curve & wavelet & their mutual map 
    AND optimization
    """
    def __init__(self, ansatz , T_i, T_f, tol_leak=0.5 , seg_chop=10, N=1000) -> None:
        self.real_curve = ansatz * (T_f-T_i)                    # the parameterized curve // dynamically changed in optimizer 💥💥💥💥 unless len(ansatz) ==1 
        self.pseudo_curve = None                                # the un_param curve \tilde{r}(t) by lin.con. non-leakly (good) WVs \
                                                                #  // dynamically changed in optimizer 
                                                                #  // \tilde~{{x(t1), y(t1)},{{x(t2), y(t2)}......}
        self.T_i = T_i                                          # initial time
        self.T_f = T_f                                          # finial time    
        self.L   = T_f-T_i                                      # real curve length
        self.seg_chop = seg_chop                                # chop number when check LOCAL normalization error of pseudo_curve
        self.N  = N                                             # 'pixel' (smallest time slices) of discretization of  ANY curve
        self.windows = np.array(np.split(np.linspace(T_i, T_f,N), seg_chop))    # partition time into segments/windows for normalizer check

        self.pool_jk = None                           # the scale-shift (j,k)-pair of WVs kept in the pool // all pass leak test // dynamical change
        self.pool_Cxy = None                          # the coeff of (j,k) kept in the pool // all pass leak test // dynamical change
        self.tol_leak = tol_leak                      # the tol used in WV_leakage
        self.leakage_error = 0                        # the current leakage error
        self.log_leakage = None                       # log the leakage @ loop 
        self.log_leak_maker  = np.array([])           # log the leaker (j,k) in each round @ loop 
        self.log_worst_window = []                    # log the bad_window   @ loop 
        self.log_pcurve_norm_error =[]                # log the p_curve's error: int |r(t)^2-1| dt  @ loop
        # =log_worst_window ~ self.log_deviations = []                      # log the most_deviated segments: 0<_<9 @loop

        self.db4_sigma_enforcer = -1                  # experimental scale-regulizer for [T_i, T_f]=[0,10]
        self.db4_mu_enforcer    = 2                   # experimental shift-regulizer for [T_i, T_f]=[0,10]

    def initialize_pool(self):
        """
        choose the best j,k, sigma on ansatz -- determined by  T_f-T_i
        --------------------------------------------------------------
        UPDATE: It initialize pool_jk as leading small j & initizae pool_Cxy as 0 !! 
        """
        # # Here we consider three resulution of pool_jk based on bandwidth & center:
        self.pool_jk = np.concatenate((np.array([[0,0]]),   \
                                       np.array([[-0.25, k] for k in np.arange(-4,4,2)]),\
                                       np.array([[-1   , k] for k in np.arange(-6,0,1)]) ), axis =0)
        self.pool_Cxy = np.zeros_like(self.pool_jk)
        return None
    
    def real_curve_to_WV(self):
        """
        use the curve_fit to decompose ansazt into inital_WVs in the pool
        -----------------------------------------------------------------
        UPDATE: It does NOT change pool_jk but will update pool_Cxy !
        """ 
        optm_coeff = curve_fit( real_curve = self.real_curve, 
                                   jk_pool = self.pool_jk, 
                                  cxy_pool = self.pool_Cxy,
                                       T_i =  self.T_i, 
                                       T_f = self.T_f, 
                              cxy_bound_lo = None, 
                              cxy_bound_hi = None ) #  💥💥💥💥 should set based on real physics here, 
        self.pool_Cxy = optm_coeff.T   # the updated should be transposed  
        return None
    
    @cache_method           #use cash method to avoid calculate same (jk) for many times
    def mom_integral(self, idx, K):
        """
        calcualte the integral of <psi_jk * m-th moment> where m<=K 
        """
        f = lambda m, t, idx:  1/2**m * t**m * db4_wavelet(self.pool_jk[idx,0], self.pool_jk[idx,1], t)
        return np.sum(Parallel(n_jobs=-1)\
                        (delayed(lambda m: quad(lambda t: abs(f(m, t, idx)), 
                        self.T_i, self.T_f, limit=100 )[0])(m)  for m in range(0,K+1)))  # check 0-th ~2nd mom

    def leakage_check_surgery(self, K=2, penalty =0.8):
        """
        check whether the containts in pool have large leakage (non_0_moment)
        -----------------------------------------------------------------
        INPUT:  K (int): the highest order moments considered
        -----------------------------------------------------------------
        UPDATE: detects the leakage of pool_jk and surgery/penalize pool_Cxy;
                log the self.log_leak_maker
        """
        leak_makers_rnd = np.array([])                # the leak_makers in this round 


        # loop the pool_jk to check leakage of each (j,k) pair
        for idx  in range(len(self.pool_jk)):
            # Use joblib to parallelize the integration and sum the results directly
            # Suppress the IntegrationWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=IntegrationWarning)
                leak_jk = self.mom_integral(idx, K)
                
            # C_xy surgery if above tol_leak 
            if leak_jk * self.pool_Cxy[idx, 0] > self.tol_leak:             #C-x
                leak_makers_rnd = np.concatenate((leak_makers_rnd , self.pool_jk[idx] ) ) 
                #leak_makers_rnd.append(self.pool_jk[idx].flatten())              
                self.pool_Cxy[idx,0] = penalty * self.tol_leak/leak_jk

            if leak_jk * self.pool_Cxy[idx, 1] > self.tol_leak:             #C-y
                #leak_makers_rnd = np.append(leak_makers_rnd , np.array([self.pool_jk[idx]]) )
                leak_makers_rnd = np.concatenate((leak_makers_rnd , self.pool_jk[idx] ) ) 
                self.pool_Cxy[idx,1] = penalty * self.tol_leak/leak_jk


        # log the leak_makers
        self.log_leak_maker = self.log_leak_maker.reshape(len(self.log_leak_maker)//2, 2)
        # 💥 💥💥💥 update pool_jk and pool_Cxy // 此处，只是把pool_Cxy变小了，没有把 jk 剔除，是不是一个隐患？-- 比如说每次迭代，optimizer又会把Cxy of this jk 调大，反反复复
        # 💥 💥💥💥 按理说这个leakage——check 应该区分 x 和 y
        return None
    
    def pool_to_pseudo_curve(self):
        """
        use linear combination to obtain the pseudo_curve based on current pool
        -----------------------------------------------------------------
        UPDATE: the pseudo_curve will be updated
        """
        times = np.linspace(self.T_i, self.T_f, self.N)
        x = np.zeros_like(times)
        y = x
        for [j,k], [c_x, c_y] in zip(self.pool_jk, self.pool_Cxy):
            x +=  c_x * db4_wavelet(j,k, times)
            y +=  c_y * db4_wavelet(j,k, times)
        self.pseudo_curve = np.dstack((x,y))[0]             # udpate the psedo_curve using current_pool
        return None

    def norm_check(self, partition =10):
        """
        check the normalization of pseudo_curve and return the main_seg of deviation
        -----------------------------------------------------------------
        UPDATE: update the normalization error of pseudo_curve
        Return:  deviations for all woindows 
        """
        pseudo_curve_windows = np.array(np.split(self.pseudo_curve, partition))
        #print('psuedo_curves', self.pseudo_curve.shape, self.pseudo_curve)
        #print('windows', pseudo_curve_windows.shape, pseudo_curve_windows)
        deviations = np.zeros(partition)
        for window in range(partition):
            x_window = pseudo_curve_windows[window, :, 0]  # NOTICE
            y_window = pseudo_curve_windows[window, :, 1]
            # handle the deviation_n = \int dt |rd(t)-1| W_n(t)
            #print('======check====',x_window.shape, x_window)
            window_deviation = np.absolute(np.square(x_window) +  np.square(y_window) - np.ones_like(x_window))
            deviations[window] = simps(window_deviation, self.windows[window] )  
        self.log_pcurve_norm_error.append(np.sum(deviations))       # The norm_error of p_curve is the sum of each window's devi
        return deviations

    def fine_cwv_candidate(self):
        """
        add fine cwv candidates from the information obtained from norm_check()
        ------------------------------
        """
        worst = np.argmax(self.norm_check())
        self.log_worst_window.append(worst)          # log the bad_window
        # need determine how to update js and ks
        j1 = self.pool_jk[-1,0] * 1.5
        j2,j3 = j1,j1
        k1, k2, k3 = np.random.rand(), np.random.rand(), np.random.rand()
        ### update the pool by addding candidate cww
        self.pool_jk = np.concatenate((self.pool_jk, np.array([[j1,k1]]), np.array([[j2,k2]]), np.array([[j3,k3]])), axis=0)
        self.pool_Cxy= np.concatenate((self.pool_Cxy,np.array([[0,0]]),   np.array([[0,0]]),   np.array([[0,0]])),   axis=0)
        return None

    #  ~~~~~ The iterative optimizer ~~~~~~ 
    def curve_wavelet_optimizer(self, tol_converge, N_iter=0):
        """
        tol_converge: when the converge of pseudo_curve -> real_curve
        N_iter: MAX interation in optimization_loop
        """

        i=0           # interation index     

        tic = time.time()

        for i in range(N_iter):     # later change to 'do while'
            """
            Interation to update real_curve based on pool_WV and update pool_WV based on real_curve
            """
            print('=== now the optimization is on ', i ,'-th iteration')
            
            # project the real_curve to WV space by pool // # it will update pool.C_xy: 
            #  // interally call curve_fit()
            self.real_curve_to_WV()

            # leakage check for 10 segments // # log the "log_leak_maker" & do surgery by depressing C_xy
            self.leakage_check_surgery()

            # update the self.pseudo_curve // # pool.Cxy & pool.jk -> p_curve:
            self.pool_to_pseudo_curve()   

            # update the self.real_curve //   # p_curve -> real_curve:                                              
            self.real_curve = curve_parameterization( pseudo_curve = self.pseudo_curve, \
                                                      curve_len = self.L)
            
            # Check the norm & add fine_CWV // from p_curve -> finer (jk)
            self.fine_cwv_candidate()

            # Curve_fit to adjust the pool's coeff again // 
            curve_fit(self.real_curve, self.pool_jk, self.pool_Cxy, T_i =self.T_i, T_f = self.T_f)
            
            #if self.log_curve_error > tol_converge:
            #    break
            
            i += 1

        toc = time.time()

        print(f"Elapsed time: {toc - tic} seconds")





##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

T_i=0
T_f=10
#print(gaussian_wavelet(1,1,5.55) )
# print(daubechies_wavelet(n=4, j=1, k=1,  t=np.linspace(0,10,10) ) ) # Pass


random_pseudo_curve = np.array([ [np.cos(0.1*t), np.sin(0.12*t)] for t in np.linspace(0,10,1000)])
real_curve_from_random_pseudo = curve_parameterization(random_pseudo_curve, curve_len=10)
#print(real_curve_from_random_pseudo.shape)
#print('x(t): ',real_curve_from_random_pseudo[:,0])  # Pass
#print('y(t): ',real_curve_from_random_pseudo[:,1])  # Pass#

#jk_pool = np.concatenate((np.array([[0,0]]),   \
#                         np.array([[-0.25, k] for k in np.arange(-4,4,2)]),\
#                         np.array([[-1   , k] for k in np.arange(-6,0,1)]) ))

#print(jk_pool[:,0].shape)
#fit_result = curve_fit(real_curve= real_curve_from_raom_pseudo, 
#          jk_pool= jk_pool,     
#          cxy_pool= np.ones_like(jk_pool),
#          T_i=0, T_f =10)
#print(fit_result)


ansatz_curve = real_curve_from_random_pseudo

#plot_2D_curve(ansatz_curve)

test = Curvelet(ansatz=  ansatz_curve, T_i=T_i, T_f=T_f, tol_leak=1)
  
""" 
#print(test.pool_jk)
test.initialize_pool()
#tmp = test.pool_jk
print(np.array(test.pool_Cxy))
test.real_curve_to_WV()
print('==now the pool_jk is : ',test.pool_jk)
print('==now the pool_Cxy is : ',test.pool_Cxy)

test.leakage_check_surgery()
# print('pool_jk',test.pool_jk)
print('pool_Cxy after surgery is: ', test.pool_Cxy)
print('leak_maker ',test.log_leak_maker)
print('p_curve', test.pseudo_curve)
test.pool_to_pseudo_curve()
print('p_curve', test.pseudo_curve)
print('bafore candidate', test.pool_jk)
test.fine_cwv_candidate()
print('worst window', test.log_worst_window)
print('after candidate', test.pool_jk)
print('after candidate', test.pool_Cxy)


#plot_2D_curve(test.real_curve)
#plot_dbWVs(test.pool_jk, T_i, T_f)
"""

test.initialize_pool()
test.real_curve_to_WV()
print('jk=', test.pool_jk, 'Cxy=', test.pool_Cxy)
test.pool_to_pseudo_curve()

test.leakage_check_surgery()
test.real_curve = curve_parameterization(pseudo_curve = test.pseudo_curve, \
                                           curve_len = test.L)

test.fine_cwv_candidate()


##################
test = Curvelet(ansatz=  ansatz_curve, T_i=T_i, T_f=T_f, tol_leak=1)
test.initialize_pool()
test.curve_wavelet_optimizer(tol_converge=1, N_iter=10)
print('pool_jk= ', test.pool_jk)
print('pool_Cxy=', test.pool_Cxy)
print('p_curve=', test.pseudo_curve)
print('r_curve=', test.real_curve)
print('log_leak_maker', test.log_leak_maker)
print('log_worst_window',test.log_worst_window)
print('log_pcurve_norm_error',test.log_pcurve_norm_error)


