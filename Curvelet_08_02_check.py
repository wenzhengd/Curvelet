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
import time


# Define the ricker wavelet(Mexican hat wavelet)
def ricker_wavelet(j,k,t, Ti=0, Tf=10):
    """
    """
    t = (t-(Ti+Tf)/2)/(Tf-Ti)*6.0             # regularizer to match T_i and T_f
    t = (t-k)*2**j
    return (-2.0+4.0*t**2) * np.exp(-t** 2) 



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


from plot_Curvelet import plot_2D_curve, plot_rickerWVs, plot_2D_tantrix

ansatz_tantrix = np.array("the tantrix ansatz") 

# Parameterizer to return arc-length curve based on input curve
def curve_parameterization(pseudo_curve, curve_len =1):
    """
    input:  pseudo_curve (\tilde{r}(t))
    curve_len (float):  the total_length of the returned real_curve
    output: real_curve (R(t)) with arclength (Lenght ==1 ) parameterization done
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
    #return np.dstack((x_new,y_new))[0]/arc_length[-1]* curve_len          # make the len(r(t)) == curve_len  !
    return pseudo_curve

# Normalize the p_tantrix to make it real_tantrix
def tantrix_normalizer(pseudo_tantrix):
    """
    input:  pseudo_tantrix(\tilde{r}(t))
    output: real_tantrix (r(t)) with normlization done |r(t)|=1 for all t  
    """
    x , y = pseudo_tantrix[:,0], pseudo_tantrix[:,1]
    norm_factor = np.sqrt(x ** 2 + y ** 2)
    x_new = x / norm_factor
    y_new = y / norm_factor
    return np.dstack((x_new,y_new))[0]
        
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
    # Define the target function x,y respectively
    times = np.linspace(T_i, T_f,  len(real_curve))
    target_x, target_y = real_curve[:,0], real_curve[:,1]         
    
    # Define the linear combination of given coeffs on WVs
    # This is a scalar function, can be used for either x(t) or y(t)
    def approximation_function(t, coeffs, j_k_):
        approx = np.zeros_like(t)
        for c, [j,k] in zip(coeffs, j_k_):
            approx = approx + c * ricker_wavelet(j=j,k=k, t=t)
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
    optimized_cx = least_squares(residuals, x0= init_guess_x,  bounds=(cxy_bound_lo, cxy_bound_hi), \
                                            args=(times, target_x, jk_pool) ).x
    optimized_cy = least_squares(residuals, x0= init_guess_y,  bounds=(cxy_bound_lo, cxy_bound_hi), \
                                            args=(times, target_y, jk_pool) ).x

    return np.array([optimized_cx, optimized_cy])

# Decompose the tantrix into the WVs by jk_pool
def tantrix_fit(real_tantrix, jk_pool, cxy_pool, T_i, T_f, cxy_bound_lo = None, cxy_bound_hi = None ):
    """
    Given real_tantrix & pool_jk as wvs and pool_Cxy as init_guess
    Return: optimzized_coeff,  that can update pool_Cxy
    -------------------------------------------------------
    Parameters:
    real_tantrix    (times*2 2d-darray): the parameterized real tantrix
    jk_pool         (size*2 2d-array): the (j,k) pairs in the pool
    cxy_pool        (size*2 2d-array): the coeff_Cxy of each (j,k)
    cxy_bound       (size*2 2d-array): the the bound of each cxy
    T_i             (float):            inital time of real_tantrix
    T_f             (float):            final  time of real_tantrix
    """
    # Define the target function x,y respectively
    times = np.linspace(T_i, T_f,  len(real_tantrix))
    target_x, target_y = real_tantrix[:,0], real_tantrix[:,1]         
    
    # Define the linear combination of given coeffs on WVs
    # This is a scalar function, can be used for either x(t) or y(t)
    def approximation_function(t, coeffs, j_k_):
        approx = np.zeros_like(t)
        for c, [j,k] in zip(coeffs, j_k_):
            approx = approx + c * ricker_wavelet(j=j,k=k, t=t)
        return approx

    # Define the residuals function for least-squares optimization
    def residuals(coeffs, t, x_y_, j_k_):
        # the x_y_ can be either x(t) or y(t) for 2d-tantrix
        return x_y_ - approximation_function(t, coeffs, j_k_)
    
    # % % % Set the scales and translations:   t_k_ = jk_pool
    
    # % % % Initialize the coefficients    initial_cx = cxy_pool[:,0] initial_cy = cxy_pool[:,1]

    # Optimize the coefficients
    init_guess_x,   init_guess_y  = cxy_pool[:,0], cxy_pool[:,1]
    if cxy_bound_lo == None : cxy_bound_lo = [-10]*len(jk_pool)
    if cxy_bound_hi == None : cxy_bound_hi = [ 10]*len(jk_pool)
    # ((internal_debug)) print(init_guess_x ,init_guess_y) #print(init_guess_x.shape,times.shape,target_x.shape, jk_pool.shape) #print(cxy_bound_lo) #print(residuals(init_guess_x, times, target_x, jk_pool))
    optimized_cx = least_squares(residuals, x0= init_guess_x,  bounds=(cxy_bound_lo, cxy_bound_hi), \
                                            args=(times, target_x, jk_pool) ).x
    optimized_cy = least_squares(residuals, x0= init_guess_y,  bounds=(cxy_bound_lo, cxy_bound_hi), \
                                            args=(times, target_y, jk_pool) ).x

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





# The Tantrix to Wavelet-Manifold optimization formalism
from turtle import width


class Curvelet():
    """
    Construct the tantrix & wavelet & their mutual map 
    AND optimization
    ------------------------------
    R(t):  error curve -- not handeld in this class
    r(t):  tantrix     --     handeld in this class
    ------------------------------
    """
    def __init__(self, ansatz , T_i, T_f, tol_leak= 0.1 , seg_chop=10, N=1000) -> None:
        self.real_tantrix = ansatz                              # the normalized tantrix // dynamically changed in optimizer 💥💥💥💥 unless len(ansatz) ==1 
        self.pseudo_tantrix = None                              # the un_normed tantrix \tilde{r}(t) by lin.con. non-leakly (good) WVs \
                                                                #  // dynamically changed in optimizer 
                                                                #  // \tilde~{{x(t1), y(t1)},{{x(t2), y(t2)}......}
        self.T_i = T_i                                          # initial time
        self.T_f = T_f                                          # finial time    
        self.L   = T_f-T_i                                      # time span /  len
        self.seg_chop = seg_chop                                # chop number when check LOCAL normalization error of pseudo_tantrix
        self.N  = N                                             # 'pixel' (smallest time slices) of discretization of  ANY tatrix
        self.windows = np.array(np.split(np.linspace(T_i, T_f,N), seg_chop))    # partition time into segments/windows for normalizer check

        self.pool_jk = None                           # the scale-shift (j,k)-pair of WVs kept in the pool // all pass leak test // dynamical change
        self.pool_Cxy = None                          # the coeff of (j,k) kept in the pool // all pass leak test // dynamical change
        self.pool_leak =[]                            #  //// /////record the moment_leak of (j,k) up to order self.K  // NOT change, but will add when pool_jk expand

        self.tol_leak = tol_leak                      # the tol used in WV_leakage
        #self.leakage_error = 0                        # the current leakage error
        self.log_leakage = np.array([])               # log the leak when call leakage_check_surgeery()   @ loop 
        self.log_leak_maker_x  = np.array([])         # log the leaker (j,k) in x-dir in each round     @ loop 
        self.log_leak_maker_y  = np.array([])         # log the leaker (j,k) in y-dir in each round     @ loop 
        self.log_worst_window = []                    # log the bad_window   @ loop 
        self.log_ptantrix_norm_error =[]              # log the p_tantrix's error: int |r(t)^2-1| dt  @ loop
        self.window_finest_j = np.zeros(seg_chop)     # what is the finest_j within this window --> help get finer_cwt_candidate
        # =log_worst_window ~ self.log_deviations = []                      # log the most_deviated segments: 0<_<9 @loop

        #self.db4_sigma_enforcer = -1                  # experimental scale-regulizer for [T_i, T_f]=[0,10]
        #self.db4_mu_enforcer    = 2                   # experimental shift-regulizer for [T_i, T_f]=[0,10]

    def initialize_pool(self):
        """
        choose the best j,k, sigma on ansatz -- determined by  T_f-T_i
        --------------------------------------------------------------
        UPDATE: It initialize pool_jk as leading small j & initizae pool_Cxy as 0 !! 
        """
        # # Here we consider three resulution of pool_jk based on bandwidth & center:
        self.pool_jk =np.concatenate([ np.array([[j,k] for j in np.linspace(0.8, 1.20, 4)  \
                                                       for k in np.linspace(-np.pi/2,  np.pi/2, 4)]) ])  

        #np.concatenate((np.array([[0,0]]),   \
        #                               np.array([[1   , k] for k in np.linspace(-2, 2, 5)]),\
        #                               np.array([[2   , k] for k in np.linspace(-2, 2, 3)])  
        #                               ), axis =0)
        self.pool_Cxy = np.zeros_like(self.pool_jk)
        self.window_finest_j =np.array([np.argmax(self.pool_jk[:,0])  \
                                        for i in np.arange(self.seg_chop)])        # pick the largest j as finest_j for each window
        return None
    
    def real_tantrix_to_WV(self, lr_bound= None ):
        """
        use the tantrix_fit to decompose ansazt into inital_WVs in the pool
        -----------------------------------------------------------------
        (input):  the lr_bound that determins the lo & hi bound <symmetric> of C_xy
                  the lr_bound >0 will be such that 
                  c_xy_bound_lo = - lr_bound
                  c_xy_bound_lo = + lr_bound
        (UPDATE): It does NOT change pool_jk but will update pool_Cxy !
        """ 
        
        optm_coeff = tantrix_fit( real_tantrix = self.real_tantrix, 
                                   jk_pool = self.pool_jk, 
                                  cxy_pool = self.pool_Cxy,
                                       T_i =  self.T_i, 
                                       T_f = self.T_f, 
                              cxy_bound_lo = (-lr_bound if lr_bound is not None else None),
                              cxy_bound_hi = ( lr_bound if lr_bound is not None else None) ) #  💥💥💥💥  
        self.pool_Cxy = optm_coeff.T   # the updated should be transposed 
        return None
    
    def _plot_init_pool_reconstruction(self):
        """
        test whether the init_pool is good by plot the  coeff*WV(t) for x and y
        and ansaz = {x(t), y(t)} 
        _____________________________________
        The self.pool_jk and self.pool_Cxy are only meant to be 
        the 1st time projection from ansatz to W_f
        -------------------------------------
        NOTICE: just for internal check
        """
        coeff = self.pool_Cxy  
        j_k_  = self.pool_jk
        times = np.linspace(self.T_i, self.T_f, self.N)

        def approximation_func(t):
            reconstruct_x = np.zeros_like(t)
            reconstruct_y = reconstruct_x
            for [c_x, c_y], [j,k] in zip(coeff,j_k_):
                tmp = ricker_wavelet(j,k,t)
                reconstruct_x = reconstruct_x + c_x * tmp
                reconstruct_y = reconstruct_y + c_y * tmp
            return (reconstruct_x,reconstruct_y)

        ansatz_x = ansatz_tantrix[:, 0]
        ansatz_y = ansatz_tantrix[:, 1]
        (WV_x, WV_y)= approximation_func(times)

        # Create a figure with two subplots: one on the left and one on the right
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
        # Plotting data on the left subplot
        ax1.plot(times, ansatz_x, label='ansatz_x', color='y')
        ax1.plot(times, WV_x,  '--',label='WV_x', color='g')
        ax1.set_title('x(t)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.legend()
        ax2.plot(times, ansatz_y, label='ansatz_y', color='y')
        ax2.plot(times, WV_y,  '--',label='WV_y', color='g')
        ax2.set_title('y(t)')
        ax2.set_xlabel('y')
        ax2.set_ylabel('t')
        ax2.legend()
        sc = ax3.scatter(ansatz_x[0:-1:10], ansatz_y[0:-1:10], \
                        c=np.arange(len(ansatz_x[0:-1:10])), cmap='rainbow',  alpha=0.4)
        cbar = plt.colorbar(sc, ax=ax3, label='Time (t)')
        #ax3.colorbar(sc, label='Time (t)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title("the ansatz tantrix r(t) ad 2D")

    def pool_to_pseudo_tantrix(self):
        """
        use linear combination to obtain the pseudo_tantrix based on current pool
        -----------------------------------------------------------------
        UPDATE: the pseudo_tantrix will be updated
        """
        times = np.linspace(self.T_i, self.T_f, self.N)
        x = np.zeros_like(times)
        y = x
        for [j,k], [c_x, c_y] in zip(self.pool_jk, self.pool_Cxy):
            #print('c_x is ', c_x, ' c_y is ', c_y)
            tmp = ricker_wavelet(j,k, times)
            x = x + c_x * tmp
            y = y + c_y * tmp
        self.pseudo_tantrix = np.dstack((x,y))[0]             # udpate the psedo_tantrix using current_pool
        return None
    
    def pseudo_to_real(self):
        """
        Parameterization arc-len to get & update real_tantrix based on the pseudo tantrix
        """
        self.real_tantrix = tantrix_normalizer( pseudo_tantrix = self.pseudo_tantrix)
        return None
        
    @cache_method           #use cash method to avoid calculate same (jk) for many times
    def mom_integral(self, idx, K):
        """
        calcualte the integral of <psi_jk * m-th moment> where m<=K 
        """
        weight = 40    # global weight to modulate the mom_leak strength
        f = lambda m, t, idx:  weight* 1/2**m * t**m * ricker_wavelet(self.pool_jk[idx,0], self.pool_jk[idx,1], t)
        # Calculate the sum of absolute values of integrals
        def calculate_integral(idx, K, T_i, T_f):
            integrals = Parallel(n_jobs=-1)( delayed(lambda m: quad(lambda t: f(m, t, idx), T_i, T_f, limit=100 )[0])(m)  for m in range(0, K+1)  )
            return np.sum(np.abs(integrals))
        return calculate_integral(idx, K, self.T_i, self.T_f)
        #return np.sum(Parallel(n_jobs=-1)\
        #                (delayed(lambda m: quad(lambda t: f(m, t, idx), 
        #                self.T_i, self.T_f, limit=100 )[0])(m)  for m in range(0,K+1)))  # check 0-th ~2nd mom

    def leakage_check_surgery(self, K=1, penalty =1.0):
        """
        check whether the containts in pool have large leakage (non_0_moment)
        -----------------------------------------------------------------
        INPUT:  K (int): the highest order moments considered
        -----------------------------------------------------------------
        UPDATE: detects the leakage of pool_jk and surgery/penalize pool_Cxy;
                log the self.log_leak_maker
        """
        leak_makers_rnd_x = np.array([])                # the leak_makers in this round 
        leak_makers_rnd_y = leak_makers_rnd_x
        leak_tot_val_x,leak_tot_val_y = 0,0             # the total leak of all wvs*COeff

        # loop the pool_jk to check leakage of each (j,k) pair
        for idx  in range(len(self.pool_jk)):
            # check the leak_jk for each j_k_ pair  [NOT include Cxy at all]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=IntegrationWarning)
                leak_jk = self.mom_integral(idx, K)         # leak_jk >=0 , as defined in mom_integral()
                self.pool_leak = np.append(self.pool_leak, leak_jk)                 # log the leak of each (jk) in jk_pool
                leak_tot_val_x = leak_tot_val_x + leak_jk * self.pool_Cxy[idx, 0]   # collect the total leakage of all pools
                leak_tot_val_y = leak_tot_val_y + leak_jk * self.pool_Cxy[idx, 1]   # collect the total leakage of all pools 
                
            # C_xy surgery if above tol_leak             
            if abs(leak_jk * self.pool_Cxy[idx, 0]) > self.tol_leak:             #C-x
                #print("idx", idx, "leak=",leak_jk,"leak*coeff=","coeff=",self.pool_Cxy[idx, 0], leak_jk * self.pool_Cxy[idx, 0])
                leak_makers_rnd_x = np.concatenate((leak_makers_rnd_x , self.pool_jk[idx] ) ) 
                #self.pool_Cxy[idx,0] = penalty * self.tol_leak/leak_jk
                self.pool_Cxy[idx,0] = 0.5* self.pool_Cxy[idx,0]

            if abs(leak_jk * self.pool_Cxy[idx, 1]) > self.tol_leak:             #C-y
                #leak_makers_rnd = np.append(leak_makers_rnd , np.array([self.pool_jk[idx]]) )
                leak_makers_rnd_y = np.concatenate((leak_makers_rnd_y , self.pool_jk[idx] ) ) 
                #self.pool_Cxy[idx,1] = penalty * self.tol_leak/leak_jk
                self.pool_Cxy[idx,1] = 0.5* self.pool_Cxy[idx,1]
        
        if leak_makers_rnd_x.size != 0:
            print('check a leak maker in x: ', leak_makers_rnd_x)
        if leak_makers_rnd_y.size != 0:
            print('check a leak maker in y: ', leak_makers_rnd_y)

        # log the total leakage along x and y 
        print('Leakage @ check [w/o surgery] is:', np.array([leak_tot_val_x,leak_tot_val_y]))
        self.log_leakage = np.concatenate( (self.log_leakage, np.array([leak_tot_val_x,leak_tot_val_y])) ) 
            
        # log the leak_makers
        self.log_leak_maker_x = np.append(self.log_leak_maker_x, leak_makers_rnd_x)
        self.log_leak_maker_y = np.append(self.log_leak_maker_y, leak_makers_rnd_y)
        self.log_leak_maker_x = self.log_leak_maker_x.reshape(len(self.log_leak_maker_x)//2, 2)
        self.log_leak_maker_y = self.log_leak_maker_y.reshape(len(self.log_leak_maker_y)//2, 2)
           
        # 💥 💥💥💥 update pool_jk and pool_Cxy // 此处，只是把pool_Cxy变小了，没有把 jk 剔除，是不是一个隐患？-- 比如说每次迭代，optimizer又会把Cxy of this jk 调大，反反复复
        
        return None
    
    def norm_check(self, partition =10):
        """
        check the normalization of pseudo_tantrix and return the main_seg of deviation
        -----------------------------------------------------------------
        UPDATE: update the normalization error of pseudo_tantrix
        Return:  deviations for all woindows 
        """
        pseudo_tantrix_windows = np.array(np.split(self.pseudo_tantrix, partition))
        deviations = np.zeros(partition)
        for window in range(partition):
            x_window = pseudo_tantrix_windows[window, :, 0]  # NOTICE
            y_window = pseudo_tantrix_windows[window, :, 1]
            # handle the deviation_n = \int dt |rd(t)-1| W_n(t)
            #print('======check====',x_window.shape, x_window)
            window_deviation = np.absolute(np.square(x_window) +  np.square(y_window) - np.ones_like(x_window))
            deviations[window] = simps(window_deviation, self.windows[window] )  
        self.log_ptantrix_norm_error.append(np.sum(deviations))       # The norm_error of p_tantrix is the sum of each window's devi
        return deviations

    def fine_cwv_candidate(self, partition=10):
        """
        add fine cwv candidates from the information obtained from norm_check()
        -----------------------------------------------------------------------
        The ricker's k=0 corresponds to t= (Ti+tf)/2 
        the center of worst (of devia) is (worst+0.5)*tau where tau = (Tf-Ti)/partition=10
        """
        worst = np.argmax(self.norm_check(partition))
        self.log_worst_window.append(worst)          # log the bad_window
        # need determine how to update js and ks
        new_j = self.pool_jk[-1,0]*np.array([2, 2.25, 2.5])
        tau = (self.T_f-self.T_i)/partition
        new_k =  2* np.array([(worst+0.5) * tau  - (self.T_f-self.T_i)/2/2**j for j in new_j])  
        # 💥💥💥 这个地方我不知道为啥要✖️最开始的系数

        ### update the pool by addding candidate cww
        self.pool_jk = np.concatenate((self.pool_jk,  np.array([[j,k] for j,k in zip(new_j, new_k)])),   axis=0)
        self.pool_Cxy= np.concatenate((self.pool_Cxy, np.array([[0,0] for j,k in zip(new_j, new_k)])),   axis=0)
        return None

    def fine_cwv_candidate(self, partition=10):
        """
        add fine cwv candidates from the information obtained from norm_check()
        -----------------------------------------------------------------------
        The ricker's k=0 corresponds to t= (Ti+tf)/2 
        the center of worst (of devia) is (worst+0.5)*tau where tau = (Tf-Ti)/partition=10
        """
        worst = np.argmax(self.norm_check(partition))
        self.log_worst_window.append(worst)          # log the bad_window
        
        # need determine how to update js and ks
        if worst ==0 or worst == partition:
            new_j = self.pool_jk[-1,0]*np.array([2.5])
        elif worst ==1 or worst == (partition-1):
            new_j = self.pool_jk[-1,0]*np.array([1.8])
        else:
            new_j = self.pool_jk[-1,0]*np.array([1.2])

        DD = (self.T_f - self.T_i )/6
        MM = (self.T_f + self.T_i )/2
        nadia = (self.T_f - self.T_i )/2
        tau = (self.T_f-self.T_i)/partition
        
        new_t = np.array([worst])*tau   # the three new t_nadia we want to have
        new_k = np.array([(-MM + 2**new_j * MM - 2**new_j * nadia + new_t)/DD  \
                            for new_j, new_t in zip(new_j, new_t)])
        
        ### update the pool by addding candidate cww
        self.pool_jk = np.concatenate((self.pool_jk,  np.array([[j,k] for j,k in zip(new_j, new_k)])),   axis=0)
        self.pool_Cxy= np.concatenate((self.pool_Cxy, np.array([[0,0] for j,k in zip(new_j, new_k)])),   axis=0)
        return None
    
    def _print_info_(self):
        """
        print the information after each iteration, that is, 
        after p_tantrix ==> r_tantrix is done
        """
        print("==================== Print Info ================ \n", \
            "Size of pool is ", self.pool_jk.size,
            ".\n Leakage of r_tantrix = ", self.log_leakage, '\n   \
            ================================================== '
            )
        return None


    def leak_error_TOTAL(self):
        """
        calculate the total leakage in the constructed "error-WV"
        """
        if self.pool_Cxy.size != self.pool_jk.size:
            raise Exception("the pool_jk and pool_Cxy have different size")
        return  np.sum(np.multiply( self.pool_Cxy, \
                self.mom_integral([idx for idx in np.arange(len(self.pool_leak))], K=1)), 
                axis=0)
    
    def norm_error_TOTAL(self):
        """
        calculate the total norm error in the constructed "error-WV"
        """
        return self.log_ptantrix_norm_error[-1] 

    #  ~~~~~ The iterative optimizer ~~~~~~ 
    def tantrix_wavelet_optimizer(self, tol_converge, N_iter=0):
        """
        tol_converge: when the converge of pseudo_tantrix -> real_tantrix
        N_iter: MAX interation in optimization_loop
        """

        i=0           # interation index   

        print('#### size of original pool=', len(self.pool_jk),' #####')  

        tic = time.time()

        for i in range(N_iter):     # later change to 'do while'
            """
            Interation to update real_tantrix based on pool_WV and update pool_WV based on real_tantrix
            """
            print('=== now the optimization is on ', i ,'-th iteration')
            
            # project the real_tantrix to WV space by pool // # it will update pool.C_xy: 
            #  // interally call tantrix_fit()

            dyn_lr_bound = 10 / (1.00**i)     # a dynamical lr_bound shrinks as loop go
            self.real_tantrix_to_WV(lr_bound= dyn_lr_bound)
            if i==0:
                pass
                #plot_2D_tantrix(self.real_tantrix, plt_title =f"xxx")

            # leakage check for 10 segments // # log the "log_leak_maker" & do surgery by depressing C_xy
            self.leakage_check_surgery()

            # update the self.pseudo_tantrix // # pool.Cxy & pool.jk -> p_tantrix:
            self.pool_to_pseudo_tantrix()   

            # update the self.real_tantrix //   # p_tantrix -> real_tantrix: 
            # It will use tantrix_normalizer !!!!!                                             
            self.pseudo_to_real()
            
            #print('before plot, |pool|= ', len(self.pool_jk), self.pool_jk)
            plot_2D_tantrix(self.pseudo_tantrix, plt_title =f"pseudo_tantrix @ {i}-th iter")
            plot_2D_tantrix(self.real_tantrix, plt_title =f"real_tantrix @ {i}-th iter")
            plot_rickerWVs(self.pool_jk,coeff_list=None, T_i= self.T_i, T_f=self.T_f,plt_title =f"Ricker WVs @ {i}-th iter")
            
            # Check the norm & add fine_CWV // from p_tantrix -> finer (jk)
            self.fine_cwv_candidate()
            #print('size of pool (after add fine) is ', len(self.pool_jk))

            # Tantrix_fit to adjust the pool's coeff again // 
            tantrix_fit(self.real_tantrix, self.pool_jk, self.pool_Cxy, T_i =self.T_i, T_f = self.T_f)

            self._print_info_()            
            i += 1

        toc = time.time()

        print(f"Elapsed time: {toc - tic} seconds")




T_i=0
T_f=10
random_pseudo_tantrix = np.array([ [np.cos(3*t), \
                                  np.sin(3*t)] \
                                for t in np.linspace(0,10,1000)])
#plot_2D_tantrix(random_pseudo_tantrix[0:-1:50])
real_tantrix_from_random_pseudo = tantrix_normalizer(random_pseudo_tantrix)
ansatz_tantrix = real_tantrix_from_random_pseudo
#plot_2D_tantrix(ansatz_tantrix)
test = Curvelet(ansatz=  ansatz_tantrix, T_i=T_i, T_f=T_f, tol_leak=1)
plot_2D_tantrix(test.real_tantrix)  #  初始化时 ansatz 就是 r_tantrix
test.initialize_pool()
test.real_tantrix_to_WV()
print('pool_jk=',test.pool_jk)
print('pool_Cxy=',test.pool_Cxy)
plot_rickerWVs(test.pool_jk,coeff_list=None, T_i = test.T_i, T_f = test.T_f)
test._plot_init_pool_reconstruction()
test.pool_to_pseudo_tantrix()
#print('pseudo_tantrix=',test.pseudo_tantrix)
#plot_2D_tantrix(test.pseudo_tantrix[0:-1:10]) 
test.pseudo_to_real()
plot_2D_tantrix(test.real_tantrix)  # 已经✅了 M_F ==> tantrix 的映射
test.leakage_check_surgery()
print("log x_leak_maker (jk=)", test.log_leak_maker_x)
print("log y_leak_maker (jk=)", test.log_leak_maker_y)
print(test.norm_check())
#test.log_ptantrix_norm_error
#print('pool_jk=',test.pool_jk)
plot_rickerWVs(test.pool_jk,coeff_list=None, T_i = test.T_i, T_f = test.T_f)
print('pool_jk=',test.pool_jk[-4:])
test.fine_cwv_candidate()
print('pool_jk=',test.pool_jk[-4:])
plot_rickerWVs(test.pool_jk,coeff_list=None, T_i = test.T_i, T_f = test.T_f)
#test.log_ptantrix_norm_error
print(test.norm_check())








T_i=0
T_f=10
my_tantrix = np.array([ [np.cos(1.0*t),  np.sin(1.0*t)]   for t in np.linspace(0,10,1000)])
#plot_2D_curve(random_pseudo_curve[0:-1:50])
my_tantrix = tantrix_normalizer(my_tantrix)
ansatz_tantrix = my_tantrix

plot_2D_tantrix(ansatz_tantrix,plt_title = 'the ORIGINAL ansatz')
test = Curvelet(ansatz = ansatz_tantrix, T_i=T_i, T_f=T_f, tol_leak= 20 )
test.initialize_pool()
test.tantrix_wavelet_optimizer(tol_converge=1, N_iter= 6 )
#print('LAST pool_jk= ', test.pool_jk[-1])
#print('LASTpool_Cxy=', test.pool_Cxy[-1])
print('pool_leak', test.pool_leak)
#print('p_tantrix=', test.pseudo_tantrix)
#print('r_tantrix=', test.real_tantrix)
print('log_worst_window',test.log_worst_window)  
#print('log_ptantrix_norm_error',test.log_ptantrix_norm_error)  
