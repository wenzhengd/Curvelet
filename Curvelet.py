"""
Curve + Wavelet to *adaptively* construct optimal control 
----------------------
caveat: search & locate 🔴 
"""
import numpy as np 
import pywt 
import matplotlib.pyplot as plt 

from scipy.interpolate import interp1d 
from scipy.integrate import simps 
from scipy.integrate import quad, IntegrationWarning 
import scipy.optimize as opt 
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator
 
from scipy.optimize import least_squares, curve_fit
import math

from joblib import Parallel, delayed 
import time 
import warnings 
from functools import wraps 
import time

from plot_Curvelet import plot_2D_curve, plot_rickerWVs, plot_2D_tantrix

from wavelet_factory import wavelet


############################################################
###       Helper functions that operate curve/tantrix   
############################################################


#  Parameterizer to return arc-length curve based on input curve
def curve_parameterization(pseudo_curve, curve_len =1):
    """
    input:  pseudo_curve (\tilde{R}(t))
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
    return np.dstack((x_new,y_new))[0]/arc_length[-1]* curve_len          # make the len(r(t)) == curve_len  !

# Normalize the p_tantrix to make it real_tantrix
def tantrix_normalizer(pseudo_tantrix):
    """
    input:  pseudo_tantrix(\tilde{r}(t))
    output: real_tantrix (r(t)) with normlization done ||r(t)|| == 1 for all t  
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

# [Uselss] Decompose the curve into the WVs by jk_pool
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
            approx = approx + c * wavelet(j=j,k=k, t=t)
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

# [Uselss] Decompose the tantrix into the WVs by jk_pool
def tantrix_fit_old(real_tantrix, jk_pool, cxy_pool, T_i, T_f, cxy_bound_lo = None, cxy_bound_hi = None ):
    """
    Given real_tantrix & pool_jk as wvs and pool_Cxy as init_guess
    Do: least_square() to fit the r_x(t) and r_y(t) seperately
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
            approx = approx + c * wavelet(j=j,k=k, t=t)
        return approx

    # Define the residuals function for least-squares optimization
    def residuals(coeffs, t, x_y_, j_k_):
        # the x_y_ can be either x(t) or y(t) for 2d-tantrix
        return x_y_ - approximation_function(t, coeffs, j_k_)
    
    # % % % Set the scales and translations:   t_k_ = jk_pool
    
    # % % % Initialize the coefficients    initial_cx = cxy_pool[:,0] initial_cy = cxy_pool[:,1]

    # Optimize the coefficients
    init_guess_x,   init_guess_y  = cxy_pool[:,0], cxy_pool[:,1]
    if cxy_bound_lo is None : 
        cxy_bound_lo = [-10]*len(jk_pool)
    if cxy_bound_hi is None : 
        cxy_bound_hi = [ 10]*len(jk_pool)
    # ((internal_debug)) print(init_guess_x ,init_guess_y) #print(init_guess_x.shape,times.shape,target_x.shape, jk_pool.shape) #print(cxy_bound_lo) #print(residuals(init_guess_x, times, target_x, jk_pool))
    optimized_cx = least_squares(residuals, x0= init_guess_x,  bounds=(cxy_bound_lo, cxy_bound_hi), \
                                            args=(times, target_x, jk_pool) ).x
    optimized_cy = least_squares(residuals, x0= init_guess_y,  bounds=(cxy_bound_lo, cxy_bound_hi), \
                                            args=(times, target_y, jk_pool) ).x

    return np.array([optimized_cx, optimized_cy])

# Decompose the tantrix into the WVs by jk_pool
def tantrix_fit(real_tantrix, 
                jk_pool, 
                cxy_pool, 
                leak_pool, 
                T_i, 
                T_f, 
                cxy_bound_lo = None, 
                cxy_bound_hi = None,
                lambda_LEAK= 10.0 ):
    """
    Given real_tantrix & pool_jk as wvs and pool_Cxy as init_guess
    Do: 1) least_square() to fit the r_x(t) and r_y(t) seperately
        2) Lagrangian constraint to reduce the leak along x and y separately
    Return: optimzized_coeff,  that can update pool_Cxy
    -------------------------------------------------------
    Parameters:
    real_tantrix    (times*2 2d-darray): the parameterized real tantrix
    jk_pool         (size*2 2d-array): the (j,k) pairs in the pool
    cxy_pool        (size*2 2d-array): the coeff_Cxy of each (j,k)
    cxy_bound       (size*2 2d-array): the the bound of each cxy
    leak_pool         (size 1d-array): the leak of each (j,k)
    T_i             (float):            inital time of real_tantrix
    T_f             (float):            final  time of real_tantrix
    lambda_         (float):        Lagrangian multiplier   🔴 [user-change?]
    """
    # Define the target function x,y respectively
    times = np.linspace(T_i, T_f,  len(real_tantrix))
    target_x, target_y = real_tantrix[:,0], real_tantrix[:,1] 
    leaks = leak_pool        
    
    # Define the linear combination of given coeffs on WVs
    # This is a scalar function, can be used for either x(t) or y(t)
    # which means the coeffs is either coeff_x or coeff_y
    def approximation_function(t, coeffs, j_k_):
        approx = np.zeros_like(t)
        for c, [j,k] in zip(coeffs, j_k_):
            approx = approx + c * wavelet(j=j,k=k, t=t)
        return approx

    # Define the residuals function for least-squares optimization
    def residuals(coeffs, t, x_y_, j_k_):
        # the x_y_ can be either x(t) or y(t) for 2d-tantrix, coeffs is either coeff_x or coeff_y
        return x_y_ - approximation_function(t, coeffs, j_k_)

    # Global constraint function: for mom_LEAK 
    def constraint(coeffs):
        # calculate the total leak = \sum_{jk} C_jk * leak_jk, coeffs is either coeff_x or coeff_y
        return np.sum(np.multiply(coeffs, leaks), axis=0)

    # Combined objective function with Lagrangian multiplier
    def combined_objective(coeffs, t, x_y_, j_k_, lambda_):
        # coeffs is either coeff_x or coeff_y
        residuals_ = residuals(coeffs, t, x_y_,  j_k_)
        constraint_  = constraint(coeffs)
        return np.concatenate([residuals_, [lambda_ * constraint_]]) 
    
    # Optimize the coefficients
    init_guess_x,   init_guess_y  = cxy_pool[:,0], cxy_pool[:,1]
    if cxy_bound_lo is None : 
        cxy_bound_lo = [-10]*len(jk_pool)
    if cxy_bound_hi is None : 
        cxy_bound_hi = [ 10]*len(jk_pool)
    # ((internal_debug)) print(init_guess_x ,init_guess_y) #print(init_guess_x.shape,times.shape,target_x.shape, jk_pool.shape) #print(cxy_bound_lo) #print(residuals(init_guess_x, times, target_x, jk_pool))
    optimized_cx = least_squares(combined_objective, x0= init_guess_x,  bounds=(cxy_bound_lo, cxy_bound_hi), \
                                                    args=(times, target_x, jk_pool, lambda_LEAK) ).x
    optimized_cy = least_squares(combined_objective, x0= init_guess_y,  bounds=(cxy_bound_lo, cxy_bound_hi), \
                                                    args=(times, target_y, jk_pool, lambda_LEAK) ).x

    return np.array([optimized_cx, optimized_cy])

# Decompose the tantrix into the WVs by jk_pool with COUPLED structure for target_gate
def tantrix_fit_COUPLED(real_tantrix, 
                jk_pool, 
                cxy_pool, 
                leak_pool, 
                T_i, 
                T_f, 
                cxy_bound_lo = None, cxy_bound_hi = None,
                lambda_LEAK = 0.0,
                lambda_GATE = 10.0,
                target_ANGLE = 0.0):
    """
    Given real_tantrix & pool_jk as wvs and pool_Cxy as init_guess
    Do: 1) least_square() to fit the r_x(t) and r_y(t) seperately
        2) Lagrangian constraint to reduce the leak along x and y separately
        3)  💥💥💥💥💥💥💥💥 coupled constraint for target gate. This makes running time very very long
    Return: optimzized_coeff,  that can update pool_Cxy
    -------------------------------------------------------
    Parameters:
    real_tantrix    (times*2 2d-darray): the parameterized real tantrix
    jk_pool         (size*2 2d-array): the (j,k) pairs in the pool
    cxy_pool        (size*2 2d-array): the coeff_Cxy of each (j,k)
    cxy_bound       (size*2 2d-array): the the bound of each cxy
    leak_pool         (size 1d-array): the leak of each (j,k)
    T_i             (float):            inital time of real_tantrix
    T_f             (float):            final  time of real_tantrix
    lambda_         (float):        Lagrangian multiplier
    """
    # Define the target function x,y respectively
    times = np.linspace(T_i, T_f,  len(real_tantrix))
    target_x, target_y = real_tantrix[:,0], real_tantrix[:,1] 
    leaks = leak_pool 
    ctrl_target = np.pi/2  # angle that is target gate       
    
    # Define the linear combination of given coeffs on WVs
    # This is a scalar function, can be used for either x(t) or y(t)
    # coeffs is either coeff_x or coeff_y
    def approximation_function(t, coeffs, j_k_):
        approx = np.zeros_like(t)
        for c, [j,k] in zip(coeffs, j_k_):
            approx = approx + c * wavelet(j=j,k=k, t=t)
        return approx

    # Define the residuals function for least-squares optimization
    def residuals(coeffs, t, x_y_, j_k_):
        # the x_y_ can be either x(t) or y(t) for 2d-tantrix, coeffs is either coeff_x or coeff_y
        return x_y_ - approximation_function(t, coeffs, j_k_)

    #  Constraint function: for mom_LEAK 
    def constraint_LEAK(coeffs):
        # calculate the total leak = \sum_{jk} C_jk * leak_jk, 
        # coeffs is either coeff_x or coeff_y
        # this is constraint along x and y seperately and NOT feel the x-y coupling
        return np.sum(np.multiply(coeffs, leaks), axis=0)
    
    #  Constraint function: for target_unitary 
    def constraint_GATE(coeff_x, coeff_y):
        # calculate r(0) and r(T) angle as target gate
        # this is Global constraint that limits x and y 
        f = lambda idx_1, idx_2:  (coeff_x[idx_1]* coeff_x[idx_2] + coeff_y[idx_1]* coeff_y[idx_2])\
                                  * wavelet(j=jk_pool[idx_1][0], k=jk_pool[idx_1][1],t=T_i) \
                                  * wavelet(j=jk_pool[idx_2][0], k=jk_pool[idx_2][1],t=T_f)
        # Above: the idx_n locate a (j,k) pair                           
        return  np.sum(np.array([f(idx_1, idx_2) for idx_1 in range(len(jk_pool)) \
                                                 for idx_2 in range(len(jk_pool))]))

    # Combined objective function with Lagrangian multiplier
    def combined_objective(coeffs, t, x_y_, j_k_, lambda_LEAK, lambda_GATE):
        """
        now the coeffs include x and y part together and optimized their coupled form 
        now the x_y_ is not either x or y, but them both
        """
        coeffs_x, coeffs_y = np.split(coeffs, 2)
        residuals_x = residuals(coeffs_x, t, x_y_[0], j_k_)
        residuals_y = residuals(coeffs_y, t, x_y_[1], j_k_)
        constraint_leak  = constraint_LEAK(coeffs_x) + constraint_LEAK(coeffs_y)
        constraint_gate  = 100* abs(target_ANGLE-constraint_GATE(coeffs_x, coeffs_y)) # magnify the distance[target, actual]
        return np.concatenate([residuals_x, residuals_y, [lambda_LEAK * constraint_leak], [lambda_GATE * constraint_gate]])
    
    # % % % Set the scales and translations:   t_k_ = jk_pool
    
    # % % % Initialize the coefficients    initial_cx = cxy_pool[:,0] initial_cy = cxy_pool[:,1]

    # Optimize the coefficients
    init_guess_x,   init_guess_y  = cxy_pool[:,0], cxy_pool[:,1]
    if cxy_bound_lo is None : 
        cxy_bound_lo = [-10]*len(jk_pool)* 2
    if cxy_bound_hi is None : 
        cxy_bound_hi = [ 10]*len(jk_pool)* 2

    init_guess_combined = np.concatenate([init_guess_x, init_guess_y])

    optimized_result = least_squares(combined_objective, x0=init_guess_combined, bounds=(cxy_bound_lo, cxy_bound_hi),
                                     args=(times, (target_x, target_y), jk_pool, lambda_LEAK, lambda_GATE))

    optimized_cx, optimized_cy = np.split(optimized_result.x, 2)
    print(' constraint_GATE=  ', constraint_GATE(optimized_cx,optimized_cy), 'target_angle=',target_ANGLE)   # 📣📣📣 : some check that could be useful for 
    return np.array([optimized_cx, optimized_cy])

# Calculate the integration of tantrix: are they closed curve?
# to check the REAL leakage of diff moments
def tantrix_repeat_intg(real_tantrix,T_i,T_f, K):
    """
    Input:  real_tantrix (size=(d*2) 2d array )
         T_i (float), T_f (float),  
         K   (int) (max integration order) if K=0, then the real_tantrix need integration ONCE !!
    Output: different orders of repeated integral as errors     
    """
    times = np.linspace(T_i, T_f, len(real_tantrix))
    x_data, y_data = real_tantrix[:,0], real_tantrix[:,1]
    results = np.zeros((2,K+1))
    # then  Cauchy formula to  calculate the tantrix repeated integral ! 
    # Avoid using high-order integration 
    for m in np.arange(0, K+1):
        """
        This is non-trivial / cf Notes. 
        for 0-th order moments: the integral is int^T_0  [r(t)] dt
        """
        # if m=0, then only need to integrate once for R(T)=R(0) closed
        weighting_function = (times[-1] - times) ** (m) / math.factorial(m) # 💥 see NOTES !!
        integrand_x = x_data * weighting_function
        integrand_y = y_data * weighting_function
        results[0, m] = simps(integrand_x, times)
        results[1, m] = simps(integrand_y, times)
    return results

# calcualte the target unitary from REAL tantrix
def tantrix_to_target_gate(real_tantrix):
    """
    Input: the real tantrix (size=(d*2) 2d array )
    Output: the target_unitary it correspond to 
    ---------------------------------------------------
    Equaiton: the target_unitary (or angle)  
    = \dot{R}(0) \cdot \dot{R}(T) = r(0)\cdot r(T)
    """
    return np.dot(real_tantrix[0], real_tantrix[-1])
     
# Rotate two vectors by the same angle to target their product
def rotate_vectors_to_target_dot_product(u1, u2, target_dot):
    """
    For given two vector u1 u2 and a target_dot_product
    rotate u1 and u2 by same (opposite) angle such that <u1,u2>= target_dot
    """
    # Convert to numpy arrays
    u1, u2 = np.array(u1), np.array(u2)
    
    # Ensure the vectors are unit vectors
    #u1 = u1 / np.linalg.norm(u1)
    #u2 = u2 / np.linalg.norm(u2)
    
    # Calculate the current dot product
    current_dot = np.dot(u1, u2)
    
    # If the current dot product is already equal to the target, return the vectors as they are
    if np.isclose(current_dot, target_dot, atol=1e-6):
        return u1, u2
    
    # Calculate the required angle to achieve the target dot product
    cos_theta = target_dot / (np.linalg.norm(u1) * np.linalg.norm(u2))
    theta = np.arccos(cos_theta) - np.arccos(current_dot)
    
    # Rotation matrix for u1 (counterclockwise)
    rotation_matrix_u1 = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta),  np.cos(theta)]])
    
    # Rotation matrix for u2 (clockwise, hence -theta)
    rotation_matrix_u2 = np.array([[np.cos(-theta), -np.sin(-theta)],
                                   [np.sin(-theta),  np.cos(-theta)]])
    
    # Rotate the vectors
    u1_rotated = np.dot(rotation_matrix_u1, u1)
    u2_rotated = np.dot(rotation_matrix_u2, u2)
    
    return u1_rotated, u2_rotated

# From real R(t) -> real r(t)
def curve_to_tantrix(real_curve):
    # return real_tantrix 
    # Compute the difference between consecutive points
    differences = np.diff(real_curve, axis=0)   
    # Compute the magnitude (norm) of each difference vector
    magnitudes = np.linalg.norm(differences, axis=1, keepdims=True)
    # Normalize the difference vectors to obtain unit tangent vectors
    tantrix = differences / magnitudes
    # Handle the case for the last point (copy the last tangent for simplicity)
    last_tantrix = np.array([tantrix[-1]])
    # Append the last tantrix vector to maintain the same length as the input data
    tantrix = np.vstack([tantrix, last_tantrix])
    real_tantrix = tantrix
    return real_tantrix

# one can use R(t) --> curve_to_tantrix -->  r(t) 
#                  --> tantrix_to_curve -->  R(t) to verify
def tantrix_to_curve(real_tantrix, T_i, T_f):  
    # return real_curve from a real_tantrix
    real_curve = np.zeros ( (len(real_tantrix), 2))
    times = np.linspace(T_i, T_f, len(real_tantrix))
    for i in range(1,len(times)):
        dt = times[i] - times[i-1]
        real_curve[i] = real_curve[i-1] + real_tantrix[i-1] * dt 
    return real_curve

#  [Useless] calculate the curvature of real r(t) 
def tantrix_to_curvature(real_tantrix, T_i, T_f):
    # return the curvature array
    N = len(real_tantrix)
    curvature = np.zeros(N)
    times = np.linspace(T_i,T_f,N)

    for i in range(1, N-1):
        # Finite difference to approximate the derivative of the tangent vector
        dr_dt = (real_tantrix[i+1] - real_tantrix[i-1]) / (times[i+1] - times[i-1]) 
        # Calculate the curvature as the magnitude of this derivative
        curvature[i] = np.linalg.norm(dr_dt)
    
    # For boundary points, use forward and backward differences
    curvature[0] = np.linalg.norm((real_tantrix[1] - real_tantrix[0]) / (times[1] - times[0]))
    curvature[-1] = np.linalg.norm((real_tantrix[-1] - real_tantrix[-2]) / (times[-1] - times[-2]))
    
    return curvature

# [Useless] interpolation method of curve_to_curvature
def curve_interp_curvature(real_curve,T_i,T_f):
    """
    form a real_curve, interp to obtain a SMOOTH expression 
    then calculate curvature
    ----------------------------
    curvature eqution: if X,Y of R
    kappa(t) = [xd(t)*ydd(t) - yd(t)*xdd(t)] / [xd(t)^2+yd(t)^2]^(3/2)
    """
    X ,Y  = real_curve[:, 0], real_curve[:, 1]
    times = np.linspace(T_i,T_f, len(real_curve))
    # Step 1: Interpolate the data
    cs_X = CubicSpline(times, X)
    cs_Y = CubicSpline(times, Y)
    
    # Step 2: Derivatives
    X_t = cs_X(times, 1)    # First derivative of X(t)
    Y_t = cs_Y(times, 1)    # First derivative of Y(t)
    X_tt = cs_X(times, 2)   # Second derivative of X(t)
    Y_tt = cs_Y(times, 2)   # Second derivative of Y(t)
    
    # Step 3: Curvature calculation
    curvature = (X_t * Y_tt - Y_t * X_tt) / (X_t**2 + Y_t**2)**(3/2)
    
    return curvature

# interpolation method of tantrix_to_curvature
def tantrix_interp_curvature(real_tantrix,T_i,T_f):
    """
    from a real_tantrix, iterp to obtain a SMOOTH exprerssion
    then derivative for curvature
    -----------------------------------
    curvature equation: if x,y of r
    kappa(t) = [x(t)*yd(t) - y(t)*xd(t)] / [x(t)^2+y(t)^2]^(3/2)
    """
    x ,y  = real_tantrix[:, 0], real_tantrix[:, 1]
    times = np.linspace(T_i,T_f, len(real_tantrix))

    # Step 1: Interpolate the data
    x_Interp = CubicSpline(times, x)
    y_Interp = CubicSpline(times, y)

    # Step 2: # Evaluate the value and derivative-values
    x_ ,y_ = x_Interp(times), y_Interp(times) 

    x_t = x_Interp(times,1)     # First derivative of x_
    y_t = y_Interp(times,1)     # First derivative of y_

    # Step 3: Curvature calculation
    curvature = (x_ * y_t - y_ * x_t) / (x_**2 + y_**2)**(3/2)

    return curvature    

# rotate the tantrix such that r(t) along y direction
def tantrix_orient_y(real_tantrix, target=np.array([0, 1])):
    """
    the tantrix is (x,y) plane and rotate r(0) -> (0,1)
    """
    # First point of the curve
    x_0, y_0 = real_tantrix[0]
    
    # Calculate the angle between T[0] and the target
    cos_theta = np.dot([x_0, y_0], target)
    theta = np.arccos(cos_theta)
    
    # Determine the direction of rotation using the cross product
    if x_0 * target[1] - y_0 * target[0] < 0:
        theta = -theta  # Rotate clockwise
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate all points in T
    real_tantrix_oriented = np.dot(real_tantrix, rotation_matrix.T)
    
    return real_tantrix_oriented

# smoothen a geometry (curve/tantrix's x, or y or )
def geometry_smoothier(geometry, T_i,T_f):
    """
    input: geometry  1d array
    ooutput: smoothier geometry 1d array that is smooth
    """
    _interp = interp1d(np.linspace(T_i, T_f, len(geometry)), geometry, kind= 'quadratic')
    return _interp( np.linspace(T_i, T_f, len(geometry)) )




# sequence_generation with fixed end_points and mean
def sequence_gen_end_mean_fixed(a,b,mean,N):
    """
    Input: a:    initial point  
           b:    end point 
           mean: the mean of the sequece generated
           N:    length of sequence
    This is a double piece-wise sequence generation with fixed ini/end points a,b, and mean.
    make the point:= (4*mean-a-b)/2 
         the sequence will be one piece in [a, point] and another [point, b] 
         <sequence> ~= mean 
    """
    m = (4*N-a-b)/2
    return np.concatenate([np.linspace(a,m, N//2),   np.linspace(m,b, N-N//2)])

############################################################
###       Ansatz curves: from R(t) to r(t)  
############################################################

# define Gerono curve
def gerono_curve(a = 1, points=1000):
    # Parametric equations for the Gerono curve
    """
    Input: a (float) the stretch along x which determines the angle
    Output: the returned gerono_curve  
    """
    theta = np.linspace(0.5*np.pi, 1.5*np.pi, points)
    x = a * np.cos(theta)
    y = np.sin(theta) * np.cos(theta)
    
    return np.column_stack((x, y))

def anstaz_R_curve(a, T_i=0, T_f=10):
    # return arc-len parameterized curve = real_R_curve
    pseudo_curve = gerono_curve(a)
    real_curve = curve_parameterization(pseudo_curve, curve_len= T_f - T_i)
    return real_curve




# Test follows
#real_curve = anstaz_R_curve(a=1)
#R_x, R_y = real_curve[:, 0], real_curve[:, 1]
#sc=plt.scatter(R_x, R_y, c= np.arange(len(R_x)), cmap='rainbow')
#plt.colorbar(sc, label='Time (t)')
#plt.show()
#real_tantrix = curve_to_tantrix(real_curve)
#plot_2D_tantrix(real_tantrix)




############################################################
###   Tantrix to Wavelet-Manifold optimization formalism   
############################################################


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


class Curvelet():
    """
    Construct the tantrix & wavelet & their mutual map 
    AND optimization
    ------------------------------
    R(t):           error curve -- NOT handeld in this class
    r(t):           real_tantrix     --     handeld in this class
    \tilde{r}(t):   pseudo_tantrix   -- handeld in this class
    wv_pool: wavelet pool that constitute the wv_manifold -- handeld in this class
    ----------------------------------------------------------------------
    Method:  iterative do  [ r(t) --> wv --> \t{r}(t) --> r(t)  ] 
    ----------------------------------------------------------------------
    Optimization
        * Take the anstaz_tantrix r(t) with r(0)=[0,1] // initial condition See Mathematica
        * regress to obtain the M_f manifold details and pool_jk
        * p_tantrix -> r_tantrix
        * Logic enforcer: enforce 1) initial cond  2) target_unit
    """
    def __init__(self, ansatz , T_i, T_f, pool_jk_initial = None, tol_leak= 0.1 , K=3, ini_lm_LEAK= 1.0, ini_lm_GATE = 10.0, seg_chop=10, N=1000) -> None:
        self.ansatz = ansatz                                    # record the ansatz and fix it
        self.real_tantrix = ansatz                              # the normalized tantrix // dynamically changed in optimizer 💥💥💥💥 unless len(ansatz) ==1 
        self.pseudo_tantrix = None                              # the un_normed tantrix \tilde{r}(t) by lin.con. non-leakly (good) WVs \
                                                                #  // dynamically changed in optimizer 
                                                                #  // \tilde~{{x(t1), y(t1)},{{x(t2), y(t2)}......}
        self.real_curve =None
        self.curvature = None                                                        
        self.T_i = T_i                                          # initial time
        self.T_f = T_f                                          # finial time    
        self.L   = T_f-T_i                                      # time span /  len
        self.target_ANGLE = self.get_target_angle(ansatz)       # The target gate's dot product 
        self.seg_chop = seg_chop                                # chop number when check LOCAL normalization error of pseudo_tantrix 🔴 sure seg_chop= 10 is fine?
        self.N  = N                                             # 'pixel' (smallest time slices) of discretization of  ANY tatrix
        self.K =  K                                             # max order of moment evaluatation (K=1 for Ricker-2 and K=3 for Ricker-4)
        self.lm_LEAK = ini_lm_LEAK                    # lagrangian multiplier in the least_square regression constraint: moment leak ()
        self.lm_GATE = ini_lm_GATE                    # lagrangian multiplier in the least_square regression constraint: target gate ()
        self.windows = np.array(np.split(np.linspace(T_i, T_f,N), seg_chop))    # partition time into segments/windows for normalizer check

        self.pool_jk = pool_jk_initial                # the scale-shift (j,k)-pair of WVs kept in the pool // all pass leak test // dynamical change
        self.pool_Cxy = None                          # the coeff of (j,k) kept in the pool // all pass leak test // dynamical change
        self.pool_leak =[]                            #  //// /////record the moment_leak of (j,k) up to order self.K  // NOT change, but will add when pool_jk expand

        self.tol_leak = tol_leak                      # the tol used in WV_leakage 🔴 what is the best value of this? 
        #self.leakage_error = 0                       # the current leakage error
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
        if self.pool_jk is None:
            # The built-in internal way to obtain the M_f manifold
            self.pool_jk =np.concatenate([ np.array([[j,k] for j in np.linspace(0.8, 1.20, 3)  \
                                                       for k in np.linspace(-np.pi/2,  np.pi/2, 3)]) ])  # 🔴  No heuristic way to choose initial pool !!
        #np.concatenate((np.array([[0,0]]),   \
        #                               np.array([[1   , k] for k in np.linspace(-2, 2, 5)]),\
        #                               np.array([[2   , k] for k in np.linspace(-2, 2, 3)])  
        #                               ), axis =0)
        self.window_finest_j =np.array([np.max(self.pool_jk[:,0])  \
                                        for i in np.arange(self.seg_chop)])        # pick the largest j as finest_j for each window
        
        self.pool_Cxy = np.zeros_like(self.pool_jk)
       
        self.pool_leak  = np.zeros(len(self.pool_jk))
        for idx  in range(len(self.pool_jk)):
            # check the leak_jk for each j_k_ pair  [NOT include Cxy at all]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=IntegrationWarning)
                leak_jk = self.mom_integral(idx, self.K)         # leak_jk >=0 , as defined in mom_integral()
                self.pool_leak[idx] = leak_jk
        
        return None

    def get_target_angle(self, ansatz_tantrix):
        """
        get the target_gate_angle from the ansatz_tantrix
        """
        return tantrix_to_target_gate(ansatz_tantrix)

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
        optm_coeff = tantrix_fit(real_tantrix = self.real_tantrix, 
                                   jk_pool = self.pool_jk, 
                                  cxy_pool = self.pool_Cxy,
                                  leak_pool= self.pool_leak,
                                       T_i = self.T_i, 
                                       T_f = self.T_f, 
                              cxy_bound_lo = (-lr_bound if lr_bound is not None else None),
                              cxy_bound_hi = ( lr_bound if lr_bound is not None else None),
                                  lambda_LEAK=  self.lm_LEAK) # 
        #optm_coeff = tantrix_fit_COUPLED( real_tantrix = self.real_tantrix, 
        #                           jk_pool = self.pool_jk, 
        #                          cxy_pool = self.pool_Cxy,
        #                          leak_pool= self.pool_leak,
        #                               T_i = self.T_i, 
        #                               T_f = self.T_f, 
        #                      cxy_bound_lo = (-lr_bound if lr_bound is not None else None),
        #                      cxy_bound_hi = ( lr_bound if lr_bound is not None else None),
        #                      lambda_LEAK  = self.lm_LEAK,
        #                      lambda_GATE  = self.lm_GATE,
        #                      target_ANGLE = self.target_ANGLE) #  
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
                tmp = wavelet(j,k,t)
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
            tmp = wavelet(j,k, times)
            x = x + c_x * tmp
            y = y + c_y * tmp
        self.pseudo_tantrix = np.dstack((x,y))[0]             # udpate the psedo_tantrix using current_pool
        return None
    
    def pseudo_to_real(self):
        """
        Parameterization arc-len to get & update real_tantrix based on the pseudo tantrix
        """
        self.real_tantrix = tantrix_normalizer(pseudo_tantrix = self.pseudo_tantrix)
        return None

    def pseudo_to_real_ANGLE(self):
        """
        Parameterization pseudo tantrix --> real_tantrix
        Then fix the vectors near t=0 &T to fit the gata_angle
        """
        _real_tantrix = np.array(tantrix_normalizer(pseudo_tantrix=self.pseudo_tantrix))
        _target = self.target_ANGLE
    
        # Rotate the r_tantrix vector @ t=0 and t = T
        (_vec_0, _vec_T) = rotate_vectors_to_target_dot_product(u1=_real_tantrix[0,:], 
                                                                u2=_real_tantrix[-1,:], 
                                                                target_dot=_target)
    
        # Smoothly change the r_tantrix vector near t=0 and t = T  ~10%
        t_0_slice = _real_tantrix[0:int(0.1*self.N)]
        t_T_slice = _real_tantrix[int(0.9*self.N):-1]
    
        _real_tantrix[0:int(0.1*self.N), 0] = np.linspace(_vec_0[0], _real_tantrix[int(0.1*self.N), 0], int(0.1*self.N))   # smooth x near t=0
        _real_tantrix[0:int(0.1*self.N), 1] = np.linspace(_vec_0[1], _real_tantrix[int(0.1*self.N), 1], int(0.1*self.N))   # smooth y near t=0
        _real_tantrix[int(0.9*self.N):-1, 0] = np.linspace(_real_tantrix[int(0.9*self.N), 0], _vec_T[0], int(0.1*self.N)-1) # smooth x near t=T
        _real_tantrix[int(0.9*self.N):-1, 1] = np.linspace(_real_tantrix[int(0.9*self.N), 1], _vec_T[1], int(0.1*self.N)-1) # smooth y near t=T
    
        self.real_tantrix = tantrix_normalizer(_real_tantrix) 
        return None

    def pseudo_to_real_swarm_ANGLE(self, _k =0.1):
        """
        (_k):  tantrix head/tail enforcing range :   k<0.1 otherwise the fix is non-local and magnify mom error
        Firstly, Parameterization pseudo tantrix --> real_tantrix
        Then,    Globally rotation all points on r_tantrix such that
        1. gate_angle is obtained
        2. the global rotation does not change the center of points swarm
        """
        _real_tantrix = np.array(tantrix_normalizer(pseudo_tantrix=self.pseudo_tantrix))
        _target = self.target_ANGLE
    
        # Rotate the r_tantrix vector @ t=0 and t = T
        (_vec_0, _vec_T) = rotate_vectors_to_target_dot_product(u1=_real_tantrix[0,:], 
                                                                u2=_real_tantrix[-1,:], 
                                                                target_dot=_target)    
        # Smoothly change the r_tantrix vector near t=0 and t = T  ~10%
        t_0_slice = _real_tantrix[0:int(_k*self.N)]
        t_T_slice = _real_tantrix[int((1-_k)*self.N):]

        _real_tantrix[0:int(_k*self.N), 0] = np.linspace(_vec_0[0], _real_tantrix[int(_k*self.N), 0], int(_k*self.N))   # smooth x near t=0
        _real_tantrix[0:int(_k*self.N), 1] = np.linspace(_vec_0[1], _real_tantrix[int(_k*self.N), 1], int(_k*self.N))   # smooth y near t=0
        _real_tantrix[int((1-_k)*self.N): , 0] = np.linspace(_real_tantrix[int((1-_k)*self.N), 0], _vec_T[0], int(_k*self.N)) # smooth x near t=T
        _real_tantrix[int((1-_k)*self.N): , 1] = np.linspace(_real_tantrix[int((1-_k)*self.N), 1], _vec_T[1], int(_k*self.N)) # smooth y near t=T
    
        self.real_tantrix = tantrix_normalizer(_real_tantrix) 
        return None

    def real_curve_Logic_Enforcer(self, _k=0.05):
        """
        For the r_tantrix, apply the logic enforcer st
        1. r(0) = initial
        2. r(T) --> target_gate unitary
        ---------------------------------
        💥 3. maybe add constraint such that logic is mean-preserving (to make m=0 good) 
        """
        r = self.real_tantrix
        #Logic constraint:
        _2theta=np.arccos(self.target_ANGLE)                    # r(0).r(T) = cos(2theta) = dot == self.target_ANGLE
        _r_0 = np.array([0,1])                                  # r(0) gives y_dir
        _r_T = np.array([np.sin(_2theta), np.cos(_2theta)])     # r(T) gives target_gate
        
        #----------------------------------------
        # logic enforcement & interpolation : 
        #---------------------------------------- 

        #//Simple intepolation
        r[0:int(_k*self.N), 0]     = np.linspace(_r_0[0], r[int(_k*self.N), 0], int(_k*self.N))   # interp x near t=0
        r[0:int(_k*self.N), 1]     = np.linspace(_r_0[1], r[int(_k*self.N), 1], int(_k*self.N))   # interp y near t=0
        r[int((1-_k)*self.N): , 0] = np.linspace(r[int((1-_k)*self.N), 0], _r_T[0], int(_k*self.N)) # interp x near t=T
        r[int((1-_k)*self.N): , 1] = np.linspace(r[int((1-_k)*self.N), 1], _r_T[1], int(_k*self.N)) # interp y near t=T

        #// mean preserving interpolation
        #mean_old_0 = np.mean(r[0:int(_k*self.N)],    axis=0)
        #mean_old_T = np.mean(r[int((1-_k)*self.N):], axis=0)
        #r[0:int(_k*self.N), 0]     = sequence_gen_end_mean_fixed(a=_r_0[0], b=r[int(_k*self.N),0] ,mean=mean_old_0[0], N=int(_k*self.N)) # interp x near t=0
        #r[0:int(_k*self.N), 1]     = sequence_gen_end_mean_fixed(a=_r_0[1], b=r[int(_k*self.N),1] ,mean=mean_old_0[1], N=int(_k*self.N)) # interp x near t=0
        #r[int((1-_k)*self.N): , 0] = sequence_gen_end_mean_fixed(a=r[int((1-_k)*self.N), 0], b=_r_T[0] ,mean=mean_old_T[0], N=int(_k*self.N)) # interp x near t=0
        #r[int((1-_k)*self.N): , 0] = sequence_gen_end_mean_fixed(a=r[int((1-_k)*self.N), 0], b=_r_T[0] ,mean=mean_old_T[0], N=int(_k*self.N)) # interp x near
        
        r[:, 0] = geometry_smoothier(geometry=r[:, 0], T_i=self.T_i, T_f= self.T_f )            # Smooth the tantrix avoid diverge curvature
        r[:, 1] = geometry_smoothier(geometry=r[:, 1], T_i=self.T_i, T_f= self.T_f )            # Smooth the tantrix avoid diverge curvature

        self.real_tantrix = tantrix_normalizer(r)
        return None


    @cache_method           #use cash method to avoid calculate same (jk) for many times
    def mom_integral(self, idx, K):
        """
        calcualte the integral of <psi_jk * m-th moment> where m<=K  # 🔴   weight =40 ? order-scaler = 1/2? 
        """
        weight = 40    # global weight to modulate the mom_leak strength  
        f = lambda m, t, idx:  weight* 1/2**m * t**m * wavelet(self.pool_jk[idx,0], self.pool_jk[idx,1], t) 
        # Calculate the sum of absolute values of integrals
        def calculate_integral(idx, K, T_i, T_f):
            integrals = Parallel(n_jobs=-1)( delayed(lambda m: quad(lambda t: f(m, t, idx), T_i, T_f, limit=100 )[0])(m)  for m in range(0, K+1)  )
            return np.sum(np.abs(integrals))
        return calculate_integral(idx, K, self.T_i, self.T_f)
        
    def leakage_check_surgery(self, K=1, penalty =1.0):
        """
        check whether the containts in pool have large leakage (non_0_moment)
        -----------------------------------------------------------------
        INPUT:  K (int): the highest order moments considered
        -----------------------------------------------------------------
        UPDATE: 1) detects/read the leakage of pool_jk 
                2) surgery/penalize pool_Cxy;
                3) log the self.log_leak_maker
        """
        leak_makers_rnd_x = np.array([])                # the leak_makers in this round 
        leak_makers_rnd_y = leak_makers_rnd_x
        #leak_tot_val_x,leak_tot_val_y = 0,0             # the total leak of all wvs*COeff

        # loop the pool_jk to check leakage of each (j,k) pair
        for idx  in range(len(self.pool_jk)):
            # check the leak_jk for each j_k_ pair  [NOT include Cxy at all]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=IntegrationWarning)
                """
                ------------------------------------------------------
                the leak_jk only read from self.pool_leak
                already calculated in 
                @@@ initialize_pool() or in 
                @@@ fine_cwv_candidate()
                ------------------------------------------------------
                """
                leak_jk = self.pool_leak[idx]
                #leak_jk = self.mom_integral(idx, K)         # leak_jk >=0 , as defined in mom_integral()
                #self.pool_leak = np.append(self.pool_leak, leak_jk)                 # log the leak of each (jk) in jk_pool
                #leak_tot_val_x = leak_tot_val_x + leak_jk * self.pool_Cxy[idx, 0]   # collect the total leakage of all pools
                #leak_tot_val_y = leak_tot_val_y + leak_jk * self.pool_Cxy[idx, 1]   # collect the total leakage of all pools 
            """
            ------------------------------------------------------------------------------- 
            The following suregry is to ** Depress** the C_xy of jk that has large leak !
            it is possible leak(jk) is small, but Cxy(jk) is large, hence needs surgery !
            ------------------------------------------------------------------------------- 
            """    
            # C_xy surgery if above tol_leak             
            if abs(leak_jk * self.pool_Cxy[idx, 0]) > self.tol_leak:             #C-x
                #print("idx", idx, "leak=",leak_jk,"leak*coeff=","coeff=",self.pool_Cxy[idx, 0], leak_jk * self.pool_Cxy[idx, 0])
                leak_makers_rnd_x = np.concatenate((leak_makers_rnd_x , self.pool_jk[idx] ) ) 
                #self.pool_Cxy[idx,0] = penalty * self.tol_leak/leak_jk
                self.pool_Cxy[idx,0] = 0.333* self.pool_Cxy[idx,0]

            if abs(leak_jk * self.pool_Cxy[idx, 1]) > self.tol_leak:             #C-y
                #leak_makers_rnd = np.append(leak_makers_rnd , np.array([self.pool_jk[idx]]) )
                leak_makers_rnd_y = np.concatenate((leak_makers_rnd_y , self.pool_jk[idx] ) ) 
                #self.pool_Cxy[idx,1] = penalty * self.tol_leak/leak_jk
                self.pool_Cxy[idx,1] = 0.333* self.pool_Cxy[idx,1]
        
        if leak_makers_rnd_x.size != 0:
            print('check a leak maker in x: ', leak_makers_rnd_x)
        if leak_makers_rnd_y.size != 0:
            print('check a leak maker in y: ', leak_makers_rnd_y)

        # log the total leakage along x and y 
        #print('Leakage @ check [w/o surgery] is:', np.array([leak_tot_val_x,leak_tot_val_y]))
        #self.log_leakage = np.concatenate( (self.log_leakage, np.array([leak_tot_val_x,leak_tot_val_y])) ) 
            
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

    def fine_cwv_candidate_old(self, partition=10): ## [Useless] old old old old old 
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
        print('hello, now I check the worst window is ', worst, ' and then new j_k_ will be added to M_f. \n')
        self.log_worst_window.append(worst)          # log the bad_window
        
        # need determine how to update js and ks
        if worst ==0 or worst == partition:             #  🔴 the new_j's factor 1.5, 1.4, 1.2 good?
            new_j = self.window_finest_j[worst]*np.array([1.5])
        elif worst ==1 or worst == (partition-1):
            new_j = self.window_finest_j[worst]*np.array([1.4])
        else:
            new_j = self.window_finest_j[worst]*np.array([1.2])

        # update the finest j in this SINGLE window
        # so if same window as worst in next round, new_j will be based on
        # other window not touch if they are not worst     
        self.window_finest_j[worst] = np.max(new_j)   

        DD = (self.T_f - self.T_i )/6
        MM = (self.T_f + self.T_i )/2
        nadia = (self.T_f - self.T_i )/2
        tau = (self.T_f-self.T_i)/partition
        
        new_t = np.array([worst])*tau   # the new t_nadia we want to have
        new_k = np.array([2**-new_j * (MM-2**new_j*MM -new_t + 2**new_j*new_t) /DD  \
                            for new_j, new_t in zip(new_j, new_t)])

        for idx  in range(len(self.pool_jk), ):
            # check the leak_jk for each j_k_ pair  [NOT include Cxy at all]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=IntegrationWarning)
                leak_jk = self.mom_integral(idx, self.K)         # leak_jk >=0 , as defined in mom_integral()
                self.pool_leak[idx] = leak_jk                  
        
        ### update the pool by addding candidate cww
        """
        Expecially, the NEW/fine elements in pool_Cxy should be zero!
        """
        self.pool_jk = np.concatenate((self.pool_jk,  np.array([[j,k] for j,k in zip(new_j, new_k)])),   axis=0)
        self.pool_Cxy= np.concatenate((self.pool_Cxy, np.array([[0,0] for j,k in zip(new_j, new_k)])),   axis=0)

        #calculate & ipdate the leakage
        new_leak = np.zeros(len(new_t))
        self.pool_leak= np.concatenate((self.pool_leak, new_leak),   axis=0) 
        for idx in range(len(self.pool_jk)-len(new_t), len(self.pool_jk)):  # only update the NEW generated jk
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=IntegrationWarning)
                new_leak_jk =  self.mom_integral(idx, self.K)         # leak_jk >=0 , as defined in mom_integral()
                self.pool_leak[idx] = new_leak_jk
        return None
    
    def lagr_multi_update(self):
        """
        update=increase the multiplier 
        """
        self.lm_LEAK = self.lm_LEAK + 20.0 # 🔴 lagra learning_rate = 1.0 is good ? 
        self.lm_GATE = self.lm_GATE + 5.0 # target gate multiplier increase faster than leak
        return None

    def _print_info_(self):
        """
        print the information after each iteration, that is, 
        after p_tantrix ==> r_tantrix is done
        """
        print("~~~~~~~~~~~~~~~~~ Print Info ~~~~~~~~~~~~~~~~~ \n", \
            "Size of pool is ", len(self.pool_jk),
            ".\n leakage eror (C_jk * mom[psi_jk])is", self.leak_error_TOTAL(),
            "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ "  )
        return None

    def get_real_tantrix_error(self):
        """
        The error of the real tantrix @ now 
        """
        _error =  tantrix_repeat_intg(real_tantrix= self.real_tantrix, T_i=self.T_i, T_f = self.T_f, K= self.K)
        return np.round(_error, 4)

    def get_real_curve(self):
        """
        from real_tantrix to real_curve
        """
        self.real_curve = tantrix_to_curve(real_tantrix=self.real_tantrix,  T_i=self.T_i,  T_f=self.T_f)
        return None
    
    def get_smooth_curvature(self):
        """
        from real_tantrix to get curvature
        ----------------------------
        kappa(t) =  |r_t(t)| see tantrix_interp_curvature()
        """
        self.curvature = tantrix_interp_curvature(self.real_tantrix, T_i=self.T_i,  T_f=self.T_f)
        return None

    def leak_error_TOTAL(self):
        """
        calculate the total leakage in the constructed "error-WV"
        """
        if len(self.pool_Cxy) != len(self.pool_leak) :
            raise Exception("the pool_leak and pool_Cxy have different size")
        # return the sum(C_x * leak)     and sum(C_y*leak)
        return  (np.sum(np.multiply(self.pool_Cxy[: , 0],self.pool_leak )), \
                 np.sum(np.multiply(self.pool_Cxy[: , 1],self.pool_leak ))) 
       
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

        # before doing any operation on the tantrix
        plot_2D_tantrix(self.real_tantrix, plt_title ="the ORIGINAL ansatz")
        print("Error of ORIGINAL ansatz: \n", self.get_real_tantrix_error(),
            "\n Target_gate_angle:", tantrix_to_target_gate(self.real_tantrix))              # 📣📣📣 :  Actual ERROR of ansatz & Target gate angle (encoded in anstaz)


        for i in range(N_iter):     # later change to 'do while'
            """
            Interation to update real_tantrix based on pool_WV and update pool_WV based on real_tantrix
            """

            print('====== Now the optimization is ', i ,'-th iteration ======')
            
            # project the real_tantrix to WV space by pool // # it will update pool.C_xy: 
            #  // interally call tantrix_fit()
            dyn_lr_bound = 5 / (1.02**i)     # a dynamical lr_bound shrinks? as loop go 🔴 difficult to increase 
            self.real_tantrix_to_WV(lr_bound= dyn_lr_bound)

            # print the wavelets once the r(t) == >  M_f is done
            plot_rickerWVs(self.pool_jk, coeff_list = None,           T_i= self.T_i,  T_f=self.T_f, plt_title =f"raw  WVs @ {i}-th iter")  # 📣📣📣 : frames of M_F manifold
            plot_rickerWVs(self.pool_jk, coeff_list = self.pool_Cxy,  T_i= self.T_i,  T_f=self.T_f, plt_title =f"coef WVs @ {i}-th iter")  # 📣📣📣 : weighted frames           
            plt.pause(0.001) # Pause briefly to allow the plot to render

            # leakage check for 10 segments // # log the "log_leak_maker" & do surgery by depressing C_xy
            self.leakage_check_surgery()

            # update the self.pseudo_tantrix // # pool.Cxy & pool.jk -> p_tantrix:
            self.pool_to_pseudo_tantrix()   

            # update the self.real_tantrix //   # p_tantrix -> real_tantrix: 
            # It will use tantrix_normalizer() !!!!!                                             
            #self.pseudo_to_real_swarm_ANGLE()             # 💥 : targe_gate angle enforced p->r
            self.pseudo_to_real()                          # 💥 : simple p -> r
            self.real_curve_Logic_Enforcer()               # 💥 : Logic enforcer on r_tantrix // fix t=0 & t=T

            # obtain the curvature from real_tantrix
            self.get_real_curve()
            self.get_smooth_curvature()
            
            # Check the norm & add fine_CWV // from p_tantrix -> finer (jk)
            self.fine_cwv_candidate() 
            #print('size of pool (after add fine) is ', len(self.pool_jk)) 

            # Tantrix_fit to adjust the pool's coeff again //  
            #tantrix_fit(self.real_tantrix, self.pool_jk, self.pool_Cxy, T_i =self.T_i, T_f = self.T_f) 
            
            self._print_info_()   
            plot_2D_tantrix(self.pseudo_tantrix, plt_title =f"pseudo_tantrix @ {i}-th iter",curve_data=None, curvature_data=None)                      # 📣📣📣 : psedudo_tantrix
            plot_2D_tantrix(self.real_tantrix,   plt_title =f"real_tantrix @ {i}-th iter",curve_data=self.real_curve, curvature_data=self.curvature)   # 📣📣📣 : real_tantrix 
            plt.pause(0.001) # Pause briefly to allow the plot to render

            # udpate the lagrange multipler 
            self.lagr_multi_update()  
                 
            i += 1  

            # Print the current repeated tantrix error  
            print( '##################################################################\n',
                    'The real_tantrix give actual (target) unitary is', tantrix_to_target_gate(self.real_tantrix),  # 📣📣📣 : Actual gate_angle of current real_tantrix
                     ' (', self.target_ANGLE,                                                                       # 📣📣📣 : target gate_angle
                     ')\n The real_tantrix error is (row=x,y) (col= k=0,1,.) \n',  
                    self.get_real_tantrix_error(),                                                                  # 📣📣📣 : Actual error of current real_tantrix
                   '\n##################################################################')          

        toc = time.time() 

        print(f"Elapsed time: {toc - tic} seconds") 
        



# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# ========================================================================================================

 
if __name__ == "__main__":
    #print('interactive running will BLOCK smooth execution')
    T_i=0
    T_f=10
    my_tantrix = np.array([ [np.cos(1.0*t),  np.sin(1.0*t)]   for t in np.linspace(0,10,1000)])
    pool_jk_initial = np.concatenate([ np.array([[j,k] for j in np.linspace(0.8, 1.20, 3)  \
                                                       for k in np.linspace(-np.pi/2,  np.pi/2, 3)]) ])
    my_tantrix = tantrix_normalizer(my_tantrix) 
    ansatz_tantrix = my_tantrix 
    #curvelet = Curvelet(ansatz = ansatz_tantrix, T_i=T_i, T_f=T_f, pool_jk_initial = pool_jk_initial, target_ANGLE =0, tol_leak= 20 )
    #curvelet.initialize_pool()
    #curvelet.real_tantrix_to_WV()
    #curvelet.tantrix_wavelet_optimizer(tol_converge=1, N_iter= 5 )

    print(tantrix_to_target_gate(ansatz_tantrix))
    tantrix_fit_COUPLED(real_tantrix= ansatz_tantrix, 
                        jk_pool = pool_jk_initial, 
                        cxy_pool = np.zeros_like(pool_jk_initial) , 
                        leak_pool = np.zeros(len(pool_jk_initial)), 
                        T_i = T_i, 
                        T_f = T_f, 
                        cxy_bound_lo = None, cxy_bound_hi = None,
                        lambda_LEAK = 0.0,
                        lambda_GATE = 100.0,
                        target_ANGLE = 0.1*np.pi)