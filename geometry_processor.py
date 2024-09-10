"""
The processors among geometries: tantrix, curves, curvatures etc. 
"""

from logging import raiseExceptions
from turtle import shape
import numpy as np 
import pywt 
import matplotlib.pyplot as plt 

from scipy.interpolate import interp1d 
from scipy.integrate import simps 
from scipy.integrate import quad, IntegrationWarning 
import scipy.optimize as opt 
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator, make_interp_spline
 
from scipy.optimize import least_squares, curve_fit 
import math

from wavelet_factory import wavelet

from plot_Curvelet import plot_3D_curve, plot_3D_tantrix



#########################################################################
#####             Geometries Used        
#########################################################################  

# Parameterizer to return arc-length curve based on input curve
def curve_parameterization(pseudo_curve, curve_len=1):
    """
    Input:
        pseudo_curve: numpy array of shape (N, D) representing the curve in D dimensions
        curve_len: the desired total length of the returned real_curve
    Output:
        real_curve: numpy array of shape (N, D) with arclength parameterization
                    and total length equal to curve_len
    """
    # Step 1: Calculate the distances between consecutive points
    # Sum the squares of differences across all dimensions
    distances = np.sqrt(np.sum(np.diff(pseudo_curve, axis=0) ** 2, axis=1))

    # Step 2: Compute the cumulative arc length
    arc_length = np.concatenate(([0], np.cumsum(distances)))

    # Step 3: Interpolate each dimension with respect to the arc length
    real_curve = np.zeros_like(pseudo_curve)
    for i in range(pseudo_curve.shape[1]):  # Loop over each dimension
        interp_dim = interp1d(arc_length, pseudo_curve[:, i], kind='linear')
        s_new = np.linspace(0, arc_length[-1], num=len(pseudo_curve))
        real_curve[:, i] = interp_dim(s_new)
    
    # Normalize the curve to have the specified total length
    real_curve = real_curve / arc_length[-1] * curve_len

    return real_curve

# From real R(t) -> real r(t)
def curve_to_tantrix(real_curve):
    """
    Convert a parameterized curve into its unit tangent vectors (tantrix).
    Input:
        real_curve (numpy array of shape (N, D)): Parameterized curve with N points in D dimensions.
    Output:
        real_tantrix (numpy array of shape (N, D)): Unit tangent vectors corresponding to the curve.
    """
    # Compute the difference between consecutive points along the curve
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

# Normalize the p_tantrix to make it real_tantrix
def tantrix_normalizer(pseudo_tantrix):   
    """
    input:  pseudo_tantrix(\tilde{r}(t))
    output: real_tantrix (r(t)) with normlization done ||r(t)|| == 1 for all t  
    """
    # Compute the norm (Euclidean norm) of each vector
    norm_factor = np.linalg.norm(pseudo_tantrix, axis=1)
    
    # Normalize each component by the norm
    real_tantrix = pseudo_tantrix / norm_factor[:, np.newaxis]
    return real_tantrix
        
# Decompose the tantrix into the WVs by jk_pool
def tantrix_fit(real_tantrix,            # 🔴 need to check 
                jk_pool, 
                cxyz_pool, 
                leak_pool, 
                T_i, 
                T_f, 
                cxyz_bound_lo=None, 
                cxyz_bound_hi=None,
                lambda_LEAK=10.0):  
    """
    Generalized version to handle real_tantrix with an arbitrary number of dimensions.
    Do: 1) least_square() to fit r(t) along each dimension separately
        2) Lagrangian constraint to reduce the leak along each dimension separately
    Return: optimized_coeff, that can update pool_Cxyz (or more dimensions)
    -------------------------------------------------------
    Parameters:
    real_tantrix    (times*D d-darray): the parameterized real tantrix with D dimensions
    jk_pool         (size*2 2d-array): the (j,k) pairs in the pool
    cxyz_pool        (size*D d-array): the coeff_Cxy of each (j,k) for each dimension
    cxyz_bound_lo    (size*D d-array or None): the lower bound of each cxy (optional)
    cxyz_bound_hi    (size*D d-array or None): the upper bound of each cxy (optional)
    leak_pool       (size 1d-array): the leak of each (j,k)
    T_i             (float): inital time of real_tantrix
    T_f             (float): final time of real_tantrix
    lambda_LEAK     (float): Lagrangian multiplier
    """
    # Number of dimensions
    D = real_tantrix.shape[1]

    # Time array
    times = np.linspace(T_i, T_f, len(real_tantrix))

    # Initialize the optimized coefficients container
    optimized_coeffs = []

    # Define the linear combination of given coeffs on WVs
    def approximation_function(t, coeffs, j_k_):
        approx = np.zeros_like(t)
        for c, (j, k) in zip(coeffs, j_k_):
            approx = approx + c * wavelet(j=j, k=k, t=t)
        return approx

    # Define the residuals function for least-squares optimization
    def residuals(coeffs, t, xyz_, j_k_):
        return xyz_ - approximation_function(t, coeffs, j_k_)

    # Global constraint function: for mom_LEAK 
    def constraint(coeffs):
        return np.sum(np.multiply(coeffs, leak_pool), axis=0)

    # Combined objective function with Lagrangian multiplier
    def combined_objective(coeffs, t, xyz_, j_k_, lambda_):
        residuals_ = residuals(coeffs, t, xyz_, j_k_)
        constraint_ = constraint(coeffs)
        return np.concatenate([residuals_, [lambda_ * constraint_]])

    # Set default bounds if none provided
    if cxy_bound_lo is None:
        cxy_bound_lo = [-10] * len(jk_pool)
    if cxy_bound_hi is None:
        cxy_bound_hi = [10] * len(jk_pool)

    # Optimize for each dimension
    for dim in range(D):
        init_guess = cxyz_pool[:, dim]
        target_dim = real_tantrix[:, dim]

        optimized_c = least_squares(combined_objective, x0=init_guess, bounds=(cxy_bound_lo, cxy_bound_hi),
                                    args=(times, target_dim, jk_pool, lambda_LEAK)).x
        
        optimized_coeffs.append(optimized_c)
    
    return np.array(optimized_coeffs)

def tantrix_fit_COUPLED(real_tantrix,    # 🔴 need test 
                        jk_pool, 
                        cxyz_pool, 
                        leak_pool, 
                        T_i, 
                        T_f, 
                        cxy_bound_lo=None, 
                        cxy_bound_hi=None,
                        lambda_LEAK=0.0,
                        lambda_GATE=10.0,
                        target_ANGLE=0.0):  
    """
    Generalized version to handle real_tantrix with an arbitrary number of dimensions.
    Given real_tantrix & pool_jk as wvs and pool_Cxy as init_guess
    Do: 1) least_square() to fit r(t) along each dimension separately
        2) Lagrangian constraint to reduce the leak along each dimension separately
        3) Coupled constraint for target gate to enforce control over multiple dimensions.
    Return: optimized_coeff, that can update pool_Cxyz (or more dimensions)
    -------------------------------------------------------
    Parameters:
    real_tantrix    (times*D d-darray): the parameterized real tantrix with D dimensions
    jk_pool         (size*2 2d-array): the (j,k) pairs in the pool
    cxyz_pool        (size*D d-array): the coeff_Cxy of each (j,k) for each dimension
    cxyz_bound_lo    (size*D d-array or None): the lower bound of each cxy (optional)
    cxyz_bound_hi    (size*D d-array or None): the upper bound of each cxy (optional)
    leak_pool       (size 1d-array): the leak of each (j,k)
    T_i             (float): inital time of real_tantrix
    T_f             (float): final time of real_tantrix
    lambda_LEAK     (float): Lagrangian multiplier for leak constraint
    lambda_GATE     (float): Lagrangian multiplier for gate constraint
    target_ANGLE    (float): Target angle for the gate constraint
    """
    # Number of dimensions
    D = real_tantrix.shape[1]

    # Time array
    times = np.linspace(T_i, T_f, len(real_tantrix))

    # Initialize the optimized coefficients container
    optimized_coeffs = []

    # Define the linear combination of given coeffs on WVs
    def approximation_function(t, coeffs, j_k_):
        approx = np.zeros_like(t)
        for c, (j, k) in zip(coeffs, j_k_):
            approx += c * wavelet(j=j, k=k, t=t)
        return approx

    # Define the residuals function for least-squares optimization
    def residuals(coeffs, t, xyz_, j_k_):
        return xyz_ - approximation_function(t, coeffs, j_k_)

    # Constraint function: for LEAK
    def constraint_LEAK(coeffs):
        return np.sum(np.multiply(coeffs, leak_pool), axis=0)
    
    # Constraint function: for GATE
    def constraint_GATE(coeffs_list):
        # Calculate the gate angle based on the combined coefficients
        f = lambda idx_1, idx_2: np.sum(
            [coeffs[idx_1] * coeffs[idx_2] * wavelet(j=jk_pool[idx_1][0], k=jk_pool[idx_1][1], t=T_i) *
             wavelet(j=jk_pool[idx_2][0], k=jk_pool[idx_2][1], t=T_f) for coeffs in coeffs_list]
        )
        return np.sum([f(idx_1, idx_2) for idx_1 in range(len(jk_pool)) for idx_2 in range(len(jk_pool))])

    # Combined objective function with Lagrangian multipliers
    def combined_objective(combined_coeffs, t, targets, j_k_, lambda_LEAK, lambda_GATE):
        coeffs_list = np.split(combined_coeffs, D)
        residuals_all = np.concatenate([residuals(coeffs_list[d], t, targets[d], j_k_) for d in range(D)])
        constraint_leak = np.sum([constraint_LEAK(coeffs_list[d]) for d in range(D)])
        constraint_gate = 100 * abs(target_ANGLE - constraint_GATE(coeffs_list))  # Magnify the distance [target, actual]
        return np.concatenate([residuals_all, [lambda_LEAK * constraint_leak], [lambda_GATE * constraint_gate]])

    # Set default bounds if none provided
    if cxy_bound_lo is None:
        cxy_bound_lo = [-10] * len(jk_pool) * D
    if cxy_bound_hi is None:
        cxy_bound_hi = [10] * len(jk_pool) * D

    # Initialize the combined coefficients for all dimensions
    init_guess_combined = np.concatenate([cxyz_pool[:, d] for d in range(D)])

    # Optimize the combined coefficients
    optimized_result = least_squares(combined_objective, x0=init_guess_combined, bounds=(cxy_bound_lo, cxy_bound_hi),
                                     args=(times, [real_tantrix[:, d] for d in range(D)], jk_pool, lambda_LEAK, lambda_GATE))

    # Split the optimized result into separate components for each dimension
    optimized_coeffs = np.split(optimized_result.x, D)

    return np.array(optimized_coeffs)


# Calculate the integration of tantrix: are they closed curve?
# to check the REAL leakage of diff moments
def tantrix_repeat_intg(real_tantrix, T_i, T_f, K):
    """
    Input:
        real_tantrix (size=(d, D) array): The parameterized real tantrix where D is the number of dimensions
        T_i (float): Initial time
        T_f (float): Final time
        K (int): Maximum integration order (if K=0, then the real_tantrix needs integration ONCE)
    Output:
        results (array of shape (D, K+1)): Different orders of repeated integrals as errors
    """
    times = np.linspace(T_i, T_f, len(real_tantrix))
    D = real_tantrix.shape[1]  # Number of dimensions in the tantrix
    results = np.zeros((D, K+1))
    # then  Cauchy formula to  calculate the tantrix repeated integral ! 
    # Avoid using high-order integration 
    for m in range(K+1):
        """
        This is non-trivial / cf Notes. 
        for 0-th order moments: the integral is int^T_0  [r(t)] dt
        """        
        weighting_function = (times[-1] - times) ** m / math.factorial(m)  # 🔴 see NOTES !!
        for d in range(D):
            integrand = real_tantrix[:, d] * weighting_function
            results[d, m] = simps(integrand, times)
    
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
 
# one can use R(t) --> curve_to_tantrix -->  r(t) 
#                  --> tantrix_to_curve -->  R(t) to verify
def tantrix_to_curve(real_tantrix, T_i, T_f):
    """
    Reconstruct a parameterized curve from its tangent vectors (tantrix).
    Input:
        real_tantrix (numpy array of shape (N, D)): Array of tangent vectors with N points in D dimensions.
        T_i (float): Initial time.
        T_f (float): Final time.
    Output:
        real_curve (numpy array of shape (N, D)): Reconstructed parameterized curve.
    """
    N, D = real_tantrix.shape           # Get the number of points and dimensions
    real_curve = np.zeros((N, D))       # Initialize the curve array
    times = np.linspace(T_i, T_f, N)    # Time array
    
    for i in range(1, N):
        dt = times[i] - times[i-1]
        real_curve[i] = real_curve[i-1] + real_tantrix[i-1] * dt
    
    return real_curve

# interpolation method of tantrix_to_curvature
def tantrix_interp_curvature(real_tantrix,T_i,T_f): # 🔴🔴 不达标啊 ！！！ 
    """
    from a real_tantrix, iterp to obtain a SMOOTH exprerssion
    then derivative for curvature
    -----------------------------------
    2D:
        curvature equation: if x,y of r
        kappa(t) = [x(t)*yd(t) - y(t)*xd(t)] / [x(t)^2+y(t)^2]^(3/2)
    3D:
        kappa(t) =  ||R'(t) \cross R''(t)|| / ||R'(t)||^3
                 =  ||r(t) \cross r'(t)|| 
        tau (t)  =  (R'(t)\cross R''(t))\dot R'''(t) / ||R'(t)\cross R''(t)||^2
                 =  [r(t)\cross r'(t)] \dot r''(t) / ||r(t)\cross r'(t)||^2      
    """
    if real_tantrix.shape[1]==2:
        """
        2D
        """
        x ,y  = real_tantrix[:, 0], real_tantrix[:, 1]
        times = np.linspace(T_i,T_f, len(real_tantrix))

        # Step 1: Interpolate the data
        x_Interp = make_interp_spline(times, x, k=3)
        y_Interp = make_interp_spline(times, y, k=3)

        # Step 2: # Evaluate the value and derivative-values
        x_ ,y_ = x_Interp(times), y_Interp(times) 

        x_t = x_Interp(times,1)     # First derivative of x_
        y_t = y_Interp(times,1)     # First derivative of y_

        # Step 3: Curvature calculation
        curvature = (x_ * y_t - y_ * x_t) / (x_**2 + y_**2)**(3/2)
        return curvature

    elif real_tantrix.shape[1] ==3:
        """
        3D
        """
        times = np.linspace(T_i, T_f, len(real_tantrix))
    
        # Extract the x, y, z components of the tantrix
        x, y, z = real_tantrix[:, 0], real_tantrix[:, 1], real_tantrix[:, 2]

        # Step 1: Interpolate the data for x, y, and z
        x_Interp = make_interp_spline(times, x, k=3)
        y_Interp = make_interp_spline(times, y, k=3)
        z_Interp = make_interp_spline(times, z, k=3)

        # Step 2: Evaluate the first, second, and third derivatives
        x_  = x_Interp(times)
        y_  = y_Interp(times)
        z_  = z_Interp(times)

        x_t = x_Interp(times, 1)     # First derivative of x_
        y_t = y_Interp(times, 1)     # First derivative of y_
        z_t = z_Interp(times, 1)     # First derivative of z_
    
        x_tt = x_Interp(times, 2)    # Second derivative of x_
        y_tt = y_Interp(times, 2)    # Second derivative of y_
        z_tt = z_Interp(times, 2)    # Second derivative of z_


        # Step 3: Compute the curvature using the formula for 3D curves

        numerator_kappa= np.sqrt((x_ * z_t - z_ * y_t) ** 2 +
                                 (z_ * x_t - x_ * z_t) ** 2 +
                                 (x_ * y_t - y_ * x_t) ** 2)
        denominator_kappa = ((y_ * z_t - z_ * y_t) ** 2 + \
                             (z_ * x_t - x_ * z_t) ** 2 + \
                             (x_ * y_t - y_ * x_t) ** 2)                    
        curvature = numerator_kappa / denominator_kappa

        # Step 4: Compute the torsion using the formula for 3D curves
        numerator_tau = (x_ * (y_t * z_tt - z_t * y_tt) +
                         y_ * (z_t * x_tt - x_t * z_tt) +
                         z_ * (x_t * y_tt - y_t * x_tt))

        denominator_tau = (y_ * z_t - z_ * y_t) ** 2 + \
                          (z_ * x_t - x_ * z_t) ** 2 + \
                          (x_ * y_t - y_ * x_t) ** 2

        torsion = numerator_tau / denominator_tau

        return curvature, torsion
    else:
        raise Exception(f'The geometry dimension {real_tantrix.shape[1]} is not supported.')


# rotate the tantrix such that r(0) along target direction
def tantrix_orient_target(real_tantrix, target= np.array([0, 1,0])):
    """
    Reorients a 3D tantrix curve such that the first tangent vector aligns with the target vector.
    
    Parameters:
    real_tantrix: (N, 3) array-like
        A 3D curve represented as an array of shape (N, 3).
    target: array-like, default [0, 0, 1]
        The target vector to align the first tangent vector with.
    
    Returns:
        real_tantrix_oriented: (N, 3) numpy array
        The rotated 3D curve.
    """
    if real_tantrix.shape[1] == 2:

        target = target[0:-1]    # dimension protector 
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


    elif real_tantrix.shape[1]==3:
        # First point of the curve
        x_0, y_0, z_0 = real_tantrix[0]
        
        # Initial vector
        initial_vector = np.array([x_0, y_0, z_0])
        
        # Normalize the initial vector and the target vector
        initial_vector_norm = initial_vector / np.linalg.norm(initial_vector)
        target_norm = target / np.linalg.norm(target)
        
        # Calculate the rotation axis using the cross product
        rotation_axis = np.cross(initial_vector_norm, target_norm)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        # If the vectors are parallel (rotation axis norm is zero), no rotation is needed
        if rotation_axis_norm == 0:
            return real_tantrix  # The curve is already aligned
        
        # Normalize the rotation axis
        rotation_axis = rotation_axis / rotation_axis_norm
        
        # Calculate the rotation angle using the dot product
        cos_theta = np.dot(initial_vector_norm, target_norm)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical inaccuracies
        
        # Construct the rotation matrix using the axis-angle formula (Rodrigues' rotation formula)
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        rotation_matrix = (
            np.eye(3) +
            np.sin(theta) * K +
            (1 - np.cos(theta)) * np.dot(K, K)
        )
        
        # Rotate all points in the curve
        real_tantrix_oriented = np.dot(real_tantrix, rotation_matrix.T)
        
        return real_tantrix_oriented    
    else:
        raiseExceptions(f'Current setup only support up to 3D')        
    

# smoothen a geometry (curve/tantrix's x, or y or )
def geometry_smoothier(geometry, T_i,T_f):
    """
    input: geometry  **(1d)** array
    ooutput: smoothier geometry 1d array that is smooth
    """
    _interp = interp1d(np.linspace(T_i, T_f, len(geometry)), geometry, kind= 'quadratic')
    return _interp( np.linspace(T_i, T_f, len(geometry)) )



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
    z = a * np.sin(0.8*theta)
    
    return np.column_stack((x, y, z))

def anstaz_R_curve(a, T_i=0, T_f=10):
    # return arc-len parameterized curve = real_R_curve
    pseudo_curve = gerono_curve(a)
    real_curve = curve_parameterization(pseudo_curve, curve_len= T_f - T_i)
    return real_curve


#########################################################################
#####             USELESS    NOW        
#########################################################################  
"""
Originall from Curvelet.py to handle only 2D geometries
"""
#  [Useless]
def get_curve_len(real_curve):
    """
    USELESS
    -------------
    """
    pass                            # useless for now [in curve_parameterization() it handles]
    return None

# [Uselss] Decompose the curve into the WVs by jk_pool
def curve_fit(real_curve, jk_pool, cxy_pool, T_i, T_f, cxy_bound_lo = None, cxy_bound_hi = None ):
    """
    USELESS
    """
    pass 

# [Uselss] Decompose the tantrix into the WVs by jk_pool
def tantrix_fit_old(real_tantrix, jk_pool, cxy_pool, T_i, T_f, cxy_bound_lo = None, cxy_bound_hi = None ):
    """
    USELESS
    """
    pass
    
# [Useless] Rotate two vectors by the same angle to target their product
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

# [Useless] interpolation method of curve_to_curvature
def curve_interp_curvature(real_curve,T_i,T_f):
    """
    USELESS
    """
    pass

# [Useless]calculate the curvature of real r(t) 
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



if __name__ == "__main__":
    #print('interactive running will BLOCK smooth execution')
    T_i = 0
    T_f = 10
    test_curve= anstaz_R_curve(a=1, T_i=0, T_f =20)
    #test_curve= np.array([ [2*np.cos(0.75*t),  2*np.sin(0.75*t), 0.2*t]   for t in np.linspace(0,10,1000)])
    plot_3D_curve(test_curve)
    test_curve = curve_parameterization(pseudo_curve=test_curve, curve_len=T_f-T_i)
    plot_3D_curve(test_curve)
    test_tantrix = curve_to_tantrix(test_curve)
    test_tantrix = tantrix_orient_target(test_tantrix)
    #print('tantrix norma', np.linalg.norm(test_tantrix,axis=1)) # All one ! good 
    plot_3D_tantrix(test_tantrix)
    #print(tantrix_repeat_intg(real_tantrix= test_tantrix, T_i=T_i, T_f=T_f, K=4))

    plot_3D_curve(tantrix_to_curve(real_tantrix =test_tantrix, T_i=T_i, T_f=T_f), plt_title='r->R')    

    test_curvatureS = tantrix_interp_curvature(real_tantrix = test_tantrix, T_i=T_i, T_f=T_f )
    #print(test_curvatureS[0].shape, test_curvatureS[1].shape )

    plot_3D_tantrix(tantrix_data = test_tantrix, 
                    plt_title = '3D tantrix',  
                    curvatureS_data = test_curvatureS)
