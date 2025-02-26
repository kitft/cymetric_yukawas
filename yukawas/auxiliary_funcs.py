import gc
import tensorflow as tf
import time
from datetime import datetime
import numpy as np

def delete_all_dicts_except(except_dict_name):
    """
    Deletes all dictionary objects from the global namespace except the specified one.

    Parameters:
    except_dict_name (str): The name of the dictionary to exclude from deletion.
    """

    # Identify all dictionary variables except the one specified
    dicts_to_delete = [
        var for var, obj in globals().items()
        if isinstance(obj, dict) and var != except_dict_name
    ]

    # Delete all identified dictionary variables
    for var in dicts_to_delete:
        del globals()[var]

    # Optionally, you can clear memory by calling garbage collector
    gc.collect()


def get_coefficients_m13(free_coefficient):
   x = free_coefficient
   coefficients=np.array([1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, \
0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1]) +\
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
   return coefficients




def batch_process_helper_func(func, args, batch_indices=(0,), batch_size=10000, compile_func=False):
    # Optionally compile the function for better performance
    if compile_func:
        func = tf.function(func)
        
    # Determine the number of batches based on the first batched argument
    num_batches = tf.math.ceil(tf.shape(args[batch_indices[0]])[0] / batch_size)
    results_list = []

    for i in range(tf.cast(num_batches, tf.int32)):
        print(i)
        start_idx = i * batch_size
        end_idx = tf.minimum((i + 1) * batch_size, tf.shape(args[batch_indices[0]])[0])
        
        # Create batched arguments
        batched_args = list(args)
        for idx in batch_indices:
            batched_args[idx] = args[idx][start_idx:end_idx]
        
        # Call the function with batched and static arguments
        batch_results = func(*batched_args)
        results_list.append(batch_results)

    return tf.concat(results_list, axis=0)


def determine_training_flags(start_from='phi'):
    training_flags = {
        'phi': True,
        'LB1': True,  # 02m20
        'LB2': True,  # 001m3
        'LB3': True,  # 0m213
        'vH': True,
        'vQ3': True,
        'vU3': True,
        'vQ1': True,
        'vQ2': True,
        'vU1': True,
        'vU2': True,
        'end': True
    }
    
    if start_from not in training_flags:
        raise ValueError(f"Error: '{start_from}' is not a valid starting point.")
    
    start_index = list(training_flags.keys()).index(start_from)
    print(f"Starting training from: {start_from}")
    
    for key in list(training_flags.keys())[:start_index]:
        training_flags[key] = False
        print(f"Skipping: {key}")
    
    return training_flags
import sys
    
def print_memory_usage(start_time_of_process=None, name = None):
    """
    Prints the current memory usage of the Python process.
    """
    import os
    import psutil
    
    # Get the current process
    process = psutil.Process(os.getpid())
    
    # Get memory info in bytes
    memory_info = process.memory_info()
    
    # Convert to more readable format (MB)
    memory_usage_mb = memory_info.rss / 1024 / 1024
    
    
    # Additional memory details
    virtual_memory = psutil.virtual_memory()
    if start_time_of_process is not None:
        
        start_time_str = datetime.fromtimestamp(start_time_of_process).strftime('%H:%M:%S')
        printafter_time = time.time() - start_time_of_process
    else:
        start_time_str = "unknown"
        printafter_time = 0.0
    
    print(f"Current memory usage of process {os.getpid()} for {name} started at {start_time_str}: {memory_usage_mb:.2f} MB",f", System memory: {virtual_memory.total / (1024 * 1024 * 1024):.2f} GB total, "
      f"{virtual_memory.available / (1024 * 1024 * 1024):.2f} GB available. (printed after {printafter_time:.2f} sec)")



# Function to calculate effective sample size from auxiliary weights
def effective_sample_size(weights):
    """Calculate effective sample size from importance sampling weights."""
    weights_sum = tf.reduce_sum(weights)
    weights_squared_sum = tf.reduce_sum(tf.square(weights))
    if weights_squared_sum == 0:
        return len(weights)  # If all weights are 0, return original sample size
    return (weights_sum ** 2) / weights_squared_sum

# # Function to calculate standard error for weighted mean of complex values
# def weighted_mean_and_standard_error(values, weights):
#     """
#     Calculate standard error for weighted mean of values.
#     For complex values, we treat real and imaginary parts separately.
    
#     Returns:
#         standard error (real or complex), effective sample size
#     """
#     # Assert that weights are real
#     tf.debugging.assert_all_finite(weights, "Weights must be real values")
    
#     # Get the weighted mean
#     weighted_mean = tf.reduce_mean(tf.cast(weights, values.dtype) * values)
    
#     # Calculate the effective sample size
#     n_eff = effective_sample_size(weights)
    
#     # Calculate variance for real part
#     real_values = tf.math.real(values)
#     real_mean = tf.math.real(weighted_mean)
#     real_squared_deviation = tf.square(real_values - real_mean)
#     real_weighted_variance = tf.reduce_sum(weights * real_squared_deviation) / tf.reduce_sum(weights)
    
#     # Standard error for real part
#     real_se = tf.sqrt(real_weighted_variance / n_eff)
    
#     # Check if values are complex
#     if hasattr(values, 'dtype') and 'complex' in str(values.dtype) and np.any(np.imag(np.array(values)) != 0):
#         # Calculate variance for imaginary part
#         imag_values = tf.math.imag(values)
#         imag_mean = tf.math.imag(weighted_mean)
#         imag_squared_deviation = tf.square(imag_values - imag_mean)
#         imag_weighted_variance = tf.reduce_sum(weights * imag_squared_deviation) / tf.reduce_sum(weights)
        
#         # Standard error for imaginary part
#         imag_se = tf.sqrt(imag_weighted_variance / n_eff)
        
#         # Return complex standard error
#         return tf.complex(real_se, imag_se), n_eff
#     else:
#         # Return real standard error
#         return real_se, n_eff

# Function to propagate errors through matrix operations for complex values
def propagate_errors_to_matrix(matrix, matrix_errors):
    """
    Propagate errors through matrix operations for complex matrices.
    
    For complex matrices, we track real and imaginary part errors separately.
    """
    return matrix_errors  # For complex matrices, we already have separate real and imag errors

# Function to propagate errors to SVD (singular values) for complex matrices
def propagate_errors_to_singular_values(matrix, matrix_errors):
    """
    Accurately propagate errors from matrix elements to singular values using perturbation theory.
    Works for both real and complex matrices.

    Parameters:
    matrix (np.ndarray): Input matrix (real or complex)
    matrix_errors (np.ndarray): Matrix of same shape containing standard errors for each element

    Returns:
    np.ndarray: Standard errors for each singular value
    """
    # Perform SVD on the input matrix
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)

    # Initialize array to store singular value errors
    s_errors = np.zeros_like(s)

    # For each singular value, calculate its error
    for i in range(len(s)):
        # Get corresponding singular vectors
        ui = u[:, i].reshape(-1, 1)  # Left singular vector (as column)
        vi = vh[i, :].reshape(1, -1)  # Right singular vector (as row)

        # Sensitivity matrix through outer product
        sensitivity = np.abs(ui @ vi)  # Take absolute value for complex case

        # Error propagation: variance is sum of squared sensitivities times squared errors
        variance = np.sum((sensitivity ** 2) * (matrix_errors ** 2))

        # Standard error is square root of variance
        s_errors[i] = np.sqrt(variance)

    return s_errors

def verify_by_monte_carlo(matrix, matrix_errors, n_samples=10000):
    """
    Verify error propagation results using Monte Carlo simulation.
    Works for both real and complex matrices.

    Parameters:
    matrix (np.ndarray): Input matrix (real or complex)
    matrix_errors (np.ndarray): Matrix of standard errors
    n_samples (int): Number of Monte Carlo samples

    Returns:
    np.ndarray: Standard errors for each singular value estimated by Monte Carlo
    """
    # Storage for all singular values from Monte Carlo trials
    all_singular_values = []
    
    # Check if matrix is complex
    is_complex = np.iscomplexobj(matrix)

    # Generate perturbed matrices and compute their SVDs
    for _ in range(n_samples):
        # Create random perturbation according to error distribution
        if is_complex:
            # For complex matrices, perturb real and imaginary parts separately
            real_perturbations = np.random.normal(0, np.real(matrix_errors))
            imag_perturbations = np.random.normal(0, np.imag(matrix_errors))
            perturbations = real_perturbations + 1j * imag_perturbations
        else:
            perturbations = np.random.normal(0, matrix_errors)
            
        perturbed_matrix = matrix + perturbations

        # Compute singular values for perturbed matrix
        s_perturbed = np.linalg.svd(perturbed_matrix, compute_uv=False)
        all_singular_values.append(s_perturbed)

    # Calculate standard deviation across all samples
    s_errors_mc = np.std(np.array(all_singular_values), axis=0)
    return s_errors_mc

def test_error_propagation():
    """Test both implementations and compare results for real and complex matrices."""
    # Example real matrix
    real_matrix = np.array([
        [4.0, 2.0, 1.0],
        [3.0, 1.0, 0.5],
        [0.5, 1.0, 3.0]
    ])
    
    # Example complex matrix
    complex_matrix = real_matrix + 1j * np.array([
        [0.5, 1.0, 0.2],
        [0.3, 0.7, 0.1],
        [0.2, 0.4, 0.8]
    ])

    # Test with real matrix
    print("Testing with real matrix:")
    matrix_errors_real = 0.05 * np.abs(real_matrix)
    singular_values_real = np.linalg.svd(real_matrix, compute_uv=False)
    print("Original singular values:", singular_values_real)
    
    s_errors_analytical_real = propagate_errors_to_singular_values(real_matrix, matrix_errors_real)
    print("Analytical errors:", s_errors_analytical_real)
    
    s_errors_mc_real = verify_by_monte_carlo(real_matrix, matrix_errors_real)
    print("Monte Carlo errors:", s_errors_mc_real)
    print("Ratio (Analytical/MC):", s_errors_analytical_real / s_errors_mc_real)
    
    # Test with complex matrix
    print("\nTesting with complex matrix:")
    matrix_errors_complex = 0.05 * np.abs(complex_matrix)
    singular_values_complex = np.linalg.svd(complex_matrix, compute_uv=False)
    print("Original singular values:", singular_values_complex)
    
    s_errors_analytical_complex = propagate_errors_to_singular_values(complex_matrix, matrix_errors_complex)
    print("Analytical errors:", s_errors_analytical_complex)
    
    s_errors_mc_complex = verify_by_monte_carlo(complex_matrix, matrix_errors_complex)
    print("Monte Carlo errors:", s_errors_mc_complex)
    print("Ratio (Analytical/MC):", s_errors_analytical_complex / s_errors_mc_complex)
    
    # Check if ratios are close to 1 for both cases
    tolerance = 0.2  # 20% tolerance
    
    ratios_real = s_errors_analytical_real / s_errors_mc_real
    ratios_complex = s_errors_analytical_complex / s_errors_mc_complex
    
    if not np.all(np.abs(ratios_real - 1) < tolerance):
        print(f"Warning: Real matrix error propagation ratios outside tolerance: {ratios_real}")
    else:
        print("Real matrix error propagation validation passed!")
        
    if not np.all(np.abs(ratios_complex - 1) < tolerance):
        print(f"Warning: Complex matrix error propagation ratios outside tolerance: {ratios_complex}")
    else:
        print("Complex matrix error propagation validation passed!")

    # Function to propagate errors through matrix square root and inversion
def propagate_errors_through_normalization(matrix, matrix_errors):
    """
    Propagate errors through matrix square root and inversion operations.
    Uses analytical error propagation formulas.
    """
    # For a 1x1 matrix, the error propagation is straightforward
    if matrix.shape == (1, 1):
        # For sqrt(x)^-1, the derivative is -0.5 * x^(-3/2)
        derivative = -0.5 * np.power(matrix[0, 0], -1.5)
        return np.abs(derivative) * matrix_errors
    
    # For larger matrices, use SVD-based approac
    u, s, vh = np.linalg.svd(matrix)
    sqrt_matrix = scipy.linalg.sqrtm(matrix).astype(complex)
    inv_sqrt_matrix = np.linalg.inv(sqrt_matrix)
    
    # Calculate how errors in matrix elements affect singular values
    s_errors = np.zeros_like(s)
    for i in range(len(s)):
        for j in range(matrix.shape[0]):
            for k in range(matrix.shape[1]):
                # Partial derivative of singular value with respect to matrix element
                # ∂s_i/∂M_jk = u_ji * vh_ik
                s_errors[i] += (u[j, i] * vh[i, k] * matrix_errors[j, k])**2
        s_errors[i] = np.sqrt(s_errors[i])
    
    # Calculate errors for inv(sqrt(s))
    s_sqrt_inv_errors = 0.5 * s_errors * np.power(s, -1.5)
    
    # Reconstruct the error matrix
    error_matrix = np.zeros_like(inv_sqrt_matrix, dtype=complex)
    for i in range(len(s)):
        error_matrix += s_sqrt_inv_errors[i] * np.outer(u[:, i], vh[i, :])
    
    return np.abs(error_matrix)

# Run the test
if __name__ == "__main__":
    test_error_propagation()

# Function to calculate both weighted mean and standard error in a single call
def weighted_mean_and_standard_error(values, weights):
    """
    Calculate weighted mean and standard error for a set of values with weights.
    Handles complex values by treating real and imaginary parts separately.
    
    Parameters:
        values: Tensor of values to calculate mean and error for
        weights: Weights for each value
        
    Returns:
        weighted_mean: The weighted mean of the values
        standard_error: The standard error of the weighted mean
        effective_n: The effective sample size
    """
    # Assert that weights are real
    tf.debugging.assert_all_finite(weights, "Weights must be real values")
    
    # Get the weighted mean
    weighted_mean = tf.reduce_mean(tf.cast(weights, values.dtype) * values)
    
    # Calculate the effective sample size
    n_eff = effective_sample_size(weights)
    
    # Calculate variance for real part
    real_values = tf.math.real(values)
    real_mean = tf.math.real(weighted_mean)
    real_squared_deviation = tf.square(real_values - real_mean)
    real_weighted_variance = tf.reduce_sum(weights * real_squared_deviation) / tf.reduce_sum(weights)
    
    # Standard error for real part
    real_se = tf.sqrt(real_weighted_variance / n_eff)
    
    # Check if values are complex
    if hasattr(values, 'dtype') and 'complex' in str(values.dtype) and np.any(np.imag(np.array(values)) != 0):
        # Calculate variance for imaginary part
        imag_values = tf.math.imag(values)
        imag_mean = tf.math.imag(weighted_mean)
        imag_squared_deviation = tf.square(imag_values - imag_mean)
        imag_weighted_variance = tf.reduce_sum(weights * imag_squared_deviation) / tf.reduce_sum(weights)
        
        # Standard error for imaginary part
        imag_se = tf.sqrt(imag_weighted_variance / n_eff)
        
        # Return complex standard error
        return weighted_mean, tf.complex(real_se, imag_se), n_eff, {'value': weighted_mean.numpy(), 'std_error': real_se.numpy(), 'eff_n': n_eff.numpy()}
    else:
        # Return real standard error
        return weighted_mean, real_se, n_eff, {'value': weighted_mean.numpy(), 'std_error': real_se.numpy(), 'eff_n': n_eff.numpy()} 

def propagate_errors_to_physical_yukawas(NormH, NormH_errors, NormQ, NormQ_errors, NormU, NormU_errors, m, m_errors):
    """
    Propagate errors from normalization matrices and holomorphic Yukawas to physical Yukawas.
    
    Parameters:
        NormH (np.ndarray): Higgs normalization matrix
        NormH_errors (np.ndarray): Errors in Higgs normalization matrix
        NormQ (np.ndarray): Q field normalization matrix
        NormQ_errors (np.ndarray): Errors in Q field normalization matrix
        NormU (np.ndarray): U field normalization matrix
        NormU_errors (np.ndarray): Errors in U field normalization matrix
        m (np.ndarray): Holomorphic Yukawa matrix
        m_errors (np.ndarray): Errors in holomorphic Yukawa matrix

    Notes:
        This function propagates measurement errors from normalization matrices and 
        holomorphic Yukawa couplings to the physical Yukawa matrix. The physical Yukawa 
        is calculated as Y = h * Q * m * U^T, where h is the Higgs normalization factor,
        Q and U are normalization matrices, and m is the holomorphic Yukawa matrix.
        
        The error propagation uses standard error propagation formulas, calculating how
        errors in each input matrix contribute to the final physical Yukawa values.

        For each element Y_ij of the physical Yukawa matrix, we consider:
        1. Error contribution from the Higgs normalization factor
        2. Error contribution from each element of the Q matrix
        3. Error contribution from each element of the U matrix
        4. Error contribution from each element of the holomorphic Yukawa matrix m
        
        These contributions are combined in quadrature to get the total error.
        
    Returns:
        tuple: (physical_yukawas, physical_yukawas_errors)
    """
    # Convert TensorFlow tensors to numpy arrays if needed
    if hasattr(NormH, 'numpy'):
        NormH_np = NormH.numpy()
    else:
        NormH_np = np.array(NormH)
        
    if hasattr(NormQ, 'numpy'):
        NormQ_np = NormQ.numpy()
    else:
        NormQ_np = np.array(NormQ)
        
    if hasattr(NormU, 'numpy'):
        NormU_np = NormU.numpy()
    else:
        NormU_np = np.array(NormU)
    
    m_np = np.array(m)
    
    # Calculate physical yukawas
    h = NormH_np[0][0]
    physical_yukawas = h * np.einsum('ab,cd,bd->ac', NormQ_np, NormU_np, m_np)
    
    # Initialize error matrix
    physical_yukawas_errors = np.zeros_like(physical_yukawas, dtype=complex)
    
    # Get scalar normalization factor error
    h_error = NormH_errors[0][0]
    
    # Calculate errors for each element of physical yukawas
    for i in range(physical_yukawas.shape[0]):
        for j in range(physical_yukawas.shape[1]):
            # 1. Error from h (scalar term)
            error_from_h = np.abs(physical_yukawas[i, j]) * (h_error / np.abs(h))
            
            # 2. Error from Q matrix elements
            error_from_Q = 0
            for k in range(NormQ_np.shape[1]):
                # ∂Y_ij/∂Q_ik = h * Σ_l (m_kl * U_lj)
                partial_deriv = h * np.sum([m_np[k, l] * NormU_np[l, j] for l in range(NormU_np.shape[0])])
                error_from_Q += (np.abs(partial_deriv) * NormQ_errors[i, k])**2
            error_from_Q = np.sqrt(error_from_Q)
            
            # 3. Error from U matrix elements
            error_from_U = 0
            for l in range(NormU_np.shape[0]):
                # ∂Y_ij/∂U_lj = h * Σ_k (Q_ik * m_kl)
                partial_deriv = h * np.sum([NormQ_np[i, k] * m_np[k, l] for k in range(NormQ_np.shape[1])])
                error_from_U += (np.abs(partial_deriv) * NormU_errors[l, j])**2
            error_from_U = np.sqrt(error_from_U)
            
            # 4. Error from m matrix elements
            error_from_m = 0
            for k in range(m_np.shape[0]):
                for l in range(m_np.shape[1]):
                    # ∂Y_ij/∂m_kl = h * Q_ik * U_lj
                    partial_deriv = h * NormQ_np[i, k] * NormU_np[l, j]
                    error_from_m += (np.abs(partial_deriv) * m_errors[k, l])**2
            error_from_m = np.sqrt(error_from_m)
            
            # Combine all error sources
            physical_yukawas_errors[i, j] = np.sqrt(
                error_from_h**2 + error_from_Q**2 + error_from_U**2 + error_from_m**2
            )
    
    return physical_yukawas, physical_yukawas_errors

def test_yukawa_error_propagation_monte_carlo(NormH, NormH_errors, NormQ, NormQ_errors, 
                                               NormU, NormU_errors, m, m_errors, n_samples=10000):
    """
    Test error propagation for physical Yukawas using Monte Carlo simulation.
    
    Parameters:
        NormH, NormH_errors, NormQ, NormQ_errors, NormU, NormU_errors, m, m_errors: 
            Same parameters as propagate_errors_to_physical_yukawas
        n_samples (int): Number of Monte Carlo samples
        
    Returns:
        tuple: (analytical_result, monte_carlo_result, ratio)
            - analytical_result: (physical_yukawas, physical_yukawas_errors) from analytical method
            - monte_carlo_result: (physical_yukawas_mean, physical_yukawas_std) from Monte Carlo
            - ratio: ratio of analytical errors to Monte Carlo errors
    """
    
    # Convert all inputs to numpy arrays
    if hasattr(NormH, 'numpy'):
        NormH_np = NormH.numpy()
    else:
        NormH_np = np.array(NormH)
        
    if hasattr(NormQ, 'numpy'):
        NormQ_np = NormQ.numpy()
    else:
        NormQ_np = np.array(NormQ)
        
    if hasattr(NormU, 'numpy'):
        NormU_np = NormU.numpy()
    else:
        NormU_np = np.array(NormU)
    
    m_np = np.array(m)
    
    # Get analytical result
    analytical_result = propagate_errors_to_physical_yukawas(
        NormH_np, NormH_errors, NormQ_np, NormQ_errors, NormU_np, NormU_errors, m_np, m_errors
    )
    physical_yukawas, physical_yukawas_errors = analytical_result
    
    # Generate perturbed variables for Monte Carlo
    is_complex = (np.iscomplexobj(NormH_np) or np.iscomplexobj(NormQ_np) or 
                  np.iscomplexobj(NormU_np) or np.iscomplexobj(m_np))
    
    # Storage for Monte Carlo results
    all_physical_yukawas = []
    
    # Run Monte Carlo simulation
    for _ in range(n_samples):
        # Perturb H
        if is_complex:
            H_perturbed = NormH_np.copy()
            H_perturbed[0, 0] += (np.random.normal(0, np.real(NormH_errors[0, 0])) + 
                                  1j * np.random.normal(0, np.imag(NormH_errors[0, 0])))
        else:
            H_perturbed = NormH_np + np.random.normal(0, NormH_errors)
        
        # Perturb Q
        if is_complex and np.iscomplexobj(NormQ_errors):
            Q_perturbed = NormQ_np.copy()
            for i in range(NormQ_np.shape[0]):
                for j in range(NormQ_np.shape[1]):
                    Q_perturbed[i, j] += (np.random.normal(0, np.real(NormQ_errors[i, j])) + 
                                          1j * np.random.normal(0, np.imag(NormQ_errors[i, j])))
        else:
            Q_perturbed = NormQ_np + np.random.normal(0, NormQ_errors)
        
        # Perturb U
        if is_complex and np.iscomplexobj(NormU_errors):
            U_perturbed = NormU_np.copy()
            for i in range(NormU_np.shape[0]):
                for j in range(NormU_np.shape[1]):
                    U_perturbed[i, j] += (np.random.normal(0, np.real(NormU_errors[i, j])) + 
                                          1j * np.random.normal(0, np.imag(NormU_errors[i, j])))
        else:
            U_perturbed = NormU_np + np.random.normal(0, NormU_errors)
        
        # Perturb m
        if is_complex and np.iscomplexobj(m_errors):
            m_perturbed = m_np.copy()
            for i in range(m_np.shape[0]):
                for j in range(m_np.shape[1]):
                    m_perturbed[i, j] += (np.random.normal(0, np.real(m_errors[i, j])) + 
                                          1j * np.random.normal(0, np.imag(m_errors[i, j])))
        else:
            m_perturbed = m_np + np.random.normal(0, m_errors)
        
        # Calculate perturbed physical yukawas
        h_perturbed = H_perturbed[0][0]
        physical_yukawas_perturbed = h_perturbed * np.einsum('ab,cd,bd->ac', 
                                                            Q_perturbed, 
                                                            U_perturbed, 
                                                            m_perturbed)
        all_physical_yukawas.append(physical_yukawas_perturbed)
    
    # Calculate statistics from Monte Carlo results
    all_physical_yukawas = np.array(all_physical_yukawas)
    physical_yukawas_mean = np.mean(all_physical_yukawas, axis=0)
    physical_yukawas_std = np.std(all_physical_yukawas, axis=0, ddof=1)
    
    # Calculate ratio between analytical and Monte Carlo errors
    ratio = physical_yukawas_errors / physical_yukawas_std
    
    # # Optionally visualize results
    # plt.figure(figsize=(10, 8))
    # plt.suptitle('Yukawa Error Propagation: Analytical vs Monte Carlo')
    
    rows, cols = physical_yukawas.shape
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, i*cols + j + 1)
            
            if is_complex:
                # Plot real and imaginary parts separately
                real_vals = np.real(all_physical_yukawas[:, i, j])
                imag_vals = np.imag(all_physical_yukawas[:, i, j])
                
                plt.hist(real_vals, bins=50, alpha=0.5, label='Real')
                plt.hist(imag_vals, bins=50, alpha=0.5, label='Imag')
                
                plt.axvline(np.real(physical_yukawas[i, j]), color='r', linestyle='--', 
                            label=f'Real Mean: {np.real(physical_yukawas[i, j]):.2e}')
                plt.axvline(np.imag(physical_yukawas[i, j]), color='g', linestyle='--', 
                            label=f'Imag Mean: {np.imag(physical_yukawas[i, j]):.2e}')
                
                plt.title(f'Y[{i},{j}], Ratio={np.real(ratio[i, j]):.2f}')
            else:
                plt.hist(all_physical_yukawas[:, i, j], bins=50)
                plt.axvline(physical_yukawas[i, j], color='r', linestyle='--', 
                            label=f'Mean: {physical_yukawas[i, j]:.2e}')
                plt.title(f'Y[{i},{j}], Ratio={ratio[i, j]:.2f}')
            
            if i == rows-1 and j == 0:
                plt.legend()
    
    plt.tight_layout()
    # Ensure data directory exists before saving
    # import os
    # os.makedirs('data', exist_ok=True)
    # plt.savefig('data/yukawa_error_propagation_test.png')
    # plt.close()
    
    # Print comparison 
    print("===== Yukawa Error Propagation Comparison =====")
    print("Analytical Method vs. Monte Carlo (n_samples =", n_samples, ")")
    print("\nPhysical Yukawa values:")
    print(np.round(physical_yukawas*1e6, 1), "×10⁻⁶")
    
    print("\nAnalytical error estimates (×10⁻⁶):")
    print(np.round(np.abs(physical_yukawas_errors)*1e6, 1))
    
    print("\nMonte Carlo error estimates (×10⁻⁶):")
    print(np.round(np.abs(physical_yukawas_std)*1e6, 1))
    
    print("\nRatio (Analytical/Monte Carlo):")
    print(np.round(np.abs(ratio), 2))
    
    # Check if the errors are consistent (within 20% of each other)
    tolerance = 0.2
    if np.all(np.abs(np.abs(ratio) - 1) < tolerance):
        print("\nVALIDATION PASSED! ✅ Analytical errors match Monte Carlo within", 
              int(tolerance*100), "% tolerance")
    else:
        print("\nVALIDATION WARNING! ⚠️ Some analytical errors differ from Monte Carlo by more than", 
              int(tolerance*100), "%")
        raise Exception("ERROR")
        
    return (physical_yukawas, physical_yukawas_errors), (physical_yukawas_mean, physical_yukawas_std), ratio


"""
Test the error propagation for physical Yukawas using Monte Carlo simulation.
This file provides a simple test case to validate the error propagation approach.
"""

import numpy as np
from auxiliary_funcs import propagate_errors_to_physical_yukawas, test_yukawa_error_propagation_monte_carlo

def run_simple_test():
    """
    Run a simple test with a 3x3 Yukawa matrix with known errors.
    """
    print("Running simple test of Yukawa error propagation...")
    
    # Create test matrices with errors
    # Simple 1x1 Higgs matrix
    NormH = np.array([[2.0 + 0.1j]])
    NormH_errors = np.array([[0.05 + 0.02j]])
    
    # 3x3 Q and U matrices
    NormQ = np.array([
        [1.2 + 0.1j, 0.1 + 0.05j, 0.01 + 0.02j],
        [0.1 + 0.03j, 1.3 + 0.2j, 0.02 + 0.01j],
        [0.01 + 0.01j, 0.02 + 0.02j, 1.4 + 0.15j]
    ])
    NormQ_errors = 0.05 * np.abs(NormQ)
    
    NormU = np.array([
        [1.1 + 0.2j, 0.15 + 0.03j, 0.02 + 0.01j],
        [0.15 + 0.05j, 1.2 + 0.1j, 0.03 + 0.02j],
        [0.02 + 0.01j, 0.03 + 0.01j, 1.3 + 0.1j]
    ])
    NormU_errors = 0.05 * np.abs(NormU)
    
    # Holomorphic Yukawa matrix
    m = np.array([
        [0.5 + 0.1j, 0.02 + 0.01j, 0.001 + 0.0005j],
        [0.02 + 0.01j, 0.4 + 0.15j, 0.03 + 0.01j],
        [0.001 + 0.0005j, 0.03 + 0.01j, 0.3 + 0.05j]
    ])
    m_errors = 0.03 * np.abs(m)
    
    # Run the Monte Carlo test
    results = test_yukawa_error_propagation_monte_carlo(
        NormH, NormH_errors, NormQ, NormQ_errors, 
        NormU, NormU_errors, m, m_errors, 
        n_samples=10000
    )
    
    return results

def run_realistic_test():
    """
    Run a test with more realistic parameters that match what's used in the actual code.
    """
    print("\nRunning realistic test of Yukawa error propagation...")
    
    # Create matrices similar to those in the actual calculation
    # Higgs matrix (1x1)
    mfieldmatrixfactors = 1/16  # Example factor
    HuHu = 8.0 + 0.1j
    HuHu_error = 0.2 + 0.05j
    Hmat = mfieldmatrixfactors * np.array([[HuHu]])
    Hmat_errors = mfieldmatrixfactors * np.array([[HuHu_error]])
    
    # Q matrix (3x3)
    Q1Q1, Q2Q2, Q3Q3 = 7.5 + 0.2j, 7.8 + 0.15j, 8.1 + 0.1j
    Q1Q2, Q2Q1 = 0.3 + 0.1j, 0.3 + 0.1j  # Should be complex conjugates in reality
    Q_errors = 0.1 + 0.05j  # Same error for all elements for simplicity
    
    Qmat = mfieldmatrixfactors * np.array([
        [Q1Q1, Q2Q1, 0],
        [Q1Q2, Q2Q2, 0],
        [0, 0, Q3Q3]
    ])
    
    Qmat_errors = mfieldmatrixfactors * np.array([
        [Q_errors, Q_errors, 0],
        [Q_errors, Q_errors, 0],
        [0, 0, Q_errors]
    ])
    
    # U matrix (3x3)
    U1U1, U2U2, U3U3 = 7.2 + 0.25j, 7.6 + 0.2j, 7.9 + 0.15j
    U1U2, U2U1 = 0.25 + 0.08j, 0.25 + 0.08j  # Should be complex conjugates in reality
    U_errors = 0.15 + 0.03j  # Same error for all elements for simplicity
    
    Umat = mfieldmatrixfactors * np.array([
        [U1U1, U2U1, 0],
        [U1U2, U2U2, 0],
        [0, 0, U3U3]
    ])
    
    Umat_errors = mfieldmatrixfactors * np.array([
        [U_errors, U_errors, 0],
        [U_errors, U_errors, 0],
        [0, 0, U_errors]
    ])
    
    # Holomorphic Yukawa matrix (example values)
    factor = 2e-6  # Scale to match values in test_integration.py
    m = factor * np.array([
        [5.0 + 1.0j, 0.2 + 0.1j, 0.01 + 0.005j],
        [0.2 + 0.1j, 4.0 + 1.5j, 0.3 + 0.1j],
        [0.01 + 0.005j, 0.3 + 0.1j, 3.0 + 0.5j]
    ])
    m_errors = 0.1 * np.abs(m)  # 10% error
    
    # Import scipy for sqrtm function
    import scipy.linalg
    
    # Calculate normalization matrices
    NormH = np.linalg.inv(scipy.linalg.sqrtm(Hmat).astype(complex))
    NormQ = np.linalg.inv(scipy.linalg.sqrtm(Qmat).astype(complex))
    NormU = np.linalg.inv(scipy.linalg.sqrtm(Umat).astype(complex))
    
    # Run the Monte Carlo test
    results = test_yukawa_error_propagation_monte_carlo(
        NormH, Hmat_errors, NormQ, Qmat_errors, 
        NormU, Umat_errors, m, m_errors, 
        n_samples=10000
    )
    
    return results

if __name__ == "__main__":
    # Run both tests
    simple_results = run_simple_test()
    realistic_results = run_realistic_test()
    
    print("\nTests completed! Check the generated plots for visualization of results.") 

