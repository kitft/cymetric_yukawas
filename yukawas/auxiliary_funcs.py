import gc
import tensorflow as tf
import time
from datetime import datetime
import numpy as np
from cymetric.config import real_dtype, complex_dtype
def delete_all_dicts_except(*except_dict_names):
    """
    Deletes all dictionary objects from the global namespace except the specified ones.

    Parameters:
    *except_dict_names (str): Names of dictionaries to exclude from deletion.
    """

    # Identify all dictionary variables except those specified
    dicts_to_delete = [
        var for var, obj in globals().items()
        if isinstance(obj, dict) and var not in except_dict_names
    ]

    # Delete all identified dictionary variables
    for var in dicts_to_delete:
        del globals()[var]

    # Optionally, you can clear memory by calling garbage collector
    gc.collect()
def batch_process_helper_func(func_orig, args, batch_indices=(0,), batch_size=10000, compile_func=False, actually_batch=True, kwargs=None, print_progress=False, batch_kwargs_keys=None):
    if kwargs is None:
        kwargs = {}
        
    if actually_batch==False:
        return func_orig(*args, **kwargs)

    # Optionally compile the function for better performance
    if compile_func:
        func = tf.function(func_orig)
    else:
        func = func_orig
        
    # Determine the number of batches based on the first batched argument
    num_batches = tf.cast(tf.math.ceil(tf.shape(args[batch_indices[0]])[0] / batch_size), tf.int32)
    # Get function name safely - handle both regular functions and tf.function objects
    func_name = func_orig.__name__ if hasattr(func_orig, '__name__') else str(func_orig)
    print(f"Batching function {func_name} with {num_batches} batches, compiled? {compile_func}")
    results_list = []
    import time
    start_time = time.time()
    first_iter_time = None
    second_iter_time = None

    for i in range(num_batches):
        iter_start_time = time.time()
        start_idx = i * batch_size
        end_idx = tf.minimum((i + 1) * batch_size, tf.shape(args[batch_indices[0]])[0])
        
        # Create batched arguments
        batched_args = list(args)
        for idx in batch_indices:
            batched_args[idx] = args[idx][start_idx:end_idx]
        
        # Create batched kwargs with only the relevant keys
        batched_kwargs = {}
        # Copy only the keys we need to batch
        if batch_kwargs_keys:
            for key in batch_kwargs_keys:
                if key in kwargs:
                    batched_kwargs[key] = kwargs[key][start_idx:end_idx]
            # Add remaining keys from original kwargs
            for key, value in kwargs.items():
                if key not in batched_kwargs:
                    batched_kwargs[key] = value
        else:
            batched_kwargs = kwargs
        # Call the function with batched and static arguments
        batch_results = func(*batched_args, **batched_kwargs)
        results_list.append(batch_results)
        
        # Time tracking and ETA calculation
        if i == 0:
            first_iter_time = time.time() - iter_start_time
            print(f"    First batch took {first_iter_time:.2f}s")
        elif i == 1:
            second_iter_time = time.time() - iter_start_time
            eta = second_iter_time * (tf.cast(num_batches, tf.float32) - 2)
            #if print_progress:
            print(f"    Second batch took {second_iter_time:.2f}s. ETA: {eta:.2f}s")
        else:
            if print_progress:
                print(i)

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
def propagate_errors_to_singular_values_legacy(matrix, matrix_errors):
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

        # For complex matrices, handle real and imaginary errors separately
        if np.iscomplexobj(matrix):
            # Calculate variance contributions from real and imaginary parts
            real_variance = np.sum((sensitivity ** 2) * (np.abs(np.real(matrix_errors)) ** 2))
            imag_variance = np.sum((sensitivity ** 2) * (np.abs(np.imag(matrix_errors)) ** 2))
            # Total variance is sum of real and imaginary variances
            variance = real_variance + imag_variance
        else:
            # For real matrices, use original calculation
            variance = np.sum((sensitivity ** 2) * (matrix_errors ** 2))

        # Standard error is square root of variance
        s_errors[i] = np.sqrt(variance)

    return s_errors
def propagate_errors_to_singular_values(matrix, matrix_errors):
    """
    Propagate errors from matrix elements to singular values using Monte Carlo simulation.
    Works for both real and complex matrices.
    
    Parameters:
    matrix (np.ndarray): Input matrix (real or complex)
    matrix_errors (np.ndarray): Matrix of standard errors for each element
    
    Returns:
    np.ndarray: Standard errors for each singular value
    """
    # Perform SVD on the original matrix to get number of singular values
    _, s_orig, _ = np.linalg.svd(matrix, full_matrices=False)
    
    # Use Monte Carlo to estimate errors in singular values
    n_samples = 10000
    all_singular_values = []
    
    # Generate perturbed matrices and compute their SVDs
    for _ in range(n_samples):
        # Create random perturbation according to error distribution
        if np.iscomplexobj(matrix):
            real_perturbations = np.random.normal(0, np.real(matrix_errors))
            imag_perturbations = np.random.normal(0, np.imag(matrix_errors))
            perturbations = real_perturbations + 1j * imag_perturbations
        else:
            perturbations = np.random.normal(0, matrix_errors)
        
        # Perturb the matrix
        perturbed_matrix = matrix + perturbations
        
        # Calculate SVD for perturbed matrix
        try:
            _, s_perturbed, _ = np.linalg.svd(perturbed_matrix, full_matrices=False)
            all_singular_values.append(s_perturbed)
        except np.linalg.LinAlgError:
            # Skip invalid matrices (e.g., non-positive definite)
            continue
    
    # Convert samples to array
    all_singular_values = np.array(all_singular_values)
    
    # Calculate standard deviation for each singular value
    s_errors = np.std(all_singular_values, axis=0)
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
    error_size = 0.005
    print("Testing singular values with real matrix:")
    matrix_errors_real = error_size * np.abs(real_matrix)
    singular_values_real = np.linalg.svd(real_matrix, compute_uv=False)
    print("Original singular values:", singular_values_real)
    
    s_errors_analytical_real = propagate_errors_to_singular_values(real_matrix, matrix_errors_real)
    print("Analytical errors:", s_errors_analytical_real)
    
    s_errors_mc_real = verify_by_monte_carlo(real_matrix, matrix_errors_real)
    print("Monte Carlo errors:", s_errors_mc_real)
    print("Ratio (Analytical/MC):", s_errors_analytical_real / s_errors_mc_real)
    
    # Test with complex matrix
    print("\nTesting with complex matrix:")
    matrix_errors_complex = error_size * complex_matrix
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
    
    # Test with many random complex 3x3 matrices
    tolerance = 0.2  # 20% tolerance
    n_test_matrices = 10
    failures = 0
    print("Testing with many random complex 3x3 matrices")
    for i in range(n_test_matrices):
        # Generate random complex matrix
        real_part = np.random.normal(0, 1, (3, 3))
        imag_part = np.random.normal(0, 1, (3, 3))
        test_matrix = real_part + 1j * imag_part
        
        # Generate error matrix (5% of matrix values)
        error_matrix = 0.05 *(np.abs(np.random.normal(0, 1, (3, 3))) + 1j*np.abs(np.random.normal(0, 1, (3, 3))))
        
        # Calculate errors using both methods
        analytical_errors = propagate_errors_to_singular_values(test_matrix, error_matrix)
        mc_errors = verify_by_monte_carlo(test_matrix, error_matrix, n_samples=5000)
        
        # Check if ratios are within tolerance
        ratios = analytical_errors / mc_errors
        if not np.all(np.abs(ratios - 1) < tolerance):
            failures += 1
            print(f"Failure on matrix {i+1}: Ratios outside tolerance: {ratios}")
            print(f"Analytical errors: {analytical_errors}")
            print(f"Monte Carlo errors: {mc_errors}")
    
    if failures == 0:
        print(f"All {n_test_matrices} random complex matrix tests passed!")
    else:
        print(f"{failures} out of {n_test_matrices} random complex matrix tests failed.")

    

def propagate_errors_through_normalization(matrix, matrix_errors, n_samples=10000):
    """
    Propagate errors through matrix square root and inversion operations
    using Monte Carlo simulation.
    
    Parameters:
        matrix: Original matrix
        matrix_errors: Error matrix
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Standard deviation of the normalization matrix elements
    """
    import scipy
    import numpy as np
    
    # Ensure inputs are complex
    matrix = np.asarray(matrix, dtype=complex)
    matrix_errors = np.asarray(matrix_errors, dtype=complex)
    
    # Prepare array for Monte Carlo results
    norm_samples = []
    
    # Run Monte Carlo simulation
    for _ in range(n_samples):
        # Generate perturbed matrix with complex perturbations
        perturbed = matrix.copy()
        
        # Add perturbations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix_errors[i, j] != 0:
                    real_error = np.random.normal(0, np.abs(np.real(matrix_errors[i, j])))
                    imag_error = np.random.normal(0, np.abs(np.imag(matrix_errors[i, j])))
                    perturbed[i, j] += real_error + 1j * imag_error
        
        # Calculate normalization matrix for perturbed values
        norm = np.linalg.inv(scipy.linalg.sqrtm(perturbed).astype(complex))
        norm_samples.append(norm)
    
    # Convert list to numpy array
    norm_samples = np.array(norm_samples)
    
    # Calculate standard deviations (separately for real and imaginary parts)
    norm_errors_real = np.std(np.real(norm_samples), axis=0, ddof=1)
    norm_errors_imag = np.std(np.imag(norm_samples), axis=0, ddof=1)
    
    return norm_errors_real + 1j * norm_errors_imag
    # Function to propagate errors through matrix square root and inversion
def propagate_errors_through_normalization_legacy(matrix, matrix_errors):
    """
    Propagate errors through matrix square root and inversion operations.
    Uses finite difference method for accurate error propagation.
    """
    import scipy
    import numpy as np
    # For a 1x1 matrix, use the same finite difference approach as for larger matrices
    if matrix.shape == (1, 1):
        # Calculate A^(-1/2) directly
        inv_sqrt_value = 1.0 / np.sqrt(matrix[0, 0])
        
        # Use finite difference to calculate derivatives in real and imaginary directions separately
        h = 1e-8
        
        # Real direction perturbation
        perturbed_matrix_real = matrix[0, 0] + h
        perturbed_value_real = 1.0 / np.sqrt(perturbed_matrix_real)
        derivative_real = (perturbed_value_real - inv_sqrt_value) / h
        
        # Imaginary direction perturbation (if complex)
        if np.iscomplexobj(matrix):
            perturbed_matrix_imag = matrix[0, 0] + 1j * h
            perturbed_value_imag = 1.0 / np.sqrt(perturbed_matrix_imag)
            derivative_imag = (perturbed_value_imag - inv_sqrt_value) / (1j * h)
            # Combine real and imaginary components in quadrature for proper error propagation
            real_part_error = np.sqrt((np.real(derivative_real) * np.real(matrix_errors[0, 0]))**2 + 
                                     (np.real(derivative_imag) * np.imag(matrix_errors[0, 0]))**2)
            imag_part_error = np.sqrt((np.imag(derivative_real) * np.real(matrix_errors[0, 0]))**2 + 
                                     (np.imag(derivative_imag) * np.imag(matrix_errors[0, 0]))**2)
            return np.array([[real_part_error + 1j * imag_part_error]])
        else:
            return np.array([[derivative_real * matrix_errors[0, 0]]])
            
    # Calculate A^(-1/2) directly
    inv_sqrt_matrix = np.linalg.inv(scipy.linalg.sqrtm(matrix).astype(complex))
    
    # Initialize error matrix with zeros
    error_matrix = np.zeros_like(matrix, dtype=complex)
    
    # Initialize matrices to accumulate squared errors
    error_matrix_real_squared = np.zeros_like(matrix, dtype=float)
    error_matrix_imag_squared = np.zeros_like(matrix, dtype=float)
    
    # For each element of the original matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Skip if error is zero
            if matrix_errors[i, j] == 0:
                continue
                
            # Create perturbation matrix (zeros with single element set to 1)
            perturbation = np.zeros_like(matrix, dtype=complex)
            perturbation[i, j] = 1.0
            
            # Calculate directional derivative using finite difference approximation
            h = 1e-8
            
            # Real direction perturbation
            perturbed_matrix_real = matrix + h * perturbation
            perturbed_inv_sqrt_real = np.linalg.inv(scipy.linalg.sqrtm(perturbed_matrix_real).astype(complex))
            frechet_derivative_real = (perturbed_inv_sqrt_real - inv_sqrt_matrix) / h
            
            # Imaginary direction perturbation (if complex)
            perturbed_matrix_imag = matrix + 1j * h * perturbation
            perturbed_inv_sqrt_imag = np.linalg.inv(scipy.linalg.sqrtm(perturbed_matrix_imag).astype(complex))
            frechet_derivative_imag = (perturbed_inv_sqrt_imag - inv_sqrt_matrix) / (1j * h)
            
            # Calculate error contribution and add squared errors
            real_part_contribution = (np.real(frechet_derivative_real) * np.real(matrix_errors[i, j]))**2 + (np.real(frechet_derivative_imag) * np.imag(matrix_errors[i, j]))**2
            imag_part_contribution = (np.imag(frechet_derivative_real) * np.real(matrix_errors[i, j]))**2 + (np.imag(frechet_derivative_imag) * np.imag(matrix_errors[i, j]))**2
            
            # Accumulate squared errors
            error_matrix_real_squared += real_part_contribution
            error_matrix_imag_squared += imag_part_contribution
    
    # Take square root of accumulated squared errors to get final error matrix
    error_matrix = np.sqrt(error_matrix_real_squared) + 1j * np.sqrt(error_matrix_imag_squared)
    
    # For complex matrices, ensure the error is properly handled
    if np.iscomplexobj(matrix) or np.iscomplexobj(matrix_errors):
        # Take absolute values for the error magnitudes
        error_matrix = np.abs(np.real(error_matrix)) + 1j * np.abs(np.imag(error_matrix))
    
    return error_matrix

# Run the test
if __name__ == "__main__":
    test_error_propagation()

def weighted_mean_std_error(weights, values):
    """
    Calculate the standard error of a weighted mean estimator.
    $$SE(\hat{\mu}_w) = \sqrt{\frac{\sum_i w_i^2 (x_i - \hat{\mu}_w)^2}$$

    Parameters:
        weights: Tensor of weights for each value
        values: Tensor of values to calculate weighted mean for

    Returns:
        standard_error: The standard error of the weighted mean
    """
    # Cast to appropriate dtype
    weights = tf.cast(weights, real_dtype)
    values = tf.cast(values, real_dtype)

    # Calculate weighted mean
    weighted_mean = tf.reduce_mean(weights * values)

    # Calculate squared deviations
    squared_deviations = tf.square(weights*values - weighted_mean)

    # Calculate standard error
    # Note the weights are squared in the numerator
    variance_estimator = tf.reduce_mean( squared_deviations) *1/(len(values)-1)

    return tf.sqrt(variance_estimator)
    # norm_weights = weights/tf.reduce_mean(weights)
    # weighted_average = tf.reduce_mean(norm_weights*values)
    # weighted_variance = tf.reduce_mean(weights**2 * (values-weighted_average)**2)
    # standard_error = tf.sqrt(weighted_variance)
    # return standard_error

# Function to calculate both weighted mean and standard error in a single call
def weighted_mean_and_standard_error(values, weights, is_top_form=False, mulweightsby=None, return_z_score=False):
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
    if mulweightsby is not None:
        weights=weights*tf.cast(mulweightsby, weights.dtype)
        values=values*1/tf.cast(mulweightsby, values.dtype)
    
    # Get the weighted mean
    weighted_mean = tf.reduce_mean(tf.cast(weights, values.dtype) * values)

    
   
    # Check if values are complex
    if (tf.is_tensor(values) and tf.dtypes.as_dtype(values.dtype).is_complex) or (isinstance(values, np.ndarray) and np.issubdtype(values.dtype, complex)):
        weighted_stddev_real2 = tf.math.reduce_std(tf.math.real(tf.cast(weights, values.dtype) * values))/np.sqrt(len(values))
        weighted_stddev_imag2 = tf.math.reduce_std(tf.math.imag(tf.cast(weights, values.dtype) * values))/np.sqrt(len(values))
        weighted_stddev_real=weighted_mean_std_error(weights, tf.math.real(values))
        weighted_stddev_imag=weighted_mean_std_error(weights, tf.math.imag(values))
        #print("compare real stddev: ", np.array(weighted_stddev_real), np.array(weighted_stddev_real2))
        #print("compare imag stddev: ", np.array(weighted_stddev_imag), np.array(weighted_stddev_imag2))
        weighted_stddev = tf.complex(weighted_stddev_real, weighted_stddev_imag)
         # Calculate the effective sample size

        n_eff_r = effective_sample_size(weights*tf.cast(tf.math.real(values), real_dtype))
        if not np.any(np.imag(np.array(values)) != 0):
            n_eff_c = 1.0
        else:
            n_eff_c = effective_sample_size(weights*tf.cast(tf.math.imag(values), real_dtype))


        # Calculate variance for real part
        real_values = tf.math.real(values)
        real_mean = tf.math.real(weighted_mean)
        #real_squared_deviation = tf.square(real_values - real_mean)
        #real_weighted_variance = tf.reduce_sum(weights * real_squared_deviation) / tf.reduce_sum(weights)
        real_variance = tf.math.reduce_std(weights*real_values)
    
        # Standard error for real part
        #real_se = real_variance / np.sqrt(n_eff_r)
    
        # Calculate variance for imaginary part
        imag_values = tf.math.imag(values)
        imag_mean = tf.math.imag(weighted_mean)
        #imag_squared_deviation = tf.square(imag_values - imag_mean)
        #imag_weighted_variance = tf.reduce_sum(weights * imag_squared_deviation) / tf.reduce_sum(weights)
        imag_variance = tf.math.reduce_std(weights*imag_values)
        
        # Calculate covariance between real and imaginary parts
        real_imag_values = tf.stack([real_values*weights, imag_values*weights], axis=1)
        cov_matrix = tf.tensordot(
            tf.transpose(real_imag_values - tf.reduce_mean(real_imag_values, axis=0)),
            real_imag_values - tf.reduce_mean(real_imag_values, axis=0),
            axes=[[1], [0]]
        )
        # Handle tensor elements individually to avoid scalar conversion error
        #cov_matrix_np = cov_matrix.numpy()
        #print(f"----Covariance matrix Re/Im: [[{np.format_float_scientific(cov_matrix_np[0,0], precision=2)},{np.format_float_scientific(cov_matrix_np[0,1], precision=2)}], [{np.format_float_scientific(cov_matrix_np[1,0], precision=2)},{np.format_float_scientific(cov_matrix_np[1,1], precision=2)}]], \
        #      sqrtabs:[[{np.format_float_scientific(np.sqrt(np.abs(cov_matrix_np[0,0])), precision=2)},{np.format_float_scientific(np.sqrt(np.abs(cov_matrix_np[0,1])), precision=2)}], [{np.format_float_scientific(np.sqrt(np.abs(cov_matrix_np[1,0])), precision=2)},{np.format_float_scientific(np.sqrt(np.abs(cov_matrix_np[1,1])), precision=2)}]]")
        
        # Standard error for imaginary part
        #imag_se = imag_variance / np.sqrt(n_eff_c)
        
        # Return complex standard error
        #complex_std_error = tf.complex(real_se, imag_se)
        neff_complex = complex(n_eff_r, n_eff_c)
        return np.array(weighted_mean), np.array(weighted_stddev), neff_complex, {'value': np.array(weighted_mean), 'std_error': np.array(weighted_stddev), 'eff_n': neff_complex, 'z_score': np.abs(np.array(weighted_mean))/(np.abs(np.array(weighted_stddev))+1e-10)}
    else:
        if is_top_form:
            n_eff = effective_sample_size(weights*values)
        else:
            n_eff = effective_sample_size(weights)
        weighted_stddev2 =  tf.math.reduce_std(weights*values)*1/np.sqrt(len(values))
        weighted_stddev = weighted_mean_std_error(weights, values)
        #print("compare real stddev: ", np.array(weighted_stddev), np.array(weighted_stddev2))
        #real_se = tf.math.reduce_std(weights*values) / np.sqrt(n_eff)
            # Return real standard error
        return np.array(weighted_mean), np.array(weighted_stddev), n_eff, {'value': np.array(weighted_mean), 'std_error': np.array(weighted_stddev), 'eff_n': n_eff, 'z_score': np.abs(np.array(weighted_mean))/(np.abs(np.array(weighted_stddev))+1e-10)}

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
    # # plt.suptitle('Yukawa Error Propagation: Analytical vs Monte Carlo')
    
    # rows, cols = physical_yukawas.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         plt.subplot(rows, cols, i*cols + j + 1)
            
    #         if is_complex:
    #             # Plot real and imaginary parts separately
    #             real_vals = np.real(all_physical_yukawas[:, i, j])
    #             imag_vals = np.imag(all_physical_yukawas[:, i, j])
                
    #             plt.hist(real_vals, bins=50, alpha=0.5, label='Real')
    #             plt.hist(imag_vals, bins=50, alpha=0.5, label='Imag')
                
    #             plt.axvline(np.real(physical_yukawas[i, j]), color='r', linestyle='--', 
    #                         label=f'Real Mean: {np.real(physical_yukawas[i, j]):.2e}')
    #             plt.axvline(np.imag(physical_yukawas[i, j]), color='g', linestyle='--', 
    #                         label=f'Imag Mean: {np.imag(physical_yukawas[i, j]):.2e}')
                
    #             plt.title(f'Y[{i},{j}], Ratio={np.real(ratio[i, j]):.2f}')
    #         else:
    #             plt.hist(all_physical_yukawas[:, i, j], bins=50)
    #             plt.axvline(physical_yukawas[i, j], color='r', linestyle='--', 
    #                         label=f'Mean: {physical_yukawas[i, j]:.2e}')
    #             plt.title(f'Y[{i},{j}], Ratio={ratio[i, j]:.2f}')
            
    #         if i == rows-1 and j == 0:
    #             plt.legend()
    
    # plt.tight_layout()
    # # Ensure data directory exists before saving
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
    NormQ_errors = 0.05 * np.abs(NormQ) + 1j*0.05 * np.abs(NormQ)
    
    NormU = np.array([
        [1.1 + 0.2j, 0.15 + 0.03j, 0.02 + 0.01j],
        [0.15 + 0.05j, 1.2 + 0.1j, 0.03 + 0.02j],
        [0.02 + 0.01j, 0.03 + 0.01j, 1.3 + 0.1j]
    ])
    NormU_errors = 0.05 * np.abs(NormU) + 1j*0.05 * np.abs(NormU)
    
    # Holomorphic Yukawa matrix
    m = np.array([
        [0.5 + 0.1j, 0.02 + 0.01j, 0.001 + 0.0005j],
        [0.02 + 0.01j, 0.4 + 0.15j, 0.03 + 0.01j],
        [0.001 + 0.0005j, 0.03 + 0.01j, 0.3 + 0.05j]
    ])
    m_errors = 0.03 * np.abs(m) + 1j*0.03 * np.abs(m)
    
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
    m_errors = 0.1 * np.abs(m) + 1j*0.1 * np.abs(m)  # 10% error
    
    # Import scipy for sqrtm function
    import scipy
    
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
def test_normalization_error_propagation_monte_carlo(Hmat, Hmat_errors, Qmat, Qmat_errors, Umat, Umat_errors, n_samples=10000):
    """
    Test error propagation through normalization matrices using Monte Carlo simulation.
    
    Parameters:
        Hmat, Qmat, Umat: Original matrices
        Hmat_errors, Qmat_errors, Umat_errors: Error matrices
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Dictionary with ratios between analytical and Monte Carlo errors
    """
    import scipy
    import numpy as np
    
    # Ensure all inputs are complex
    Hmat = np.asarray(Hmat, dtype=complex)
    Qmat = np.asarray(Qmat, dtype=complex)
    Umat = np.asarray(Umat, dtype=complex)
    Hmat_errors = np.asarray(Hmat_errors, dtype=complex)
    Qmat_errors = np.asarray(Qmat_errors, dtype=complex)
    Umat_errors = np.asarray(Umat_errors, dtype=complex)
    
    # Calculate analytical errors
    NormH_errors_analytical = propagate_errors_through_normalization(Hmat, Hmat_errors)
    NormQ_errors_analytical = propagate_errors_through_normalization(Qmat, Qmat_errors)
    NormU_errors_analytical = propagate_errors_through_normalization(Umat, Umat_errors)
    
    # Prepare arrays for Monte Carlo results
    NormH_samples = []
    NormQ_samples = []
    NormU_samples = []
    
    # Run Monte Carlo simulation
    for _ in range(n_samples):
        # Generate perturbed matrices with complex perturbations
        H_perturbed = Hmat.copy()
        Q_perturbed = Qmat.copy()
        U_perturbed = Umat.copy()
        
        # Add complex perturbations
        for matrix, errors in [(H_perturbed, Hmat_errors), (Q_perturbed, Qmat_errors), (U_perturbed, Umat_errors)]:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if errors[i, j] != 0:
                        real_error = np.random.normal(0, np.abs(np.real(errors[i, j])))
                        imag_error = np.random.normal(0, np.abs(np.imag(errors[i, j])))
                        matrix[i, j] += real_error + 1j * imag_error
        
        # Calculate normalization matrices for perturbed values
        NormH = np.linalg.inv(scipy.linalg.sqrtm(H_perturbed).astype(complex))
        NormQ = np.linalg.inv(scipy.linalg.sqrtm(Q_perturbed).astype(complex))
        NormU = np.linalg.inv(scipy.linalg.sqrtm(U_perturbed).astype(complex))
        
        NormH_samples.append(NormH)
        NormQ_samples.append(NormQ)
        NormU_samples.append(NormU)
    
    # Convert lists to numpy arrays
    NormH_samples = np.array(NormH_samples)
    NormQ_samples = np.array(NormQ_samples)
    NormU_samples = np.array(NormU_samples)
    
    # Calculate standard deviations from Monte Carlo samples (separately for real and imaginary parts)
    NormH_errors_mc_real = np.std(np.real(NormH_samples), axis=0, ddof=1)
    NormH_errors_mc_imag = np.std(np.imag(NormH_samples), axis=0, ddof=1)
    NormH_errors_mc = NormH_errors_mc_real + 1j * NormH_errors_mc_imag
    
    NormQ_errors_mc_real = np.std(np.real(NormQ_samples), axis=0, ddof=1)
    NormQ_errors_mc_imag = np.std(np.imag(NormQ_samples), axis=0, ddof=1)
    NormQ_errors_mc = NormQ_errors_mc_real + 1j * NormQ_errors_mc_imag
    
    NormU_errors_mc_real = np.std(np.real(NormU_samples), axis=0, ddof=1)
    NormU_errors_mc_imag = np.std(np.imag(NormU_samples), axis=0, ddof=1)
    NormU_errors_mc = NormU_errors_mc_real + 1j * NormU_errors_mc_imag
    
    # Calculate ratios between analytical and Monte Carlo errors (separately for real and imaginary parts)
    ratioH_real = np.abs(np.real(NormH_errors_analytical)) / NormH_errors_mc_real
    ratioH_imag = np.abs(np.imag(NormH_errors_analytical)) / NormH_errors_mc_imag
    
    ratioQ_real = np.abs(np.real(NormQ_errors_analytical)) / NormQ_errors_mc_real
    ratioQ_imag = np.abs(np.imag(NormQ_errors_analytical)) / NormQ_errors_mc_imag
    
    ratioU_real = np.abs(np.real(NormU_errors_analytical)) / NormU_errors_mc_real
    ratioU_imag = np.abs(np.imag(NormU_errors_analytical)) / NormU_errors_mc_imag
    
    # Check if all ratios are close to 1 (within tolerance)
    tolerance = 0.2  # 20% tolerance
    
    print("\nNormalization Error Propagation Validation:")
    print("Ratio (Analytical/Monte Carlo) for NormH (real part):", np.round(ratioH_real, 2))
    print("Ratio (Analytical/Monte Carlo) for NormH (imag part):", np.round(ratioH_imag, 2))
    print("Ratio (Analytical/Monte Carlo) for NormQ (real part):", np.round(ratioQ_real, 2))
    print("Ratio (Analytical/Monte Carlo) for NormQ (imag part):", np.round(ratioQ_imag, 2))
    print("Ratio (Analytical/Monte Carlo) for NormU (real part):", np.round(ratioU_real, 2))
    print("Ratio (Analytical/Monte Carlo) for NormU (imag part):", np.round(ratioU_imag, 2))
    
    validation_passed = (
        np.all(np.abs(ratioH_real - 1) < tolerance) and np.all(np.abs(ratioH_imag - 1) < tolerance) and
        np.all(np.abs(ratioQ_real - 1) < tolerance) and np.all(np.abs(ratioQ_imag - 1) < tolerance) and
        np.all(np.abs(ratioU_real - 1) < tolerance) and np.all(np.abs(ratioU_imag - 1) < tolerance)
    )
    
    if validation_passed:
        print("VALIDATION PASSED! ✅ Analytical errors match Monte Carlo within tolerance")
    else:
        print("VALIDATION WARNING! ⚠️ Some analytical errors differ from Monte Carlo")
        raise Exception("ERROR")
    
    return {
        'H_real': ratioH_real, 'H_imag': ratioH_imag,
        'Q_real': ratioQ_real, 'Q_imag': ratioQ_imag,
        'U_real': ratioU_real, 'U_imag': ratioU_imag
    }

# Test the normalization error propagation
if __name__ == "__main__":
    # Create test matrices with complex values
    H = np.array([[2.0]]) + 1j * np.array([[ 0.2]])
    Q = np.array([[1.8, 0.3], [0.3, 1.2]]) + 1j * np.array([[0.2, 0.4], [0.4, 0.1]])
    U = np.array([[1.5, 0.2], [0.2, 1.3]]) + 1j * np.array([[0.1, 0.3], [0.3, 0.2]])
    
    # Create complex error matrices (5% of absolute values for both real and imaginary parts)
    error_scale = 0.001 
    H_errors = error_scale * np.abs(H.real) + 1j * error_scale * np.abs(H.imag)
    Q_errors = error_scale * np.abs(Q.real) + 1j * error_scale * np.abs(Q.imag)
    U_errors = error_scale * np.abs(U.real) + 1j * error_scale * np.abs(U.imag)
    
    # Calculate analytical errors
    H_norm_errors = propagate_errors_through_normalization(H, H_errors)
    Q_norm_errors = propagate_errors_through_normalization(Q, Q_errors)
    U_norm_errors = propagate_errors_through_normalization(U, U_errors)
    
    print("Testing normalization error propagation with complex matrices...")
    print("Original complex matrices:")
    print("H =\n", H)
    print("Q =\n", Q)
    print("U =\n", U)
    
    print("\nComplex error matrices:")
    print("H_errors =\n", H_errors)
    print("Q_errors =\n", Q_errors)
    print("U_errors =\n", U_errors)
    
    print("\nAnalytical normalization errors:")
    print("H_norm_errors =\n", np.round(H_norm_errors, 5))
    print("Q_norm_errors =\n", np.round(Q_norm_errors, 5))
    print("U_norm_errors =\n", np.round(U_norm_errors, 5))
    
    # Validate with Monte Carlo
    validation_results = test_normalization_error_propagation_monte_carlo(
        H, H_errors, Q, Q_errors, U, U_errors, n_samples=10000)
