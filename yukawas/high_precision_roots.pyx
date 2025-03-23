# distutils: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
import mpmath
from mpmath import mp
import gmpy2
from gmpy2 import mpfr, mpc
import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial

# Enable 64-bit floating point precision for JAX
jax.config.update("jax_enable_x64", True)


# Set precision to approximately float128 (113 bits)
mp.dps = 34  # Decimal digits of precision

# Set precision to ~34 decimal digits
gmpy2.get_context().precision = 113  

def polynomial_value_gmpy2(coeffs, x):
    """Evaluate polynomial at x using gmpy2 precision"""
    cdef int n = len(coeffs)
    cdef int i
    
    # Ensure x is mpc type for consistent complex arithmetic
    if not isinstance(x, gmpy2.mpc):
        x = gmpy2.mpc(x.real if hasattr(x, 'real') else x, 
                      x.imag if hasattr(x, 'imag') else 0)
    
    # Always use complex for result to avoid type issues
    if hasattr(coeffs[0], 'imag'):
        result = gmpy2.mpc(coeffs[0].real, coeffs[0].imag)
    else:
        result = gmpy2.mpc(float(coeffs[0]), 0)

    for i in range(1, n):
        if hasattr(coeffs[i], 'imag'):
            coeff = gmpy2.mpc(coeffs[i].real, coeffs[i].imag)
        else:
            coeff = gmpy2.mpc(float(coeffs[i]), 0)
        result = result * x + coeff

    return result

def polynomial_derivative_gmpy2(coeffs, x):
    """Evaluate polynomial derivative at x using gmpy2 precision"""
    cdef int n = len(coeffs)
    cdef int i
    
    # Ensure x is mpc type for consistent complex arithmetic
    if not isinstance(x, gmpy2.mpc):
        x = gmpy2.mpc(x.real if hasattr(x, 'real') else x, 
                      x.imag if hasattr(x, 'imag') else 0)
    
    # For derivative, we need to multiply each coefficient by its degree
    # Note: coeffs are ordered with highest degree first
    if n <= 1:
        return gmpy2.mpc(0, 0)  # Constant polynomial has zero derivative
        
    result = gmpy2.mpc(0, 0)
    
    for i in range(n-1):
        degree = n - 1 - i  # Actual degree of the term
        
        if hasattr(coeffs[i], 'imag'):
            coeff = gmpy2.mpc(coeffs[i].real, coeffs[i].imag)
        else:
            coeff = gmpy2.mpc(float(coeffs[i]), 0)
        
        # Multiply by degree for derivative
        coeff = coeff * degree
        
        if i == 0:
            result = coeff
        else:
            result = result * x + coeff
            
    return result

def preconvert_coeffs_gmpy2(coeffs):
    """Pre-convert polynomial coefficients to gmpy2 types for better performance"""
    cdef list g_coeffs = []
    for c in coeffs:
        if hasattr(c, 'imag'):
            g_coeffs.append(gmpy2.mpc(c.real, c.imag))
        else:
            g_coeffs.append(gmpy2.mpc(float(c), 0))
    return g_coeffs
def newton_refine_root_gmpy2(coeffs, root, max_iter=50, tol=1e-20):
    """Refine a single root using Newton's method with gmpy2 precision"""
    cdef int i
    
    # Pre-convert coefficients to gmpy2 complex types
    cdef list g_coeffs = preconvert_coeffs_gmpy2(coeffs)
    
    # Set tolerance as gmpy2 type
    g_tol = gmpy2.mpfr(tol)
    
    # Convert root to gmpy2 complex type
    x = gmpy2.mpc(root.real if hasattr(root, 'real') else root, 
                  root.imag if hasattr(root, 'imag') else 0)
    
    for i in range(max_iter):
        p = polynomial_value_gmpy2(g_coeffs, x)
        dp = polynomial_derivative_gmpy2(g_coeffs, x)

        if abs(dp) < g_tol:  # Avoid division by near-zero
            break

        x_new = x - p / dp
        if abs(x_new - x) < g_tol:
            x = x_new
            break

        x = x_new

    # Always return complex values
    return complex(float(x.real), float(x.imag))

def refine_roots_gmpy2(coeffs, initial_roots, max_iter=50, tol=1e-20):
    """Refine all roots of a polynomial using Newton's method with gmpy2 precision"""
    refined_roots = []
    for root in initial_roots:
        refined = newton_refine_root_gmpy2(coeffs, root, max_iter, tol)
        refined_roots.append(refined)

    return refined_roots


def identify_complex_roots(roots, tol=1e-12):
    """Identify truly complex roots vs real roots with tiny imaginary parts"""
    result = []
    for root in roots:
        if isinstance(root, complex) and abs(root.imag) > tol:
            # Keep as complex
            result.append(root)
        elif isinstance(root, complex):
            # Convert to real if imaginary part is tiny
            result.append(root.real)
        else:
            result.append(root)
    return result

def find_complex_conjugate_pairs(roots, tol=1e-12):
    """Group complex roots into conjugate pairs to verify correctness"""
    pairs = []
    remaining = list(roots)
    
    while remaining:
        root = remaining.pop(0)
        if isinstance(root, complex):
            # Look for conjugate
            for i, other in enumerate(remaining):
                if isinstance(other, complex) and abs(root.real - other.real) < tol and abs(root.imag + other.imag) < tol:
                    pairs.append((root, other))
                    remaining.pop(i)
                    break
            else:
                # No conjugate found, orphan complex root
                pairs.append((root,))
        else:
            # Real root
            pairs.append((root,))
    
    return pairs



# Add JAX-based initial root finding
def jax_find_roots(coeffs):
    """Find roots using JAX's polynomial roots function"""
    return jnp.roots((coeffs), strip_zeros=False)

# Vectorized version for batches of polynomials
def jax_batch_find_roots(coeffs_batch):
    """Find roots for a batch of polynomials using JAX"""
    vectorized_roots = vmap(jax_find_roots)
    return vectorized_roots(jnp.array(coeffs_batch))

def batch_refine_roots_gmpy2(all_coeffs, all_initial_roots, max_iter =50, tol = 1e-20):
    """Refine batches of roots using gmpy2 high precision"""
    all_refined_roots = []
    
    for i, (coeffs, initial_roots) in enumerate(zip(all_coeffs, all_initial_roots)):
        refined = []
        for root in initial_roots:
            # Always use complex for consistency
            complex_root = complex(root) if not isinstance(root, complex) else root
            refined_root = newton_refine_root_gmpy2(coeffs, complex_root,max_iter= max_iter,tol = tol )
            refined.append(refined_root)
        all_refined_roots.append(refined)
    
    return all_refined_roots

def find_roots_vectorized(coeffs_batch, use_gmpy2=True, max_iter = 50, tol = 1e-20):
    """Find roots for multiple polynomials at once
    
    Args:
        coeffs_batch: List of polynomial coefficient arrays
        use_gmpy2: If True, refine with gmpy2 high precision
        
    Returns:
        List of root lists for each polynomial
    """
    # Get initial guesses using JAX (fast vectorized approach)
    initial_roots_batch = jax_batch_find_roots(coeffs_batch)
    
    # Convert JAX arrays to Python lists/complex numbers
    initial_roots_batch = jax.device_get(initial_roots_batch)
    
    if not use_gmpy2:
        # Just return the JAX results if high precision not needed
        return initial_roots_batch
    
    # Refine with high precisionnewton_refine_root_gmpy2
    return batch_refine_roots_gmpy2(coeffs_batch, initial_roots_batch, max_iter = max_iter)#,i initial_roots_batch
def batch_eval_polynomial_gmpy2(coeffs, roots):
    """Evaluate a polynomial at multiple roots using gmpy2 precision"""
    cdef int n = len(coeffs)
    cdef int i
    
    # Pre-convert coefficients to gmpy2 complex types
    cdef list g_coeffs = preconvert_coeffs_gmpy2(coeffs)
    cdef list g_roots = []
    
    # Convert roots to gmpy2 complex types
    for root in roots:
        if hasattr(root, 'imag'):
            g_roots.append(gmpy2.mpc(root.real, root.imag))
        else:
            g_roots.append(gmpy2.mpc(float(root), 0))
    
    # Evaluate polynomial at each root
    results = []
    for root in g_roots:
        p = polynomial_value_gmpy2(g_coeffs, root)
        results.append(complex(float(p.real), float(p.imag)))
    
    return results
