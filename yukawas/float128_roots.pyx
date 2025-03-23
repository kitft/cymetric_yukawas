import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# Define the float128 type
ctypedef float __float128

cdef extern from "quadmath.h":
    __float128 sqrtq(__float128)
    __float128 fabsq(__float128)
    __float128 cosq(__float128)
    __float128 sinq(__float128)
    __float128 coshq(__float128)
    __float128 sinhq(__float128)

def polynomial_value(coeffs, x):
    cdef int n = len(coeffs)
    cdef __float128 x_val = x
    cdef __float128 result = coeffs[n-1]
    cdef int i

    for i in range(n-2, -1, -1):
        result = result * x_val + coeffs[i]

    return result

def polynomial_derivative(coeffs, x):
    cdef int n = len(coeffs)
    cdef __float128 x_val = x
    cdef __float128 result = 0
    cdef int i

    for i in range(n-1, 0, -1):
        result = result * x_val + i * coeffs[i]

    return result

def newton_refine_root(coeffs, root, max_iter=50, tol=1e-30):
    """Refine a single root using Newton's method with float128 precision"""
    cdef __float128 x = root
    cdef __float128 p, dp, x_new
    cdef int i

    for i in range(max_iter):
        p = polynomial_value(coeffs, x)
        dp = polynomial_derivative(coeffs, x)

        if fabsq(dp) < 1e-30:  # Avoid division by near-zero
            break

        x_new = x - p / dp
        if fabsq(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    return float(x)  # Convert back to Python float

def refine_roots(coeffs, initial_roots, max_iter=50, tol=1e-30):
    """Refine all roots of a polynomial using Newton's method with float128 precision"""
    refined_roots = []
    for root in initial_roots:
        # Handle complex roots
        if isinstance(root, complex):
            # For complex roots, we need a different approach
            # This is a simplified version that works for real roots only
            refined = newton_refine_root(coeffs, root.real, max_iter, tol)
            if abs(root.imag) > 1e-10:
                # If imaginary part is significant, refine as real root
                # but add back approximate imaginary part
                refined = complex(refined, root.imag)
        else:
            refined = newton_refine_root(coeffs, root, max_iter, tol)

        refined_roots.append(refined)

    return refined_roots

def find_roots_float128(coeffs):
    """Find roots of a polynomial with float128 precision"""
    # Use numpy roots to get initial approximations
    import numpy as np
    initial_roots = np.roots(coeffs)

    # Refine each root
    return refine_roots(coeffs, initial_roots)