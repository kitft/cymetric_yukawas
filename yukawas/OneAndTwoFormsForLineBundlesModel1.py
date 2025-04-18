from cymetric.config import real_dtype, complex_dtype
import tensorflow as tf
import numpy as np


def get_coefficients_m1(free_coefficient, deformation = 'regular_deformation'):
    x = free_coefficient
    coefficients=       np.array([1., 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, \
            0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1], dtype=complex)
    if deformation == 'split_deformation':
        coefficients += np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
    elif deformation == 'regular_deformation':
        coefficients += np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
    else:
        raise ValueError("Invalid deformation specified")
    return coefficients


def compute_kappa(cpoints,n_ambientspace):
    return tf.reduce_sum(cpoints[:,(n_ambientspace-1)*2:(n_ambientspace)*2]*tf.math.conj(cpoints[:,(n_ambientspace-1)*2:(n_ambientspace)*2]),1)

def compute_dzi(cpoints,which_i):
    return tf.tensordot(cpoints[:,(which_i-1)*2],tf.cast(tf.eye(8)[(which_i-1)*2+1],complex_dtype),axes=0)-tf.tensordot(cpoints[:,(which_i-1)*2+1],tf.cast(tf.eye(8)[(which_i-1)*2],complex_dtype),axes=0)


def functionforbaseharmonicform_jbar_for_vH(cpoints):
    K3 = compute_kappa(cpoints,3)
    y0y1=tf.reduce_prod(cpoints[:,2:4],axis=-1)
    dz3b=tf.math.conj(compute_dzi(cpoints,3))
    return tf.einsum('x,xj->xj',K3**(-2)*y0y1,dz3b)

def functionforbaseharmonicform_jbar_for_vQ3(cpoints):
    K4 = compute_kappa(cpoints,4)
    polynomial=(cpoints[:,2]*cpoints[:,1] + cpoints[:,3]*cpoints[:,0])
    dz4b=tf.math.conj(compute_dzi(cpoints,4))
    return tf.einsum('x,xj->xj',K4**(-2)*polynomial,dz4b)

def functionforbaseharmonicform_jbar_for_vU3(cpoints):
    K4 = compute_kappa(cpoints,4)
    polynomial=(cpoints[:,2]*cpoints[:,1] - cpoints[:,3]*cpoints[:,0])
    dz4b=tf.math.conj(compute_dzi(cpoints,4))
    return tf.einsum('x,xj->xj',K4**(-2)*polynomial,dz4b)

functionforbaseharmonicform_jbar_for_vH.line_bundle = np.array([0,2,-2,0])
functionforbaseharmonicform_jbar_for_vQ3.line_bundle = np.array([1,1,0,-2])
functionforbaseharmonicform_jbar_for_vU3.line_bundle = np.array([1,1,0,-2])       



def getTypeIIs(cpoints,monomials,coefficients,formtype): 
    #Define the coordinates in a convenient way - already have the kappas
    x0, x1 = cpoints[:,0], cpoints[:,1]
    y0, y1 = cpoints[:,2], cpoints[:,3]
    dz2b=tf.math.conj(compute_dzi(cpoints,2))

    K1 = compute_kappa(cpoints,1)
    K2 = compute_kappa(cpoints,2)
    #Extract the parts of the defining polynomial P = p0 x0^2 + p1 x0 x1 + p2 x1^2

    # Create masks for each p value based on the second column of monomials
    mask_p2 = tf.equal(monomials[:, 1], 2)
    mask_p1 = tf.equal(monomials[:, 1], 1)
    mask_p0 = tf.equal(monomials[:, 1], 0)

    # Compute the products for all terms, omitting the terms on the first projective space.
    all_products = tf.einsum('i,xi->xi', coefficients, tf.reduce_prod(tf.pow(cpoints[:, tf.newaxis, 2:], monomials[tf.newaxis, :, 2:]), axis=-1))

    # Use the masks to separate the products into p0, p1, and p2
    p2 = tf.reduce_sum(tf.where(mask_p2, all_products, 0), axis=-1)
    p1 = tf.reduce_sum(tf.where(mask_p1, all_products, 0), axis=-1)
    p0 = tf.reduce_sum(tf.where(mask_p0, all_products, 0), axis=-1)
    
    #Now define the r0s and r1s for each form - Q = r0 \bar{x0} + r1 \bar{x1}
    if formtype== 'vQ1':
        r0Q1 = -tf.math.conj(y0**3)
        r1Q1 = tf.math.conj(y1**3)
        RQ1 = (-0.5) * (p1 * r0Q1 + p2* r1Q1) * x0 * tf.math.conj(x0**2) + p0 * r0Q1 * x0 * tf.math.conj(x1 * x0) + 0.5 * p0 * r1Q1 * x0 * tf.math.conj(x1**2) - 0.5 * p2 * r0Q1 * tf.math.conj(x0**2) * x1 - p2 * r1Q1  * tf.math.conj(x1 * x0) * x1 + 0.5 * (p0 * r0Q1 + p1 * r1Q1)* tf.math.conj(x1**2) * x1
        vQ1 = tf.einsum('xj,x->xj',tf.cast(dz2b,complex_dtype),K1**(-2) * K2**(-5) * RQ1)
        out=vQ1
    elif formtype==  'vQ2':
        r0Q2 = -tf.math.conj (y0 * (y1**2))
        r1Q2 = tf.math.conj(y0**2 * y1)
        RQ2 = (-0.5) * (p1 * r0Q2 + p2* r1Q2) * x0 * tf.math.conj(x0**2) + p0 * r0Q2 * x0 * tf.math.conj(x1 * x0) + 0.5 * p0 * r1Q2 * x0 * tf.math.conj(x1**2) - 0.5 * p2 * r0Q2 * tf.math.conj(x0**2) * x1 - p2 * r1Q2  * tf.math.conj(x1 * x0) * x1 + 0.5 * (p0 * r0Q2 + p1 * r1Q2)* tf.math.conj(x1**2) * x1
        vQ2 = tf.einsum('xj,x->xj',tf.cast(dz2b,complex_dtype),K1**(-2) * K2**(-5) * RQ2)
        out=vQ2
    elif formtype==  'vU1':
        r0U1 = tf.math.conj(y0**3)
        r1U1 = tf.math.conj(y1**3)
        RU1 = (-0.5) * (p1 * r0U1 + p2* r1U1) * x0 * tf.math.conj(x0**2) + p0 * r0U1 * x0 * tf.math.conj(x1 * x0) + 0.5 * p0 * r1U1 * x0 * tf.math.conj(x1**2) - 0.5 * p2 * r0U1 * tf.math.conj(x0**2) * x1 - p2 * r1U1  * tf.math.conj(x1 * x0) * x1 + 0.5 * (p0 * r0U1 + p1 * r1U1)* tf.math.conj(x1**2) * x1
        vU1 = tf.einsum('xj,x->xj',tf.cast(dz2b,complex_dtype),K1**(-2) * K2**(-5) * RU1)
        out=vU1
    elif formtype==  'vU2':
        r0U2 = tf.math.conj(y0 * (y1**2))
        r1U2 = tf.math.conj(y0**2 * y1)
        #Now using the above we can define the forms we need
        RU2 = (-0.5) * (p1 * r0U2 + p2* r1U2) * x0 * tf.math.conj(x0**2) + p0 * r0U2 * x0 * tf.math.conj(x1 * x0) + 0.5 * p0 * r1U2 * x0 * tf.math.conj(x1**2) - 0.5 * p2 * r0U2 * tf.math.conj(x0**2) * x1 - p2 * r1U2  * tf.math.conj(x1 * x0) * x1 + 0.5 * (p0 * r0U2 + p1 * r1U2)* tf.math.conj(x1**2) * x1 
        vU2 = tf.einsum('xj,x->xj',tf.cast(dz2b,complex_dtype),K1**(-2) * K2**(-5) * RU2)
        out=vU2
        
    return out
    
# Define line bundles for Type II forms
getTypeIIs.line_bundles = {
    'vQ1': np.array([-1,-3,2,2]),
    'vQ2': np.array([-1,-3,2,2]),
    'vU1': np.array([-1,-3,2,2]),
    'vU2': np.array([-1,-3,2,2])
}

def check_scaling_behavior(form_func, cpoints,formtype, line_bundle, lambda_val=0.5, print_detail=False):#+0.33333j):
    """
    Check that the form scales correctly according to its line bundle.
    
    Args:
        form_func: Function that computes the form given cpoints
        cpoints: Tensor of shape [batch, 8] representing 4 complex coordinates
        formtype: Type of form ('vQ1', 'vQ2', etc.)
        lambda_val: Scaling factor to test with
        
    Returns:
        Boolean tensor indicating whether the form scales correctly
    """
    # Get the line bundle for this form type
    # Compute the original form
    original_form = form_func(cpoints)
    
    # Test scaling for all four projective spaces (P1) - coordinates 0-7
    results = []
    
    for i in range(4):  # Loop through all 4 line bundles (0 to 3)
        scaled_cpoints = tf.identity(cpoints)
        
        # Scale the coordinates for the i-th projective space
        start_idx = 2*i
        end_idx = 2*i+2
        scaled_cpoints = tf.concat([
            scaled_cpoints[:, :start_idx],
            scaled_cpoints[:, start_idx:end_idx] * lambda_val,
            scaled_cpoints[:, end_idx:]
        ], axis=1)
        
        # print("scaled_cpoints",scaled_cpoints/cpoints)
        # Compute the form with scaled coordinates
        scaled_form = form_func(scaled_cpoints)
        
        # Expected scaling factor based on line bundle weight
        expected_scale = lambda_val ** line_bundle[i]
        extra_scale_multiply = tf.ones_like(cpoints)
        for j in range(4):
            start_idx = 2*j
            end_idx = 2*j+2
            # Create a tensor with the scaling factor and assign to the appropriate slice
            scale_factor = tf.ones_like(extra_scale_multiply[:,start_idx:end_idx]) *(lambda_val if i==j else 1)
            extra_scale_multiply = tf.concat([
                extra_scale_multiply[:,:start_idx],
                scale_factor,
                extra_scale_multiply[:,end_idx:]
            ], axis=1)

        #why is this * lambvda_val? because we have just (1,0,0) and it should be (dz,0,0)
        # Calculate actual scaling ratio (ignore zero values in original form)
        mask = tf.cast(tf.abs(original_form) > 1e-7, tf.bool)
        # Where the form is nonzero, multiply by lambda_val
        scaled_form = scaled_form * extra_scale_multiply
        all_ratios = tf.where(mask, scaled_form /(expected_scale * original_form), tf.zeros_like(scaled_form))
        # Only compute mean for non-zero values
        ratio = tf.reduce_mean(tf.boolean_mask(all_ratios, mask)) if tf.reduce_any(mask) else tf.constant(0.0, dtype=all_ratios.dtype)
        #print(f"{i} ratio",ratio, np.round(all_ratios.numpy(),2), "expected", np.round(expected_scale,2))
        
        # Check if the ratio is close to expected value
        tolerance = 3e-5
        scales_correctly = tf.abs(ratio - 1) < tolerance
        results.append(scales_correctly.numpy().item())
        if print_detail:
            print(f"P{i} ratio",ratio, np.round(all_ratios.numpy(),2))
    
    # All projective spaces must scale correctly
    #print("results",results)
    return tf.reduce_all(results), results

# Main execution block
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate some test points
    num_points = 10000
    test_points = tf.random.uniform([num_points, 16], minval=-1.0, maxval=1.0, dtype=tf.float32)
    complex_points = tf.complex(test_points[:, :8], test_points[:, 8:])
    
    # Define some test monomials and coefficients

    test_monomials = np.array([[2, 0, 2, 0, 2, 0, 2, 0], [2, 0, 2, 0, 2, 0, 1, 1], [2, 0, 2, 0, 2, 
      0, 0, 2], [2, 0, 2, 0, 1, 1, 2, 0], [2, 0, 2, 0, 1, 1, 1, 1], [2, 0,
       2, 0, 1, 1, 0, 2], [2, 0, 2, 0, 0, 2, 2, 0], [2, 0, 2, 0, 0, 2, 1, 
      1], [2, 0, 2, 0, 0, 2, 0, 2], [2, 0, 1, 1, 2, 0, 2, 0], [2, 0, 1, 1,
       2, 0, 1, 1], [2, 0, 1, 1, 2, 0, 0, 2], [2, 0, 1, 1, 1, 1, 2, 
      0], [2, 0, 1, 1, 1, 1, 1, 1], [2, 0, 1, 1, 1, 1, 0, 2], [2, 0, 1, 1,
       0, 2, 2, 0], [2, 0, 1, 1, 0, 2, 1, 1], [2, 0, 1, 1, 0, 2, 0, 
      2], [2, 0, 0, 2, 2, 0, 2, 0], [2, 0, 0, 2, 2, 0, 1, 1], [2, 0, 0, 2,
       2, 0, 0, 2], [2, 0, 0, 2, 1, 1, 2, 0], [2, 0, 0, 2, 1, 1, 1, 
      1], [2, 0, 0, 2, 1, 1, 0, 2], [2, 0, 0, 2, 0, 2, 2, 0], [2, 0, 0, 2,
       0, 2, 1, 1], [2, 0, 0, 2, 0, 2, 0, 2], [1, 1, 2, 0, 2, 0, 2, 
      0], [1, 1, 2, 0, 2, 0, 1, 1], [1, 1, 2, 0, 2, 0, 0, 2], [1, 1, 2, 0,
       1, 1, 2, 0], [1, 1, 2, 0, 1, 1, 1, 1], [1, 1, 2, 0, 1, 1, 0, 
      2], [1, 1, 2, 0, 0, 2, 2, 0], [1, 1, 2, 0, 0, 2, 1, 1], [1, 1, 2, 0,
       0, 2, 0, 2], [1, 1, 1, 1, 2, 0, 2, 0], [1, 1, 1, 1, 2, 0, 1, 
      1], [1, 1, 1, 1, 2, 0, 0, 2], [1, 1, 1, 1, 1, 1, 2, 0], [1, 1, 1, 1,
       1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 2], [1, 1, 1, 1, 0, 2, 2, 
      0], [1, 1, 1, 1, 0, 2, 1, 1], [1, 1, 1, 1, 0, 2, 0, 2], [1, 1, 0, 2,
       2, 0, 2, 0], [1, 1, 0, 2, 2, 0, 1, 1], [1, 1, 0, 2, 2, 0, 0, 
      2], [1, 1, 0, 2, 1, 1, 2, 0], [1, 1, 0, 2, 1, 1, 1, 1], [1, 1, 0, 2,
       1, 1, 0, 2], [1, 1, 0, 2, 0, 2, 2, 0], [1, 1, 0, 2, 0, 2, 1, 
      1], [1, 1, 0, 2, 0, 2, 0, 2], [0, 2, 2, 0, 2, 0, 2, 0], [0, 2, 2, 0,
       2, 0, 1, 1], [0, 2, 2, 0, 2, 0, 0, 2], [0, 2, 2, 0, 1, 1, 2, 
      0], [0, 2, 2, 0, 1, 1, 1, 1], [0, 2, 2, 0, 1, 1, 0, 2], [0, 2, 2, 0,
       0, 2, 2, 0], [0, 2, 2, 0, 0, 2, 1, 1], [0, 2, 2, 0, 0, 2, 0, 
      2], [0, 2, 1, 1, 2, 0, 2, 0], [0, 2, 1, 1, 2, 0, 1, 1], [0, 2, 1, 1,
       2, 0, 0, 2], [0, 2, 1, 1, 1, 1, 2, 0], [0, 2, 1, 1, 1, 1, 1, 
      1], [0, 2, 1, 1, 1, 1, 0, 2], [0, 2, 1, 1, 0, 2, 2, 0], [0, 2, 1, 1,
       0, 2, 1, 1], [0, 2, 1, 1, 0, 2, 0, 2], [0, 2, 0, 2, 2, 0, 2, 
      0], [0, 2, 0, 2, 2, 0, 1, 1], [0, 2, 0, 2, 2, 0, 0, 2], [0, 2, 0, 2,
       1, 1, 2, 0], [0, 2, 0, 2, 1, 1, 1, 1], [0, 2, 0, 2, 1, 1, 0, 
      2], [0, 2, 0, 2, 0, 2, 2, 0], [0, 2, 0, 2, 0, 2, 1, 1], [0, 2, 0, 2,
       0, 2, 0, 2]])
    test_coeffs = get_coefficients_m1(1.0)
    
    # Compute all form types
    form_types = ['vQ1', 'vQ2', 'vU1', 'vU2']
    results = {}
    
    # Verify scaling behavior
    print("\nVerifying scaling behavior...")
    for form_type in form_types:
        form_func = lambda x: getTypeIIs(x, test_monomials, test_coeffs, form_type)
        test_point = tf.expand_dims(complex_points[0], 0)
        line_bundle = getTypeIIs.line_bundles[form_type]
        is_correct, results = check_scaling_behavior(form_func, test_point, form_type, line_bundle)
        if not is_correct.numpy():
            print(f"{form_type} fails scaling test at point 0: {test_point}, results: {results}")
        else:
            print(f"{form_type} scales correctly: {is_correct.numpy()} with line bundle {line_bundle}")

    for form_func in [functionforbaseharmonicform_jbar_for_vH,functionforbaseharmonicform_jbar_for_vU3,functionforbaseharmonicform_jbar_for_vQ3]:
        test_point = tf.expand_dims(complex_points[0], 0)
        line_bundle = form_func.line_bundle
        is_correct, results = check_scaling_behavior(form_func, test_point, 'vH', line_bundle)      
        if not is_correct.numpy():
            print(f"{form_func.__name__} fails scaling test at point 0: {test_point}, results: {results}")
        else:
            print(f"{form_func.__name__} scales correctly: {is_correct.numpy()} with line bundle {line_bundle}")
        


    # Test scaling behavior for all points in the dataset
    print("\nVerifying scaling behavior for all points in the dataset...")
    for form_type in form_types:
        form_func = lambda x: getTypeIIs(x, test_monomials, test_coeffs, form_type)
        line_bundle = getTypeIIs.line_bundles[form_type]
        all_correct = True
        for i, point in enumerate(complex_points):
            test_point = tf.expand_dims(point, 0)
            is_correct, results = check_scaling_behavior(form_func, test_point, form_type, line_bundle)
            if not is_correct.numpy():
                all_correct = False
                print(f"{form_type} fails scaling test at point {i}: {point}, results: {results}")
                
                break
        print(f"{form_type} scales correctly for all points: {all_correct}")

    for form_func in [functionforbaseharmonicform_jbar_for_vH, functionforbaseharmonicform_jbar_for_vQ3, functionforbaseharmonicform_jbar_for_vU3]:
        line_bundle = form_func.line_bundle
        all_correct = True
        for i, point in enumerate(complex_points):
            test_point = tf.expand_dims(point, 0)
            is_correct, results = check_scaling_behavior(form_func, test_point, 'vH', line_bundle)
            if not is_correct.numpy():
                all_correct = False
                print(f"{form_func.__name__} fails scaling test at point {i}: {point}, results: {results}")
                break
        print(f"{form_func.__name__} scales correctly for all points: {all_correct}")

    # Analyze the problematic forms in detail
    print("\nDetailed analysis of problematic forms...")
    
    # # Analyze specific points that failed in previous tests
    # print("\nAnalyzing specific points that failed scaling tests:")
    
    # # Point that failed for vQ1
    # vQ1_point = tf.constant([[-0.9065864 -0.5898986j, -0.7647636 +0.53426766j, 0.57997894-0.16393948j,
    #                          -0.4371791 -0.3024733j, -0.6511016 +0.88802576j, 0.9978664 +0.60525656j,
    #                          -0.53630257-0.7942703j, 0.809108 -0.8930712j]], dtype=complex_dtype)
    
    # # Point that failed for vQ2
    # vQ2_point = tf.constant([[-0.15478611-0.5467601j, 0.27424073-0.03628635j, -0.15076947-0.1920898j,
    #                          -0.67389965-0.84321976j, -0.5648675 -0.3534844j, -0.31287456+0.65742064j,
    #                          0.65165234-0.9621804j, -0.19137716+0.1281333j]], dtype=complex_dtype)
    
    # # Point that failed for vU1
    # vU1_point = tf.constant([[-0.60094166-0.24303031j, 0.3669095 -0.9506347j, 0.90168095-0.9847765j,
    #                          0.09522223+0.3844216j, -0.750407 -0.8682096j, -0.8162694 +0.91065526j,
    #                          0.5152743 +0.15132546j, -0.57137084+0.17416167j]], dtype=complex_dtype)
    
    # # Point that failed for vU2
    # vU2_point = tf.constant([[ 0.99127626-0.64768267j, 0.67008376+0.9108846j, 0.8626993 +0.4262185j,
    #                          -0.4359653 +0.91390395j, -0.11907673+0.67890096j, -0.7594247 +0.31995487j,
    #                          0.03038573-0.5820801j, -0.97986627+0.09696555j]], dtype=complex_dtype)
    
    # # Analyze vQ1
    # print("\n=== Detailed analysis for vQ1 ===")
    # vQ1_func = lambda x: getTypeIIs(x, test_monomials, test_coeffs, 'vQ1')
    # line_bundle = getTypeIIs.line_bundles['vQ1']
    # is_correct, results = check_scaling_behavior(vQ1_func, vQ1_point, 'vQ1', line_bundle, print_detail=True)
    # print(f"vQ1 scales correctly: {is_correct.numpy()}, results by coordinate: {results}")
    
    # # Analyze vQ2
    # print("\n=== Detailed analysis for vQ2 ===")
    # vQ2_func = lambda x: getTypeIIs(x, test_monomials, test_coeffs, 'vQ2')
    # line_bundle = getTypeIIs.line_bundles['vQ2']
    # is_correct, results = check_scaling_behavior(vQ2_func, vQ2_point, 'vQ2', line_bundle, print_detail=True)
    # print(f"vQ2 scales correctly: {is_correct.numpy()}, results by coordinate: {results}")
    
    # # Analyze vU1
    # print("\n=== Detailed analysis for vU1 ===")
    # vU1_func = lambda x: getTypeIIs(x, test_monomials, test_coeffs, 'vU1')
    # line_bundle = getTypeIIs.line_bundles['vU1']
    # is_correct, results = check_scaling_behavior(vU1_func, vU1_point, 'vU1', line_bundle, print_detail=True)
    # print(f"vU1 scales correctly: {is_correct.numpy()}, results by coordinate: {results}")
    
    # # Analyze vU2
    # print("\n=== Detailed analysis for vU2 ===")
    # vU2_func = lambda x: getTypeIIs(x, test_monomials, test_coeffs, 'vU2')
    # line_bundle = getTypeIIs.line_bundles['vU2']
    # is_correct, results = check_scaling_behavior(vU2_func, vU2_point, 'vU2', line_bundle, print_detail=True)
    # print(f"vU2 scales correctly: {is_correct.numpy()}, results by coordinate: {results}")