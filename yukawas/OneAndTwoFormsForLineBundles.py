from cymetric.config import real_dtype, complex_dtype
import tensorflow as tf
import numpy as np

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

    # p0,p1,p2 = [],[],[]
    # for term in range(len(coefficients)):
    #     if coefficients[term] != 0:
    #         if monomials[term,1] == 2:
    #             p2.append(coefficients[term] * tf.reduce_prod(tf.pow(cpoints, monomials[term]), axis=-1))
    #         elif monomials[term,1] == 1:
    #             p1.append(coefficients[term] * tf.reduce_prod(tf.pow(cpoints, monomials[term]), axis=-1))
    #         else:
    #             p0.append(coefficients[term] * tf.reduce_prod(tf.pow(cpoints, monomials[term]), axis=-1))
    
    # p0 = tf.reduce_sum(p0, axis=0)
    # p1 = tf.reduce_sum(p1, axis=0)
    # p2 = tf.reduce_sum(p2, axis=0)

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
    #p0,p1,p2 = sum(p0),sum(p1),sum(p2)
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

#getTypeIIs(point_vec_to_complex(databeta_val_dict['X_val']),phimodel1,'vQ1')
#lambda x: getTypeIIs(x,PHIMODELPICK,'vQ1')
#lambda x: getTypeIIs(x,PHIMODELPICK,'vQ2')
#lambda x: getTypeIIs(x,PHIMODELPICK,'vU1')
#lambda x: getTypeIIs(x,PHIMODELPICK,'vU2')
