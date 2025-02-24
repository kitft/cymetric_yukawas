from cymetric.config import real_dtype, complex_dtype
import tensorflow as tf

def convertcomptoreal(complexvec):
    # this converts from complex to real
    return tf.concat([tf.math.real(complexvec),tf.math.imag(complexvec)],-1) 

def point_vec_to_real(complexvec):
    # this converts from complex to real
    return tf.concat([tf.math.real(complexvec),tf.math.imag(complexvec)],-1) 
def point_vec_to_complex(p):
    plen = tf.shape(p)[-1] // 2
    return tf.complex(p[..., :plen], p[..., plen:])

@tf.function
def laplacian(betamodel,points,pullbacks,invmetrics):
    ncoords = tf.shape(points[0])[-1] // 2 
    with tf.GradientTape(persistent=False) as tape1:#why persistent?
        tape1.watch(points)
        with tf.GradientTape(persistent=False) as tape2:
            tape2.watch(points)
            # Need to disable training here, because batch norm
            # and dropout mix the batches, such that batch_jacobian
            # is no longer reliable.
            phi = betamodel(points,training=False)
        d_phi = tape2.gradient(phi, points)# the derivative index is inserted at the (1) index, just after the batch index
    dd_phi = tape1.batch_jacobian(d_phi, points) # the derivative index is inserted at the (1) index again, so now we have the structure xab
    dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = \
        0.25*dd_phi[:, :ncoords, :ncoords], \
        0.25*dd_phi[:, ncoords:, :ncoords], \
        0.25*dd_phi[:, :ncoords, ncoords:], \
        0.25*dd_phi[:, ncoords:, ncoords:]
    dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)# this should be d_dbar
    # this second imaginary part is a vector equation, so whilst the result is hermitian it is not necessarily real?
    # comes from df/dz = f_x -i f_y/2. Do it twice in the correct order!
    # the result is dy_dx_phi has the y index first, then the x index
    # so the resuklt has the holo d index first, and the antiholo index second!!
    #This is implemented correctly below? Hopefully

    #check |z|^2 = (x+iy)(x-iy) = x^2 +y^2, d/dz dzbar is 1? Or 1/4(2+2)+i*0 = 1. So this works. First index is d/dz, second is d/dy
    #try z^2, ddbar_ (x^2+2ixy -y^2), dz dzbar = 0, 1/4(2-2)+i1/4(2-2 = 0)
    #factor of 2 as the laplacian CY =  2g_CY^abbar ∂_a∂_(b),
    # note that invmetric looks like g^(b)a not g^(a)b. Actually needs a transpose. ddbar_phi has indices
    #j_elim (tf.tensor([bSize, nHyper], tf.int64), optional):
                    #Coordinates(s) to be eliminated in the pullbacks.
                    #If None will take max(dQ/dz). Defaults to None.
                    #PULLBACKS SHOULD BE GIVEN WITH THIS? Or decide I want to use none?
    #factor of 2 because the laplacian is 2g^ab da db 2gCY∂a∂ ̄b,. pb_dd_phi_Pbbar is just
    #no, ditch the factor of two!
    gdd_phi = tf.einsum('xba,xai,xji,xbj->x', invmetrics,pullbacks, dd_phi, tf.math.conj(pullbacks))
    return gdd_phi


#@tf.function
#def laplacianWithH(sigmamodel,points,pullbacks,invmetrics,Hfunc):
#    ncoords = tf.shape(points[0])[-1] // 2
#    with tf.GradientTape(persistent=True) as tape1:
#        tape1.watch(points)
#        with tf.GradientTape(persistent=True) as tape2:
#            tape2.watch(points)
#            # Need to disable training here, because batch norm
#            # and dropout mix the batches, such that batch_jacobian
#            # is no longer reliable.
#            phi = sigmamodel(points,training=False)#sigma is complex! here
#            #real_part = tf.math.real(phi)
#            real_part = tf.math.real(phi)
#            #real_part=phi
#            imag_part = tf.math.imag(phi)
#
#            # Stack them along a new dimension
#            #phireal = tf.stack([real_part, imag_part], axis=-1)
#            #print('phi')
#            #tf.print(phi)
#        d_phiR = tape2.gradient(real_part, points)
#        #d_phiI = d_phiR#tape2.gradient(imag_part, points)
#        d_phiI = tape2.gradient(imag_part, points)
#        del tape2
#        #d_phireal = tf.stack([d_phiR, d_phiI], axis=-2)#add to second last axis
#        #print('dphi')
#        #tf.print(d_phi)
#        #dphiH=tf.einsum('xQa,x->xQa',d_phireal,Hfunc(points))#hfunc is real, so can just multiply
#
#
#        Hs = Hfunc(points,training=False)#added training false - check if this owrksj:
#        dphiH_R=tf.einsum('xa,x->xa',d_phiR,Hs)#hfunc is real, so can just multiply
#        dphiH_I=tf.einsum('xa,x->xa',d_phiI,Hs)#hfunc is real, so can just multiply
#    #dd_phi_R = tape1.batch_jacobian(dphiH_R, points)
#    dd_phi_R = tape1.batch_jacobian(dphiH_R, points)
#    dd_phi_I = tape1.batch_jacobian(dphiH_I, points)
#    del tape1
#    #dd_phi= tf.stack([dd_phi_R, dd_phi_I], axis=-3)#add to thirdlast axis
#
#    #print('ddphi')
#    #tf.print(dd_phi)
#    # Note - these are auxiliary xs and ys. They are not the same as the z = 1/sqrt[2] (x+iy) defined in the note
#    #the second derivative is added to the end... so this ordering is now correct
#    r_dx_Hdx_phi, r_dx_Hdy_phi, r_dy_Hdx_phi, r_dy_Hdy_phi = \
#        0.25*dd_phi_R[:,:ncoords, :ncoords], \
#        0.25*dd_phi_R[:,ncoords:, :ncoords], \
#        0.25*dd_phi_R[:,:ncoords, ncoords:], \
#        0.25*dd_phi_R[:,ncoords:, ncoords:]
#    i_dx_Hdx_phi, i_dx_Hdy_phi, i_dy_Hdx_phi, i_dy_Hdy_phi = \
#        0.25*dd_phi_I[:, :ncoords, :ncoords], \
#        0.25*dd_phi_I[:, ncoords:, :ncoords], \
#        0.25*dd_phi_I[:, :ncoords, ncoords:], \
#        0.25*dd_phi_I[:, ncoords:, ncoords:]
#
#    #i_dx_Hdx_phi, i_dx_Hdy_phi, i_dy_Hdx_phi, i_dy_Hdy_phi = r_dx_Hdx_phi, r_dx_Hdy_phi, r_dy_Hdx_phi, r_dy_Hdy_phi 
#    #conventionally, dxdy means dy first then dx. Only matters for middle two, but ncoords: means y is first
#    #r_dx_Hdx_phi = dx_Hdx_phi[:,0]
#    #r_dx_Hdy_phi = dx_Hdy_phi[:,0]
#    #r_dy_Hdx_phi = dy_Hdx_phi[:,0]
#    #r_dy_Hdy_phi = dy_Hdy_phi[:,0]
#    #i_dx_Hdx_phi = dx_Hdx_phi[:,1]
#    #i_dx_Hdy_phi = dx_Hdy_phi[:,1]
#    #i_dy_Hdx_phi = dy_Hdx_phi[:,1]
#    #i_dy_Hdy_phi = dy_Hdy_phi[:,1]
#    dd_phi = tf.complex(r_dx_Hdx_phi + r_dy_Hdy_phi, r_dx_Hdy_phi - r_dy_Hdx_phi)+ 1j*tf.complex(i_dx_Hdx_phi + i_dy_Hdy_phi, i_dx_Hdy_phi - i_dy_Hdx_phi)
#    #dd_phi = tf.complex(r_dx_Hdx_phi + r_dy_Hdy_phi,0.)# r_dx_Hdy_phi - r_dy_Hdx_phi)+ 1j*tf.complex(i_dx_Hdx_phi + i_dy_Hdy_phi, i_dx_Hdy_phi - i_dy_Hdx_phi)
#    #dd_phi = tf.complex(r_dy_Hdy_phi,0.)# r_dx_Hdy_phi - r_dy_Hdx_phi)+ 1j*tf.complex(i_dx_Hdx_phi + i_dy_Hdy_phi, i_dx_Hdy_phi - i_dy_Hdx_phi)
#    # this second imaginary part is a vector equation, so whilst the result is hermitian it is not necessarily real?
#    #re = dx_dx_phi + dy_dy_phi
#    #im = dx_dy_phi - dy_dx_phi
#    #print("re/im")
#    #tf.print(im)
#    ##tf.print(re)
#    #dd_phi = tf.complex(tf.math.real(re)-tf.math.imag(im), tf.math.real(im)+tf.math.imag(re))# this should be d_dbar
#    #check |z|^2 = (x+iy)(x-iy) = x^2 +y^2, d/dz dzbar is 1? Or 1/4(2+2)+i*0 = 1. So this works. First index is d/dz, second is d/dy
#    #try z^2, ddbar_ (x^2+2ixy -y^2), dz dzbar = 0, 1/4(2-2)+i1/4(2-2 = 0)
#    #factor of 2 as the laplacian CY =  2g_CY^abbar ∂_a∂_(b),
#    # note that invmetric looks like g^(b)a not g^(a)b. Actually needs a transpose. ddbar_phi has indices
#    #j_elim (tf.tensor([bSize, nHyper], tf.int64), optional):
#                    #Coordinates(s) to be eliminated in the pullbacks.
#                    #If None will take max(dQ/dz). Defaults to None.
#                    #PULLBACKS SHOULD BE GIVEN WITH THIS? Or decide I want to use none?
#    #factor of 2 because the laplacian is 2g^ab da db 2gCY∂a∂ ̄b,. pb_dd_phi_Pbbar is just
#    gdd_phi = tf.einsum('xba,xai,xji,xbj->x', invmetrics,pullbacks, dd_phi, tf.math.conj(pullbacks))
#    #gdd_phi = tf.einsum('xai,xji,xbj->xab', pullbacks, dd_phi, tf.math.conj(pullbacks))
#    return gdd_phi

@tf.function
def laplacianWithH(sigmamodel,points,pullbacks,invmetrics,Hfunc,training=False):
    ncoords = tf.shape(points[0])[-1] // 2 
    with tf.GradientTape(persistent=False) as tape1:
        tape1.watch(points)
        with tf.GradientTape(persistent=False) as tape2:
            tape2.watch(points)
            # Need to disable training here, because batch norm
            # and dropout mix the batches, such that batch_jacobian
            # is no longer reliable.
            phi = sigmamodel(points,training=training)#sigma is complex! here
            real_part = tf.math.real(phi)
            imag_part = tf.math.imag(phi)

            # Stack them along a new dimension
            phireal = tf.stack([real_part, imag_part], axis=-1)
            #print('phi')
            #tf.print(phi)
        d_phi = tape2.batch_jacobian(phireal, points)
        #print('dphi')
        #tf.print(d_phi)
        dphiH=tf.einsum('xQa,x->xQa',d_phi,Hfunc(points))#hfunc is real, so can just multiply
    dd_phi = tape1.batch_jacobian(dphiH, points)

    #print('ddphi')
    #tf.print(dd_phi)
    # Note - these are auxiliary xs and ys. They are not the same as the z = 1/sqrt[2] (x+iy) defined in the note
    #the second derivative is added to the end... so this ordering is now correct
    dx_Hdx_phi, dx_Hdy_phi, dy_Hdx_phi, dy_Hdy_phi = \
        0.25*dd_phi[:,:, :ncoords, :ncoords], \
        0.25*dd_phi[:,:, ncoords:, :ncoords], \
        0.25*dd_phi[:,:, :ncoords, ncoords:], \
        0.25*dd_phi[:,:, ncoords:, ncoords:]
    #conventionally, dxdy means dy first then dx. Only matters for middle two, but ncoords: means y is first
    r_dx_Hdx_phi = dx_Hdx_phi[:,0]
    r_dx_Hdy_phi = dx_Hdy_phi[:,0]
    r_dy_Hdx_phi = dy_Hdx_phi[:,0]
    r_dy_Hdy_phi = dy_Hdy_phi[:,0]
    i_dx_Hdx_phi = dx_Hdx_phi[:,1]
    i_dx_Hdy_phi = dx_Hdy_phi[:,1]
    i_dy_Hdx_phi = dy_Hdx_phi[:,1]
    i_dy_Hdy_phi = dy_Hdy_phi[:,1]
    dd_phi = tf.complex(r_dx_Hdx_phi + r_dy_Hdy_phi, r_dx_Hdy_phi - r_dy_Hdx_phi)+ 1j*tf.complex(i_dx_Hdx_phi + i_dy_Hdy_phi, i_dx_Hdy_phi - i_dy_Hdx_phi)
    #dd_phi = tf.complex(r_dx_Hdx_phi + r_dy_Hdy_phi,0.)# r_dx_Hdy_phi - r_dy_Hdx_phi)+ 1j*tf.complex(i_dx_Hdx_phi + i_dy_Hdy_phi, i_dx_Hdy_phi - i_dy_Hdx_phi)
    #dd_phi = tf.complex(r_dy_Hdy_phi,0.)# r_dx_Hdy_phi - r_dy_Hdx_phi)+ 1j*tf.complex(i_dx_Hdx_phi + i_dy_Hdy_phi, i_dx_Hdy_phi - i_dy_Hdx_phi)
    # this second imaginary part is a vector equation, so whilst the result is hermitian it is not necessarily real?
    #re = dx_dx_phi + dy_dy_phi
    #im = dx_dy_phi - dy_dx_phi
    #print("re/im")
    #tf.print(im)
    ##tf.print(re)
    #dd_phi = tf.complex(tf.math.real(re)-tf.math.imag(im), tf.math.real(im)+tf.math.imag(re))# this should be d_dbar
    #check |z|^2 = (x+iy)(x-iy) = x^2 +y^2, d/dz dzbar is 1? Or 1/4(2+2)+i*0 = 1. So this works. First index is d/dz, second is d/dy
    #try z^2, ddbar_ (x^2+2ixy -y^2), dz dzbar = 0, 1/4(2-2)+i1/4(2-2 = 0)
    #factor of 2 as the laplacian CY =  2g_CY^abbar ∂_a∂_(b),
    # note that invmetric looks like g^(b)a not g^(a)b. Actually needs a transpose. ddbar_phi has indices
    #j_elim (tf.tensor([bSize, nHyper], tf.int64), optional):
                    #Coordinates(s) to be eliminated in the pullbacks.
                    #If None will take max(dQ/dz). Defaults to None.
                    #PULLBACKS SHOULD BE GIVEN WITH THIS? Or decide I want to use none?
    #factor of 2 because the laplacian is 2g^ab da db 2gCY∂a∂ ̄b,. pb_dd_phi_Pbbar is just
    gdd_phi = tf.einsum('xba,xai,xji,xbj->x', invmetrics,pullbacks, dd_phi, tf.math.conj(pullbacks))

    #gdd_phi = tf.einsum('xai,xji,xbj->xab', pullbacks, dd_phi, tf.math.conj(pullbacks))
    return gdd_phi

def coclosure_check(points,HYMmetric,harmonicform_jbar,sigma,invmetric,pullbacks):
    ncoords = tf.shape(points[0])[-1] // 2 
    pointstensor=points#tf.constant(points)
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(pointstensor)
        cpoints=point_vec_to_complex(pointstensor)
        dbarsigma = extder_jbar_for_sigma(pointstensor,sigma)
        HNu=tf.einsum('x,xb->xb',tf.cast(HYMmetric(pointstensor),complex_dtype),tf.cast(harmonicform_jbar(tf.cast(cpoints,complex_dtype)) + dbarsigma,complex_dtype))#complexpoints vs noncomplex
        real_part = tf.math.real(HNu)
        imag_part = tf.math.imag(HNu)

        # Stack them along a new dimension
        Hnustack = tf.stack([real_part, imag_part], axis=1)# put at the 0 position
        dHnu = tape1.batch_jacobian(Hnustack, pointstensor)
    dx_Hnu, dy_Hnu = \
        0.5*dHnu[:,:,:, :ncoords], \
        0.5*dHnu[:,:,:, ncoords:]
        # this should be holo derivative on the second index
    #dz_Hnu = tf.complex(tf.math.real(dx_Hnu)+tf.math.imag(dy_Hnu),tf.math.imag(dx_Hnu)-tf.math.real(dy_Hnu)) #note that the (holo) derivative index is the second index (the last index, as seen from above)
    dz_Hnu = tf.complex(dx_Hnu[:,0]+dy_Hnu[:,1],dx_Hnu[:,1]-dy_Hnu[:,0]) #note that the (holo) derivative index is the second index (the last index, as seen from above)
    #dz_Hnu = tf.complex(tf.math.real(dx_Hnu)+tf.math.imag(dy_Hnu),0.)#tf.math.imag(dx_Hnu)-tf.math.real(dy_Hnu)) #note that the (holo) derivative index is the second index (the last index, as seen from above)
    #print("dHnu")
    #print(dHnu)
    #tf.print(dy_Hnu)
    #dz_Hnu = tf.complex(tf.math.imag(dy_Hnu),0.)#tf.math.imag(dx_Hnu)-tf.math.real(dy_Hnu)) #note that the (holo) derivative index is the second index (the last index, as seen from above)
        #check - e.g.  Hnu might be x+iy = z. We want the derivative to be 1.
        # so take Hnu = R + iY
        # now do: dxR + dyI + i(dxI-dyR), gives 1/2 + 1/2  +0 + 0= 1
        # same for iz, now get 0 + 0 + i( 1/2 -(-1/2)) =i 
        # note that the pullbacks can come afterwards, as the holomorphic derivative should just pass straight through the conjugated pullback
    return tf.einsum('xba,xbj,xai,xji->x',invmetric,tf.math.conj(pullbacks),pullbacks,dz_Hnu) #the barred index is first, and the derivative index is second! Same is true, in fact, for the inverse metric
    #return dz_Hnu#tf.einsum('xbj,xai,xji->xab',tf.math.conj(pullbacks),pullbacks,dz_Hnu) #the barred index is first, and the derivative index is second! Same is true, in fact, for the inverse metric

def closure_check(points,harmonicform_jbar,sigma,pullbacks):
    #break
    ncoords = tf.shape(points[0])[-1] // 2 
    pointstensor=points#tf.constant(points)
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(pointstensor)
        cpoints=point_vec_to_complex(pointstensor)
        dbarsigma = extder_jbar_for_sigma(pointstensor,sigma)
        Nu=tf.cast(harmonicform_jbar(tf.cast(cpoints,complex_dtype)) + dbarsigma,complex_dtype)#complexpoints vs noncomplex
        real_part = tf.math.real(Nu)
        imag_part = tf.math.imag(Nu)

        # Stack them along a new dimension
        Hnustack = tf.stack([real_part, imag_part], axis=1)# put at the 0 position
        dHnu = tape1.batch_jacobian(Hnustack, pointstensor)
    dx_Hnu, dy_Hnu = \
        0.5*dHnu[:,:,:, :ncoords], \
        0.5*dHnu[:,:,:, ncoords:]
        # this should be holo derivative on the second index
    #dz_Hnu = tf.complex(tf.math.real(dx_Hnu)+tf.math.imag(dy_Hnu),tf.math.imag(dx_Hnu)-tf.math.real(dy_Hnu)) #note that the (holo) derivative index is the second index (the last index, as seen from above)
    dz_Hnu = tf.complex(dx_Hnu[:,0]-dy_Hnu[:,1],dx_Hnu[:,1]+dy_Hnu[:,0]) #note that the (holo) derivative index is the second index (the last index, as seen from above)
    #dz_Hnu = tf.complex(tf.math.real(dx_Hnu)+tf.math.imag(dy_Hnu),0.)#tf.math.imag(dx_Hnu)-tf.math.real(dy_Hnu)) #note that the (holo) derivative index is the second index (the last index, as seen from above)
    #print("dHnu")
    #print(dHnu)
    #tf.print(dy_Hnu)
    #dz_Hnu = tf.complex(tf.math.imag(dy_Hnu),0.)#tf.math.imag(dx_Hnu)-tf.math.real(dy_Hnu)) #note that the (holo) derivative index is the second index (the last index, as seen from above)
        #check - e.g.  Hnu might be x+iy = z. We want the derivative to be 1.
        # so take Hnu = R + iY
        # now do: dxR + dyI + i(dxI-dyR), gives 1/2 + 1/2  +0 + 0= 1
        # same for iz, now get 0 + 0 + i( 1/2 -(-1/2)) =i 
        # note that the pullbacks can come afterwards, as the holomorphic derivative should just pass straight through the conjugated pullback
    outval = tf.einsum('xbj,xck,xkj->xcb',tf.math.conj(pullbacks),tf.math.conj(pullbacks),dz_Hnu) #the barred index is first, and the derivative index is second! Same is true, in fact, for the inverse metric
    return outval - tf.einsum('xab->xba',outval)
    #return dz_Hnu#tf.einsum('xbj,xai,xji->xab',tf.math.conj(pullbacks),pullbacks,dz_Hnu) #the barred index is first, and the derivative index is second! Same is true, in fact, for the inverse metric



def extder_jbar_for_sigma(points,sigma):
    ncoords = tf.shape(points[0])[-1] // 2 
    #pointstensor=tf.constant(points)
    pointstensor=points
    with tf.GradientTape(persistent=False) as tape2:
        tape2.watch(pointstensor)
        cpoints=point_vec_to_complex(pointstensor)
        #HNu=tf.einsum('x,xb->xb',tf.cast(HYMmetric(pointstensor),complex_dtype),tf.cast(harmonicform_jbar(tf.cast(cpoints,complex_dtype)),complex_dtype))#complexpoints vs noncomplex
        sigma = sigma(pointstensor)
        real_part = tf.math.real(sigma)
        imag_part = tf.math.imag(sigma)

        # Stack them along a new dimension
        sigmastack = tf.stack([real_part, imag_part], axis=-1)
    dsigma = tape2.batch_jacobian(sigmastack, pointstensor)
    dx_sigma, dy_sigma = \
        0.5*dsigma[:,:, :ncoords], \
        0.5*dsigma[:,:, ncoords:]
    dzbar_sigma = tf.complex(dx_sigma[:,0]-dy_sigma[:,1],dx_sigma[:,1]+dy_sigma[:,0])#do d(R+iI)/dzbar = (v_x +i v_y)/2 if v = R+iI
    #dzbar_sigma = tf.complex(tf.math.real(dx_sigma[:,0],tf.math.real(dy_sigma))#-tf.math.imag(dy_sigma),tf.math.imag(dx_sigma)+tf.math.real(dy_sigma))#do d(R+iI)/dzbar = (v_x +i v_y)/2 if v = R+iI
    return dzbar_sigma 

# def compute_source_for_harmonicForm(points,HYMmetric,harmonicform_jbar,invmetric,pullbacks):
#     ncoords = tf.shape(points[0])[-1] // 2 
#     pointstensor=tf.constant(points)
#     with tf.GradientTape(persistent=True) as tape1:
#         tape1.watch(pointstensor)
#         cpoints=point_vec_to_complex(pointstensor)
#         HNu=tf.einsum('x,xb->xb',tf.cast(HYMmetric(pointstensor),complex_dtype),tf.cast(harmonicform_jbar(tf.cast(cpoints,complex_dtype)),complex_dtype))#complexpoints vs noncomplex
#         #print("updated to fix! 5th dec")
#         #print(np.shape(HNu))
#         dHnu = tape1.batch_jacobian(HNu, pointstensor)
#         dx_Hnu, dy_Hnu = \
#             0.5*dHnu[:,:, :ncoords], \
#             0.5*dHnu[:,:, ncoords:]
        
#         dz_Hnu = tf.complex(tf.math.real(dx_Hnu)+tf.math.imag(dy_Hnu),tf.math.imag(dx_Hnu)-tf.math.real(dy_Hnu)) #note that the derivative index is the second index (the last index, as seen from above)
#         #check - e.g.  Hnu might be x+iy = z. We want the derivative to be 1.
#         # so take Hnu = R + iY
#         # now do: dxR + dyI + i(dxI-dyR), gives 1/2 + 1/2  +0 + 0= 1
#         # same for iz, now get 0 + 0 + i( 1/2 -(-1/2)) =i 
#     return -tf.einsum('xba,xbj,xai,xji->x',invmetric,tf.math.conj(pullbacks),pullbacks,dz_Hnu)


def antiholo_extder_for_nu_w_dzbar(points,nu):
    ncoords = tf.shape(points[0])[-1] // 2 
    #pointstensor=tf.constant(points)
    pointstensor=points
    with tf.GradientTape(persistent=False) as tape2:
        tape2.watch(pointstensor)
        cpoints=point_vec_to_complex(pointstensor)
        #HNu=tf.einsum('x,xb->xb',tf.cast(HYMmetric(pointstensor),complex_dtype),tf.cast(harmonicform_jbar(tf.cast(cpoints,complex_dtype)),complex_dtype))#complexpoints vs noncomplex
        nujbar = nu(pointstensor)
        real_part = tf.math.real(nujbar)
        imag_part = tf.math.imag(nujbar)

        # Stack them along a new dimension
        nu_stack = tf.stack([real_part, imag_part], axis=1)
    dnubar = tape2.batch_jacobian(nu_stack, pointstensor)
    #batch axis, real complex axis, original jbar axis, derivative axis
    dx_nu, dy_nu = \
        0.5*dnubar[:,:,:, :ncoords], \
        0.5*dnubar[:,:,:, ncoords:]
    dzbar_sigma = tf.complex(dx_nu[:,0]-dy_nu[:,1],dx_nu[:,1]+dy_nu[:,0])#do d(R+iI)/dzbar = (v_x +i v_y)/2 if v = R+iI
    dzbar_sigma = 0.5*(dzbar_sigma - tf.transpose(dzbar_sigma,perm=[0,1,3,2]))
    #dzbar_sigma = tf.complex(tf.math.real(dx_sigma[:,0],tf.math.real(dy_sigma))#-tf.math.imag(dy_sigma),tf.math.imag(dx_sigma)+tf.math.real(dy_sigma))#do d(R+iI)/dzbar = (v_x +i v_y)/2 if v = R+iI
    return dzbar_sigma 


def compute_transition_pointwise_measure(functionmodel, points):
        r"""Computes transition loss at each point for a function!
        Args:
            points (tf.tensor([bSize, 2*ncoords], real_dtype)): Points.

        Returns:
            tf.tensor([bSize], real_dtype): Transition loss at each point.
        """
        inv_one_mask = functionmodel._get_inv_one_mask(points)
        patch_indices = tf.where(~inv_one_mask)[:, 1]
        patch_indices = tf.reshape(patch_indices, (-1, functionmodel.nProjective))
        current_patch_mask = functionmodel._indices_to_mask(patch_indices)
        fixed = functionmodel._find_max_dQ_coords(points)
        cpoints = tf.complex(points[:, :functionmodel.ncoords], points[:, functionmodel.ncoords:])
        if functionmodel.nhyper == 1:
            other_patches = tf.gather(functionmodel.fixed_patches, fixed)
        else:
            combined = tf.concat((fixed, patch_indices), axis=-1)
            other_patches = functionmodel._generate_patches_vec(combined)
        
        other_patches = tf.reshape(other_patches, (-1, functionmodel.nProjective))
        other_patch_mask = functionmodel._indices_to_mask(other_patches)
        # NOTE: This will include same to same patch transitions
        exp_points = tf.repeat(cpoints, functionmodel.nTransitions, axis=-2)#expanded points
        patch_points = functionmodel._get_patch_coordinates(exp_points, tf.cast(other_patch_mask, dtype=tf.bool)) # other patches
        real_patch_points = tf.concat((tf.math.real(patch_points), tf.math.imag(patch_points)), axis=-1)
        gj = functionmodel.model(real_patch_points, training=True)
        gi = tf.repeat(functionmodel.model(points), functionmodel.nTransitions, axis=0)
        #print(np.shape(gj))
        all_t_loss = tf.math.abs(gj-gi)
        all_t_loss = tf.reshape(all_t_loss, (-1))
        #delete elements of all_t_loss that are zero
        indices = tf.math.not_equal(all_t_loss, 0.)

        # use boolean mask to remove zero elements
        all_t_loss = tf.boolean_mask(all_t_loss, indices)

        evalmodelonpoints=functionmodel.model(points)
        stddev=tf.math.reduce_std(evalmodelonpoints)
        #averagevalueofphi = tf.reduce_mean(tf.math.abs(evalmodelonpoints))
        #averagevalueofphisquared = tf.reduce_mean(tf.math.abs(evalmodelonpoints)**2)
        #stddev=tf.math.sqrt(averagevalueofphisquared-averagevalueofphi**2)
        meanoverstddev=tf.reduce_mean(all_t_loss)/stddev
        #print("average value/stddev "+ str(meanoverstddev))

        return meanoverstddev,all_t_loss/stddev
        

def HYM_measure_val(betamodel,databeta):
    #arguments: betamodel, databeta
    #outputs: weighted by the point weights, the failure to solve the equation i.e.:
    # 1: number: sum(w*|laplacian(beta)-rho|)/|sum(w.|rho|)|, where w is the point weight, rho is the source
    # 2: vector: w*|laplacian(beta)-rho|/|sum(w.|rho|)|, where w is the point weight, rho is the source
    # 3: number: w*|laplacian(beta)-rho|)/sum(w.|rho|), where w is the point weight, rho is the source
    
    vals=databeta['y_val'][:,0]*tf.math.abs(laplacian(betamodel.model,databeta['X_val'],databeta['val_pullbacks'],databeta['inv_mets_val'])-databeta['sources_val'])
    val=tf.reduce_mean(vals, axis=-1)
    absolutevalsofsourcetimesweight=databeta['y_val'][:,0]*tf.math.abs(databeta['sources_val'])
    mean_ofabsolute_valofsourcetimesweight=tf.reduce_mean(absolutevalsofsourcetimesweight, axis=-1)
    return val/mean_ofabsolute_valofsourcetimesweight, vals/mean_ofabsolute_valofsourcetimesweight,vals/absolutevalsofsourcetimesweight

def HYM_measure_val_for_batching(betamodel, X_val, y_val, val_pullbacks, inv_mets_val, sources_val):
    #arguments: betamodel, X_val, y_val, val_pullbacks, inv_mets_val, sources_val
    #outputs: weighted by the point weights, the failure to solve the equation i.e.:
    # 1: number: sum(w*|laplacian(beta)-rho|)/|sum(w.|rho|)|, where w is the point weight, rho is the source
    # 2: vector: w*|laplacian(beta)-rho|/|sum(w.|rho|)|, where w is the point weight, rho is the source
    # 3: number: w*|laplacian(beta)-rho|)/sum(w.|rho|), where w is the point weight, rho is the source
    
    vals = y_val[:, 0] * tf.math.abs(laplacian(betamodel.model, X_val, val_pullbacks, inv_mets_val) - sources_val)
    val = tf.reduce_mean(vals, axis=-1)
    absolutevalsofsourcetimesweight = y_val[:, 0] * tf.math.abs(sources_val)
    mean_ofabsolute_valofsourcetimesweight = tf.reduce_mean(absolutevalsofsourcetimesweight, axis=-1)
    return val/mean_ofabsolute_valofsourcetimesweight#, vals/mean_ofabsolute_valofsourcetimesweight, vals/absolutevalsofsourcetimesweight

    
# def measure_laplacian_failure(betamodel,databeta):

# COME UP WITH A WAY TO DO THIS
#     #arguments: betamodel, databeta
#     #outputs: weighted by the point weights, the failure to solve the equation i.e.:
#     # 1: number: sum(w*|laplacian(beta)-rho|)/|sum(w.|rho|)|, where w is the point weight, rho is the source
#     # 2: vector: w*|laplacian(beta)-rho|/|sum(w.|rho|)|, where w is the point weight, rho is the source
#     # 3: number: w*|laplacian(beta)-rho|)/sum(w.|rho|), where w is the point weight, rho is the source
    
#     vals=databeta['y_val'][:,0]*tf.math.abs(laplacian(betamodel.model,databeta['X_val'],databeta['val_pullbacks'],databeta['inv_mets_val'])-databeta['sources_val'])
#     val=tf.reduce_mean(vals, axis=-1)
#     absolutevalsofsourcetimesweight=databeta['y_val'][:,0]*tf.math.abs(databeta['sources_val'])
#     mean_ofabsolute_valofsourcetimesweight=tf.reduce_mean(absolutevalsofsourcetimesweight, axis=-1)
#     return val/mean_ofabsolute_valofsourcetimesweight, vals/mean_ofabsolute_valofsourcetimesweight,vals/absolutevalsofsourcetimesweight


def HYM_measure_val_with_H(HFmodel,dataHF):
    #returns ratio means of deldagger V_corrected/deldagger V_FS
    #and returns
    pts = tf.cast(dataHF['X_val'],real_dtype)
    # compute the laplacian (withH) acting on the HFmodel
    laplacianvals=laplacianWithH(HFmodel,pts,dataHF['val_pullbacks'],dataHF['inv_mets_val'],HFmodel.HYMmetric)
    coclosuretrained=coclosure_check(pts,HFmodel.HYMmetric,HFmodel.functionforbaseharmonicform_jbar,HFmodel,dataHF['inv_mets_val'],dataHF['val_pullbacks'])
    coclosureofjustdsigma=coclosure_check(pts,HFmodel.HYMmetric,lambda x: 0*HFmodel.functionforbaseharmonicform_jbar(x),HFmodel,dataHF['inv_mets_val'],dataHF['val_pullbacks'])
    coclosureofvFS = coclosuretrained-coclosureofjustdsigma # by linearity
    averageoftraineddivaverageofvFS = tf.reduce_mean(tf.math.abs(coclosuretrained))/tf.reduce_mean(tf.math.abs(coclosureofvFS))
    traineddivaverageofvFS = tf.reduce_mean(tf.math.abs(coclosuretrained))/tf.reduce_mean(tf.math.abs(coclosureofvFS))

    #print("check this is tiny: ",tf.math.reduce_std(coclosureofjustdsigma/(laplacianvals)))
    return averageoftraineddivaverageofvFS,traineddivaverageofvFS,tf.math.reduce_std(coclosureofjustdsigma/laplacianvals)

def HYM_measure_val_with_H_relative_to_norm(HFmodel,dataHF,HYMmetric_model,metric_model):
    #returns ratio means of deldagger V_corrected/deldagger V_FS
    #and returns
    pts = tf.cast(dataHF['X_val'],real_dtype)
    # compute the laplacian (withH) acting on the HFmodel
    laplacianvals=laplacianWithH(HFmodel,pts,dataHF['val_pullbacks'],dataHF['inv_mets_val'],HFmodel.HYMmetric)
    coclosuretrained=coclosure_check(pts,HFmodel.HYMmetric,HFmodel.functionforbaseharmonicform_jbar,HFmodel,dataHF['inv_mets_val'],dataHF['val_pullbacks'])
    coclosureofjustdsigma=coclosure_check(pts,HFmodel.HYMmetric,lambda x: 0*HFmodel.functionforbaseharmonicform_jbar(x),HFmodel,dataHF['inv_mets_val'],dataHF['val_pullbacks'])
    coclosureofFSdirect=coclosure_check(pts,HFmodel.HYMmetric,HFmodel.functionforbaseharmonicform_jbar,lambda x: tf.ones(x.shape[0],dtype=complex_dtype),dataHF['inv_mets_val'],dataHF['val_pullbacks'])
    coclosureofvFS = coclosuretrained-coclosureofjustdsigma # by linearity
    averageoftraineddivaverageofvFS = tf.reduce_mean(tf.math.abs(coclosuretrained))/tf.reduce_mean(tf.math.abs(coclosureofvFS))
    traineddivaverageofvFS = tf.reduce_mean(tf.math.abs(coclosuretrained))/tf.reduce_mean(tf.math.abs(coclosureofvFS))

    trained_one_form = HFmodel.corrected_harmonicform(pts)
    trained_one_form_conj = tf.math.conj(trained_one_form)
    FS_one_form = HFmodel.uncorrected_FS_harmonicform(pts)
    FS_one_form_conj = tf.math.conj(FS_one_form)
    HYMmetric_pts  = tf.cast(HYMmetric_model(pts),complex_dtype)
    FS_HYMmetric_pts = tf.cast(HYMmetric_model.raw_FS_HYM_r(pts),complex_dtype)
    trained_inv_metric = tf.linalg.inv(metric_model(pts))
    FS_inv_metric = tf.linalg.inv(metric_model.fubini_study_pb(pts))
    
    trained_one_form_conj_times_metric = tf.math.sqrt(tf.math.abs(tf.einsum('x,xBa,xa,xB->x',HYMmetric_pts,trained_inv_metric,trained_one_form_conj,trained_one_form)))#trained is bbar indices
    FS_one_form_conj_times_metric = tf.math.sqrt(tf.math.abs(tf.einsum('x,xBa,xa,xB->x',FS_HYMmetric_pts,FS_inv_metric,FS_one_form_conj,FS_one_form)))#untrained is b indices
    print('average mean of trained coclosure: ',tf.reduce_mean(tf.math.abs(coclosuretrained)))
    print('average mean of FS coclosure: ',tf.reduce_mean(tf.math.abs(coclosureofvFS)), 'coclosure of FS direct: ',tf.reduce_mean(tf.math.abs(coclosureofFSdirect)))
    print('average mean of trained one form: ',tf.reduce_mean(tf.math.abs(trained_one_form_conj_times_metric)))
    print('average mean of FS one form: ',tf.reduce_mean(tf.math.abs(FS_one_form_conj_times_metric)))
    
    TrainedDivTrained = tf.reduce_mean(tf.math.abs(coclosuretrained)/trained_one_form_conj_times_metric)
    avgavagTrainedDivTrained = tf.reduce_mean(tf.math.abs(coclosuretrained))/tf.reduce_mean(trained_one_form_conj_times_metric)
    TrainedDivFS = tf.reduce_mean(tf.math.abs(coclosuretrained)/FS_one_form_conj_times_metric)
    avgavagTrainedDivFS = tf.reduce_mean(tf.math.abs(coclosuretrained))/tf.reduce_mean(FS_one_form_conj_times_metric)
    FS_DivFS = tf.reduce_mean(tf.math.abs(coclosureofvFS)/FS_one_form_conj_times_metric)
    avgavagFS_DivFS = tf.reduce_mean(tf.math.abs(coclosureofvFS))/tf.reduce_mean(FS_one_form_conj_times_metric)
    #print("check this is tiny: ",tf.math.reduce_std(coclosureofjustdsigma/(laplacianvals)))
    return TrainedDivTrained, avgavagTrainedDivTrained, TrainedDivFS, avgavagTrainedDivFS, FS_DivFS, avgavagFS_DivFS

def HYM_measure_val_with_H_for_batching(HFmodel, X_val, y_val, val_pullbacks, inv_mets_val):
    #returns ratio means of deldagger V_corrected/deldagger V_FS
    #and returns
    pts = tf.cast(X_val, real_dtype)
    # compute the laplacian (withH) acting on the HFmodel
    laplacianvals = laplacianWithH(HFmodel, pts, val_pullbacks, inv_mets_val, HFmodel.HYMmetric)
    coclosuretrained = coclosure_check(pts, HFmodel.HYMmetric, HFmodel.functionforbaseharmonicform_jbar, HFmodel, inv_mets_val, val_pullbacks)
    coclosureofjustdsigma = coclosure_check(pts, HFmodel.HYMmetric, lambda x: 0*HFmodel.functionforbaseharmonicform_jbar(x), HFmodel, inv_mets_val, val_pullbacks)
    coclosureofvFS = coclosuretrained - coclosureofjustdsigma # by linearity
    
    # Use y_val as weights for the averages
    weights = y_val[:, 0]  # Assuming the first column of y_val contains the weights
    averageoftraineddivaverageofvFS = tf.reduce_mean(weights * tf.math.abs(coclosuretrained)) / tf.reduce_mean(weights * tf.math.abs(coclosureofvFS))
    traineddivaverageofvFS = tf.reduce_mean(weights * tf.math.abs(coclosuretrained)) / tf.reduce_mean(weights * tf.math.abs(coclosureofvFS))

    #print("check this is tiny: ",tf.math.reduce_std(coclosureofjustdsigma/(laplacianvals)))
    return averageoftraineddivaverageofvFS#, traineddivaverageofvFS, tf.math.reduce_std(weights * coclosureofjustdsigma/laplacianvals)

def compute_transition_pointwise_measure_section(HFmodel, points):
        r"""Computes transition loss at each point. In the case of the harmonic form model, we demand that the section transforms as a section of the line bundle to which it belongs. \phi(\lambda^q_i z_i)=\phi(z_i)
        also can separately check that the 1-form itHFmodel transforms appropriately?

        Args:
            points (tf.tensor([bSize, 2*ncoords], real_dtype)): Points.

        Returns:
            tf.tensor([bSize], real_dtype): Transition loss at each point.
        """
        inv_one_mask = HFmodel._get_inv_one_mask(points)
        patch_indices = tf.where(~inv_one_mask)[:, 1]
        patch_indices = tf.reshape(patch_indices, (-1, HFmodel.nProjective))
        current_patch_mask = HFmodel._indices_to_mask(patch_indices)
        fixed = HFmodel._find_max_dQ_coords(points)
        cpoints = tf.complex(points[:, :HFmodel.ncoords], points[:, HFmodel.ncoords:])
        if HFmodel.nhyper == 1:
            other_patches = tf.gather(HFmodel.fixed_patches, fixed)
        else:
            combined = tf.concat((fixed, patch_indices), axis=-1)
            other_patches = HFmodel._generate_patches_vec(combined)
        
        other_patches = tf.reshape(other_patches, (-1, HFmodel.nProjective))
        other_patch_mask = HFmodel._indices_to_mask(other_patches)
        # NOTE: This will include same to same patch transitions
        exp_points = tf.repeat(cpoints, HFmodel.nTransitions, axis=-2)#expanded points
        patch_points = HFmodel._get_patch_coordinates(exp_points, tf.cast(other_patch_mask, dtype=tf.bool)) # other patches
        real_patch_points = tf.concat((tf.math.real(patch_points), tf.math.imag(patch_points)), axis=-1)
        sigmaj = HFmodel(real_patch_points, training=True)
        sigmai = tf.repeat(HFmodel(points), HFmodel.nTransitions, axis=0)
        # this takes (1,z1,1,z2,1,z3,1,z4), picks out the ones we want to set to 1 - i.e. z1,z2,z3,z4 for the (1,1,1,1) patch,
        # and returns z1^k1 x z2^k2 etc... It therefore should multiply the w!
        transformation,weights=HFmodel.get_section_transition_to_patch_mask(exp_points,other_patch_mask) 
        all_t_loss = tf.math.abs(sigmai-transformation*sigmaj)*weights
        all_t_loss = tf.reshape(all_t_loss, (-1))
        #delete elements of all_t_loss that are zero
        indices = tf.math.not_equal(all_t_loss, 0.)

        # use boolean mask to remove zero elements
        all_t_loss = tf.boolean_mask(all_t_loss, indices)

        # evaluate HFmodel on points, this returns sigma, complexified
        evalmodelonpoints=HFmodel(points)
        #stdev returns a real number from complex arguments
        stddev = tf.math.reduce_std(evalmodelonpoints)
        
        meanoverstddev=tf.reduce_mean(all_t_loss)/stddev

        return meanoverstddev,all_t_loss/stddev

    


#check the corrected FS
def compute_transition_loss_for_uncorrected_HF_model(HFmodel, points, weights=None):
    r"""Computes transition loss at each point.

    .. math::

        \mathcal{L} = \frac{1}{d} \sum_{k,j} 
            ||g^k - T_{jk} \cdot g^j T^\dagger_{jk}||_n

    Args:
        points (tf.tensor([bSize, 2*ncoords], real_dtype)): Points.

    Returns:
        tf.tensor([bSize], real_dtype): Transition loss at each point.
    """
    inv_one_mask = HFmodel._get_inv_one_mask(points)
    patch_indices = tf.where(~inv_one_mask)[:, 1]
    patch_indices = tf.reshape(patch_indices, (-1, HFmodel.nProjective))
    current_patch_mask = HFmodel._indices_to_mask(patch_indices)
    cpoints = tf.complex(points[:, :HFmodel.ncoords],
                         points[:, HFmodel.ncoords:])
    fixed = HFmodel._find_max_dQ_coords(points)
    if HFmodel.nhyper == 1:
        other_patches = tf.gather(HFmodel.fixed_patches, fixed)
    else:
        combined = tf.concat((fixed, patch_indices), axis=-1)
        other_patches = HFmodel._generate_patches_vec(combined)
    other_patches = tf.reshape(other_patches, (-1, HFmodel.nProjective))
    other_patch_mask = HFmodel._indices_to_mask(other_patches)
    # NOTE: This will include same to same patch transitions
    exp_points = tf.repeat(cpoints, HFmodel.nTransitions, axis=-2)
    patch_points = HFmodel._get_patch_coordinates(
        exp_points,
        tf.cast(other_patch_mask, dtype=tf.bool))
    fixed = tf.reshape(
        tf.tile(fixed, [1, HFmodel.nTransitions]), (-1, HFmodel.nhyper))
    real_patch_points = tf.concat(
        (tf.math.real(patch_points), tf.math.imag(patch_points)),
        axis=-1)
    vj = HFmodel.uncorrected_FS_harmonicform(real_patch_points)
    # NOTE: We will compute this twice.
    # TODO: disentangle this to save one computation?
    vi = tf.repeat(HFmodel.uncorrected_FS_harmonicform(points), HFmodel.nTransitions, axis=0)
    current_patch_mask = tf.repeat(
        current_patch_mask, HFmodel.nTransitions, axis=0)
    print("ALL DTYPES: ", patch_points.dtype, other_patch_mask.dtype, current_patch_mask.dtype, fixed.dtype)
    Tij = HFmodel.get_transition_matrix(
        patch_points, other_patch_mask, current_patch_mask, fixed)
    patch_transformation,weights=HFmodel.get_section_transition_to_patch_mask(exp_points,other_patch_mask) 
    # work out what to do with weights here
    all_t_loss = tf.math.abs(tf.einsum('xj,x->xj',vj,patch_transformation)- tf.einsum('xk,xkl->xl', vi,
                              tf.transpose(Tij, perm=[0, 2, 1], conjugate=True)))
    all_t_loss = tf.math.reduce_sum(all_t_loss**HFmodel.n[1], axis=[1])
    #This should now be nTransitions 
    all_t_loss = tf.reshape(all_t_loss, (-1, HFmodel.nTransitions))
    all_t_loss = tf.math.reduce_sum(all_t_loss, axis=-1)
    if weights is not None:
        weights = weights/tf.reduce_mean(weights)
        all_t_loss = all_t_loss*weights
    return all_t_loss/(HFmodel.nTransitions*HFmodel.nfold)


def compute_transition_loss_for_corrected_HF_model(HFmodel, points, weights=None):
    r"""Computes transition loss at each point.

    .. math::

        \mathcal{L} = \frac{1}{d} \sum_{k,j} 
            ||g^k - T_{jk} \cdot g^j T^\dagger_{jk}||_n

    Args:
        points (tf.tensor([bSize, 2*ncoords], real_dtype)): Points.

    Returns:
        tf.tensor([bSize], real_dtype): Transition loss at each point.
    """
    inv_one_mask = HFmodel._get_inv_one_mask(points)
    patch_indices = tf.where(~inv_one_mask)[:, 1]
    patch_indices = tf.reshape(patch_indices, (-1, HFmodel.nProjective))
    current_patch_mask = HFmodel._indices_to_mask(patch_indices)
    cpoints = tf.complex(points[:, :HFmodel.ncoords],
                         points[:, HFmodel.ncoords:])
    fixed = HFmodel._find_max_dQ_coords(points)
    if HFmodel.nhyper == 1:
        other_patches = tf.gather(HFmodel.fixed_patches, fixed)
    else:
        combined = tf.concat((fixed, patch_indices), axis=-1)
        other_patches = HFmodel._generate_patches_vec(combined)
    other_patches = tf.reshape(other_patches, (-1, HFmodel.nProjective))
    other_patch_mask = HFmodel._indices_to_mask(other_patches)
    # NOTE: This will include same to same patch transitions
    exp_points = tf.repeat(cpoints, HFmodel.nTransitions, axis=-2)
    patch_points = HFmodel._get_patch_coordinates(
        exp_points,
        tf.cast(other_patch_mask, dtype=tf.bool))
    fixed = tf.reshape(
        tf.tile(fixed, [1, HFmodel.nTransitions]), (-1, HFmodel.nhyper))
    real_patch_points = tf.concat(
        (tf.math.real(patch_points), tf.math.imag(patch_points)),
        axis=-1)
    vj = HFmodel.corrected_harmonicform(real_patch_points)
    # NOTE: We will compute this twice.
    # TODO: disentangle this to save one computation?
    vi = tf.repeat(HFmodel.corrected_harmonicform(points), HFmodel.nTransitions, axis=0)
    current_patch_mask = tf.repeat(
        current_patch_mask, HFmodel.nTransitions, axis=0)
    #print("ALL DTYPES: ", patch_points.dtype, other_patch_mask.dtype, current_patch_mask.dtype, fixed.dtype)
    Tij = HFmodel.get_transition_matrix(
        patch_points, other_patch_mask, current_patch_mask, fixed)
    patch_transformation,weights=HFmodel.get_section_transition_to_patch_mask(exp_points,other_patch_mask) 
    # work out what to do with weights here
    all_t_loss = tf.math.abs(tf.einsum('xj,x->xj',vj,patch_transformation)- tf.einsum('xk,xkl->xl', vi,
                              tf.transpose(Tij, perm=[0, 2, 1], conjugate=True)))
    all_t_loss = tf.math.reduce_sum(all_t_loss**HFmodel.n[1], axis=[1])
    #This should now be nTransitions 
    
    all_t_loss = tf.reshape(all_t_loss, (-1, HFmodel.nTransitions))
    all_t_loss = tf.math.reduce_sum(all_t_loss, axis=-1)
    if weights is not None:
        weights = weights/tf.reduce_mean(weights)
        all_t_loss = all_t_loss*weights
    return all_t_loss/(HFmodel.nTransitions*HFmodel.nfold)




