import os
import numpy as np
import tensorflow as tf
from cymetric.config import real_dtype, complex_dtype


from auxiliary_funcs import *
from laplacian_funcs import *
import gc
import uuid
from datetime import datetime

def do_integrals(manifold_name_and_data, pg, dataEval, phimodel, betamodel_LB1, betamodel_LB2, betamodel_LB3, HFmodel_vH, HFmodel_vQ3, HFmodel_vU3, HFmodel_vQ1, HFmodel_vQ2, HFmodel_vU1, HFmodel_vU2, network_params, do_extra_stuff = None):
    (_, kmoduli, _, _, foldername, unique_id_or_coeff) = manifold_name_and_data
    n_p = len(dataEval['X_train'])
    points64=dataEval['X_train'][0:n_p]
    real_pts=tf.cast(dataEval['X_train'][0:n_p],real_dtype)
    volCY_from_Om=tf.reduce_mean(dataEval['y_train'][:n_p,0])/6 #this is the actual volume of the CY computed from Omega.since J^J^J^ = K Om^Ombar?? not totally sure here:
    pointsComplex=tf.cast(point_vec_to_complex(points64),complex_dtype)# we have to cast to complex64 from complex128 sometimes. Just for safety. point_vec_to_complex does the obvious thing
    weights=dataEval['y_train'][:n_p,0]
    omegasquared=dataEval['y_train'][:n_p,1]

    print("\n do analysis: \n\n\n")

    #Vol_from_dijk_J=pg.get_volume_from_intersections(kmoduli)
   # Vol_reference_dijk_with_k_is_1=pg.get_volume_from_intersections(np.ones_like(kmoduli)) 
    #volCY_from_Om=tf.reduce_mean(dataEval['y_train'][:n_p,0])
    print("Compute holomorphic Yukawas")
    #consider omega normalisation
    omega = tf.cast(batch_process_helper_func(pg.holomorphic_volume_form, [pointsComplex], batch_indices=[0], batch_size=100000),complex_dtype)
    # Verify that dataEval['y_train'][:n_p,1] equals |omega|^2
    omega_abs_squared = tf.math.real(omega * tf.math.conj(omega))
    assert tf.reduce_all(tf.abs(omega_abs_squared[:100] - dataEval['y_train'][:100,1]) < 1e-5), "First elements of dataEval['y_train'][:,1] should equal |omega|^2"
    #put the omega here, not the omegabar
    omega_normalised_to_one=omega/tf.cast(np.sqrt(volCY_from_Om),complex_dtype) # this is the omega that's normalised to 1. VERIFIED yes.

    # this is integral omega wedge omegabar? yeah.


    aux_weights=tf.cast((weights/(omegasquared))*(1/(6)),real_dtype)### these are the appropriate 'flat measure' so sum_i aux_weights f is int_X f(x)
    aux_weights_c = tf.cast(aux_weights, complex_dtype)

    #convertcomptoreal=lambda complexvec: tf.concat([tf.math.real(complexvec),tf.math.imag(complexvec)],-1) # this converts from complex to real

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    batch_size_for_processing=min(100000, len(real_pts))
    if len(real_pts)>100000:
        actually_batch=True
    else:
        actually_batch=False
    mets_bare = batch_process_helper_func(phimodel.fubini_study_pb, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch, kwargs={'ts':tf.cast(kmoduli,complex_dtype)})
    vH_bare = batch_process_helper_func(HFmodel_vH.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch)
    vQ3_bare = batch_process_helper_func(HFmodel_vQ3.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch)
    vU3_bare = batch_process_helper_func(HFmodel_vU3.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch)
    vQ1_bare = batch_process_helper_func(HFmodel_vQ1.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch)
    vQ2_bare = batch_process_helper_func(HFmodel_vQ2.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch)
    vU1_bare = batch_process_helper_func(HFmodel_vU1.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch)
    vU2_bare = batch_process_helper_func(HFmodel_vU2.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch)

    mats=[]
    masses_trained_and_ref=[]
    use_trained = False
    holomorphic_Yukawas_trained_and_ref=[]
    for use_trained in [True,False]:
        print("using trained? " + str(use_trained))
        if use_trained:
            mets = batch_process_helper_func(phimodel, (real_pts,), batch_indices=(0,), batch_size=10000, compile_func=True)
            print('got mets', flush=True)
            dets = tf.linalg.det(mets)
            batchbetamin = 1000000
            if len(real_pts)>batchbetamin:
                batch_betamodel= True
            else:
                batch_betamodel= False
            
            H1=batch_process_helper_func(betamodel_LB1, (real_pts,), batch_indices=(0,), batch_size=batchbetamin, compile_func=True, actually_batch=batch_betamodel) 
            H2=batch_process_helper_func(betamodel_LB2, (real_pts,), batch_indices=(0,), batch_size=batchbetamin, compile_func=True, actually_batch=batch_betamodel) 
            H3=batch_process_helper_func(betamodel_LB3, (real_pts,), batch_indices=(0,), batch_size=batchbetamin, compile_func=True, actually_batch=batch_betamodel) 

            LB1c=tf.cast(H1, complex_dtype)
            LB2c=tf.cast(H2, complex_dtype)
            LB3c=tf.cast(H3, complex_dtype)
            # Batch process corrected harmonic forms
            vH = batch_process_helper_func(HFmodel_vH.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvHb = tf.einsum('x,xb->xb', LB1c, tf.math.conj(vH))
            print('got vH', flush=True)

            vQ3 = batch_process_helper_func(HFmodel_vQ3.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvQ3b = tf.einsum('x,xb->xb', LB2c, tf.math.conj(vQ3))
            print('got vQ3', flush=True)

            vU3 = batch_process_helper_func(HFmodel_vU3.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvU3b = tf.einsum('x,xb->xb', LB2c, tf.math.conj(vU3))
            print('got vU3', flush=True)

            vQ1 = batch_process_helper_func(HFmodel_vQ1.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvQ1b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vQ1))
            print('got vQ1', flush=True)

            vQ2 = batch_process_helper_func(HFmodel_vQ2.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvQ2b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vQ2))
            print('got vQ2', flush=True)

            vU1 = batch_process_helper_func(HFmodel_vU1.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvU1b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vU1))
            print('got vU1', flush=True)

            vU2 = batch_process_helper_func(HFmodel_vU2.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvU2b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vU2))
            print('got vU2', flush=True)
        elif not use_trained:
            H1=betamodel_LB1.raw_FS_HYM_r(real_pts) 
            H2=betamodel_LB2.raw_FS_HYM_r(real_pts) 
            H3=betamodel_LB3.raw_FS_HYM_r(real_pts) 
            LB1c=tf.cast(H1, complex_dtype)
            LB2c=tf.cast(H2, complex_dtype)
            LB3c=tf.cast(H3, complex_dtype)

            mets = mets_bare
            vH = vH_bare
            hvHb = tf.einsum('x,xb->xb', LB1c, tf.math.conj(vH))
            
            vQ3 = vQ3_bare
            hvQ3b = tf.einsum('x,xb->xb', LB2c, tf.math.conj(vQ3))
            
            vU3 = vU3_bare
            hvU3b = tf.einsum('x,xb->xb', LB2c, tf.math.conj(vU3))
            
            vQ1 = vQ1_bare
            hvQ1b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vQ1))
            
            vQ2 = vQ2_bare
            hvQ2b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vQ2))
            
            vU1 = vU1_bare
            hvU1b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vU1))
            
            vU2 = vU2_bare
            hvU2b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vU2))
         

        print("Now compute the integrals")
        #(1j) for each FS form #maybe should be (1j/2)??
        #(-j)**3 to get the flat measure dx1dy1dx2dy2dx3dy3 # and this should be (-2j)???
        #(-1) to rearrange dz wedge dzbar in the v wedge hvb in canonical order.
        # whence the -1j/2???? it comes from the hodge star (the one out the front)
        #overall have added a factor of 2
        print("The field normalisations:")
        #check this is constant!
        lc_c = tf.cast(pg.lc,complex_dtype)

        # Dictionary to store integral values and their errors
        integral_stats = {}

        # Calculate HuHu and its standard error
        HuHu_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vH,hvHb,lc_c,lc_c)
        HuHu, HuHu_se, HuHu_eff_n, integral_stats['HuHu'] = weighted_mean_and_standard_error(HuHu_integrand, aux_weights,is_top_form=True)
        print("(Hu,Hu) = " + str(HuHu) + " ± " + str(HuHu_se) + " (eff. n = " + str(HuHu_eff_n) + ")")
        print(f"  Real part: {tf.math.real(HuHu).numpy():.6e} ± {tf.math.real(HuHu_se).numpy():.6e}")
        print(f"  Imag part: {tf.math.imag(HuHu).numpy():.6e} ± {tf.math.imag(HuHu_se).numpy():.6e}")

        # Calculate Q3Q3 and its standard error
        Q3Q3_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ3,hvQ3b,lc_c,lc_c)
        Q3Q3, Q3Q3_se, Q3Q3_eff_n, integral_stats['Q3Q3'] = weighted_mean_and_standard_error(Q3Q3_integrand, aux_weights, is_top_form=True)
        print("(Q3,Q3) = " + str(Q3Q3) + " ± " + str(Q3Q3_se) + " (eff. n = " + str(Q3Q3_eff_n) + ")")

        # Calculate U3U3 and its standard error
        U3U3_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU3,hvU3b,lc_c,lc_c,)
        U3U3, U3U3_se, U3U3_eff_n, integral_stats['U3U3'] = weighted_mean_and_standard_error(U3U3_integrand, aux_weights, is_top_form=True)
        print("(U3,U3) = " + str(U3U3) + " ± " + str(U3U3_se) + " (eff. n = " + str(U3U3_eff_n) + ")")

        # Calculate U1U1 and its standard error
        U1U1_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU1,hvU1b,lc_c,lc_c)
        U1U1, U1U1_se, U1U1_eff_n, integral_stats['U1U1'] = weighted_mean_and_standard_error(U1U1_integrand, aux_weights,is_top_form=True)
        print("(U1,U1) = " + str(U1U1) + " ± " + str(U1U1_se) + " (eff. n = " + str(U1U1_eff_n) + ")")

        # Calculate U2U2 and its standard error
        U2U2_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU2,hvU2b,lc_c,lc_c)
        U2U2, U2U2_se, U2U2_eff_n, integral_stats['U2U2'] = weighted_mean_and_standard_error(U2U2_integrand, aux_weights,is_top_form=True)
        print("(U2,U2) = " + str(U2U2) + " ± " + str(U2U2_se) + " (eff. n = " + str(U2U2_eff_n) + ")")

        # Calculate Q1Q1 and its standard error
        Q1Q1_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ1,hvQ1b,lc_c,lc_c)
        Q1Q1, Q1Q1_se, Q1Q1_eff_n, integral_stats['Q1Q1'] = weighted_mean_and_standard_error(Q1Q1_integrand, aux_weights,is_top_form=True)
        print("(Q1,Q1) = " + str(Q1Q1) + " ± " + str(Q1Q1_se) + " (eff. n = " + str(Q1Q1_eff_n) + ")")

        # Calculate Q2Q2 and its standard error
        Q2Q2_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ2,hvQ2b,lc_c,lc_c)
        Q2Q2, Q2Q2_se, Q2Q2_eff_n, integral_stats['Q2Q2'] = weighted_mean_and_standard_error(Q2Q2_integrand, aux_weights,is_top_form=True)
        print("(Q2,Q2) = " + str(Q2Q2) + " ± " + str(Q2Q2_se) + " (eff. n = " + str(Q2Q2_eff_n) + ")")

        print("The field mixings:")

        # Calculate Q1Q2 and its standard error
        Q1Q2_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ1,hvQ2b,lc_c,lc_c)
        Q1Q2, Q1Q2_se, Q1Q2_eff_n, integral_stats['Q1Q2'] = weighted_mean_and_standard_error(Q1Q2_integrand, aux_weights,is_top_form=True)
        print("(Q1,Q2) = " + str(Q1Q2) + " ± " + str(Q1Q2_se) + " (eff. n = " + str(Q1Q2_eff_n) + ")")

        # Calculate Q2Q1 and its standard error
        Q2Q1_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ2,hvQ1b,lc_c,lc_c)
        Q2Q1, Q2Q1_se, Q2Q1_eff_n, integral_stats['Q2Q1'] = weighted_mean_and_standard_error(Q2Q1_integrand, aux_weights,is_top_form=True)
        print("(Q2,Q1) = " + str(Q2Q1) + " ± " + str(Q2Q1_se) + " (eff. n = " + str(Q2Q1_eff_n) + ")")

        # Calculate U1U2 and its standard error
        U1U2_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU1,hvU2b,lc_c,lc_c)
        U1U2, U1U2_se, U1U2_eff_n, integral_stats['U1U2'] = weighted_mean_and_standard_error(U1U2_integrand, aux_weights,is_top_form=True)
        print("(U1,U2) = " + str(U1U2) + " ± " + str(U1U2_se) + " (eff. n = " + str(U1U2_eff_n) + ")")

        # Calculate U2U1 and its standard error
        U2U1_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU2,hvU1b,lc_c,lc_c)
        U2U1, U2U1_se, U2U1_eff_n, integral_stats['U2U1'] = weighted_mean_and_standard_error(U2U1_integrand, aux_weights,is_top_form=True)
        print("(U2,U1) = " + str(U2U1) + " ± " + str(U2U1_se) + " (eff. n = " + str(U2U1_eff_n) + ")")



        tfsqrtandcast=lambda x: tf.cast(tf.math.sqrt(x),complex_dtype)
        print("CONSIDERING A PARTICULAR ELEMENT:")
        factor =(1/(8*np.sqrt(30))) *   2*np.sqrt(2)/4
        elements_21= factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU2)*omega_normalised_to_one
        elements_12 =factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vQ2,vU3)*omega_normalised_to_one 
        print("Max absolute value of elements_21:", tf.reduce_max(tf.abs(elements_21)))
        print("Mean of elements_21:", tf.reduce_mean(elements_21).numpy())
        print("Mean of abs(elements_21):", tf.reduce_mean(tf.abs(elements_21)).numpy())
        print("Max absolute value of elements_12:", tf.reduce_max(tf.abs(elements_12)).numpy())
        print("Mean of elements_12:", tf.reduce_mean(elements_12).numpy())
        print("Mean of abs(elements_12):", tf.reduce_mean(tf.abs(elements_12)).numpy())

        print("--------CONSIDERING THE INVERSION (not mixing  Qs and Us) PARTICULAR ELEMENT:")
        elements_Q3Q2= factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vQ2)*omega_normalised_to_one
        elements_U2U3 =factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vU2,vU3)*omega_normalised_to_one 
        print("Max absolute value of elements_Q3Q2:", tf.reduce_max(tf.abs(elements_Q3Q2)).numpy())
        print("Mean of elements_Q3Q2:", tf.reduce_mean(elements_Q3Q2).numpy())
        print("Mean of abs(elements_Q3Q2):", tf.reduce_mean(tf.abs(elements_Q3Q2)).numpy())
        print("Max absolute value of elements_U2U3:", tf.reduce_max(tf.abs(elements_U2U3)).numpy())
        print("Mean of elements_U2U3:", tf.reduce_mean(elements_U2U3).numpy())
        print("Mean of abs(elements_U2U3):", tf.reduce_mean(tf.abs(elements_U2U3)).numpy())
        print("--------------------------------")



        print("--------CONSIDERING THE (0,0) ELEMENT:")
        elements_00 = factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU1)*omega_normalised_to_one
        print("Max absolute value of elements_00:", tf.reduce_max(tf.abs(elements_00)).numpy())
        print("Mean of elements_00:", tf.reduce_mean(elements_00).numpy())
        print("Mean of abs(elements_00):", tf.reduce_mean(tf.abs(elements_00)).numpy())
        print("--------------------------------")
       

        print("--------CONSIDERING THE (1,1) ELEMENT:")
        elements_11 = factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU2)*omega_normalised_to_one
        print("Max absolute value of elements_11:", tf.reduce_max(tf.abs(elements_11)).numpy())     
        print("Mean of elements_11:", tf.reduce_mean(elements_11).numpy())
        print("Mean of abs(elements_11):", tf.reduce_mean(tf.abs(elements_11)).numpy())
        print("--------------------------------")


        print("--------CONSIDERING THE (2,2) ELEMENT:")
        elements_22 = factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ3,vU3)*omega_normalised_to_one
        print("Max absolute value of elements_22:", tf.reduce_max(tf.abs(elements_22)).numpy())     
        print("Mean of elements_22:", tf.reduce_mean(elements_22).numpy())
        print("Mean of abs(elements_22):", tf.reduce_mean(tf.abs(elements_22)).numpy())
        print("--------------------------------")
        #print("Integral of omega_normalised_to_one = ", tf.reduce_mean(aux_weights_c * omega_normalised_to_one* tf.math.conj(omega_normalised_to_one))) # verified that this is correct!!!yes


        if do_extra_stuff:
            print("\n\n\n\n\n\n\n\n Checking topological invariance if one of the forms is pure derivatives!")
            # Check topological invariance with pure derivatives
            integrand_Q3U2 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU2)*omega_normalised_to_one
            integrand_bare_Q3U2_vH = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH-vH_bare,vQ3,vU2)*omega_normalised_to_one
            integrand_bare_Q3U2_vQ3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3-vQ3_bare,vU2)*omega_normalised_to_one
            integrand_bare_Q3U2_vU2 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3,vU2-vU2_bare)*omega_normalised_to_one
            
            # Q1U3 combination
            integrand_Q1U3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vQ1,vU3)*omega_normalised_to_one
            integrand_bare_Q1U3_vH = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH-vH_bare,vQ1,vU3)*omega_normalised_to_one
            integrand_bare_Q1U3_vQ1 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ1-vQ1_bare,vU3)*omega_normalised_to_one
            integrand_bare_Q1U3_vU3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ1,vU3-vU3_bare)*omega_normalised_to_one
            
            # Q2U3 combination
            integrand_Q2U3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vQ2,vU3)*omega_normalised_to_one
            integrand_bare_Q2U3_vH = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH-vH_bare,vQ2,vU3)*omega_normalised_to_one
            integrand_bare_Q2U3_vQ2 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ2-vQ2_bare,vU3)*omega_normalised_to_one
            integrand_bare_Q2U3_vU3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ2,vU3-vU3_bare)*omega_normalised_to_one
            
            # Q3U1 combination
            integrand_Q3U1 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU1)*omega_normalised_to_one
            integrand_bare_Q3U1_vH = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH-vH_bare,vQ3,vU1)*omega_normalised_to_one
            integrand_bare_Q3U1_vQ3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3-vQ3_bare,vU1)*omega_normalised_to_one
            integrand_bare_Q3U1_vU1 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3,vU1-vU1_bare)*omega_normalised_to_one
            
            # Calculate statistics for all integrands
            int_Q3U2, int_Q3U2_se, int_Q3U2_eff_n, int_Q3U2_stats = weighted_mean_and_standard_error(integrand_Q3U2, aux_weights)
            int_bare_Q3U2_vH, int_bare_Q3U2_vH_se, int_bare_Q3U2_vH_eff_n, int_bare_Q3U2_vH_stats = weighted_mean_and_standard_error(integrand_bare_Q3U2_vH, aux_weights)
            int_bare_Q3U2_vQ3, int_bare_Q3U2_vQ3_se, int_bare_Q3U2_vQ3_eff_n, int_bare_Q3U2_vQ3_stats = weighted_mean_and_standard_error(integrand_bare_Q3U2_vQ3, aux_weights)
            int_bare_Q3U2_vU2, int_bare_Q3U2_vU2_se, int_bare_Q3U2_vU2_eff_n, int_bare_Q3U2_vU2_stats = weighted_mean_and_standard_error(integrand_bare_Q3U2_vU2, aux_weights)

            int_Q1U3, int_Q1U3_se, int_Q1U3_eff_n, int_Q1U3_stats = weighted_mean_and_standard_error(integrand_Q1U3, aux_weights)
            int_bare_Q1U3_vH, int_bare_Q1U3_vH_se, int_bare_Q1U3_vH_eff_n, int_bare_Q1U3_vH_stats = weighted_mean_and_standard_error(integrand_bare_Q1U3_vH, aux_weights)
            int_bare_Q1U3_vQ1, int_bare_Q1U3_vQ1_se, int_bare_Q1U3_vQ1_eff_n, int_bare_Q1U3_vQ1_stats = weighted_mean_and_standard_error(integrand_bare_Q1U3_vQ1, aux_weights)
            int_bare_Q1U3_vU3, int_bare_Q1U3_vU3_se, int_bare_Q1U3_vU3_eff_n, int_bare_Q1U3_vU3_stats = weighted_mean_and_standard_error(integrand_bare_Q1U3_vU3, aux_weights)

            int_Q2U3, int_Q2U3_se, int_Q2U3_eff_n, int_Q2U3_stats = weighted_mean_and_standard_error(integrand_Q2U3, aux_weights)
            int_bare_Q2U3_vH, int_bare_Q2U3_vH_se, int_bare_Q2U3_vH_eff_n, int_bare_Q2U3_vH_stats = weighted_mean_and_standard_error(integrand_bare_Q2U3_vH, aux_weights)
            int_bare_Q2U3_vQ2, int_bare_Q2U3_vQ2_se, int_bare_Q2U3_vQ2_eff_n, int_bare_Q2U3_vQ2_stats = weighted_mean_and_standard_error(integrand_bare_Q2U3_vQ2, aux_weights)
            int_bare_Q2U3_vU3, int_bare_Q2U3_vU3_se, int_bare_Q2U3_vU3_eff_n, int_bare_Q2U3_vU3_stats = weighted_mean_and_standard_error(integrand_bare_Q2U3_vU3, aux_weights)

            int_Q3U1, int_Q3U1_se, int_Q3U1_eff_n, int_Q3U1_stats = weighted_mean_and_standard_error(integrand_Q3U1, aux_weights)
            int_bare_Q3U1_vH, int_bare_Q3U1_vH_se, int_bare_Q3U1_vH_eff_n, int_bare_Q3U1_vH_stats = weighted_mean_and_standard_error(integrand_bare_Q3U1_vH, aux_weights)
            int_bare_Q3U1_vQ3, int_bare_Q3U1_vQ3_se, int_bare_Q3U1_vQ3_eff_n, int_bare_Q3U1_vQ3_stats = weighted_mean_and_standard_error(integrand_bare_Q3U1_vQ3, aux_weights)
            int_bare_Q3U1_vU1, int_bare_Q3U1_vU1_se, int_bare_Q3U1_vU1_eff_n, int_bare_Q3U1_vU1_stats = weighted_mean_and_standard_error(integrand_bare_Q3U1_vU1, aux_weights)
            
            print("\n Q3U2:")
            print(f"int_Q3U2 = {int_Q3U2:.6e} ± {int_Q3U2_se:.6e} (eff. n = {int_Q3U2_eff_n})")
            print(f"int_bare_Q3U2_vH = {int_bare_Q3U2_vH:.6e} ± {int_bare_Q3U2_vH_se:.6e} (eff. n = {int_bare_Q3U2_vH_eff_n})")
            print(f"int_bare_Q3U2_vQ3 = {int_bare_Q3U2_vQ3:.6e} ± {int_bare_Q3U2_vQ3_se:.6e} (eff. n = {int_bare_Q3U2_vQ3_eff_n})")
            print(f"int_bare_Q3U2_vU2 = {int_bare_Q3U2_vU2:.6e} ± {int_bare_Q3U2_vU2_se:.6e} (eff. n = {int_bare_Q3U2_vU2_eff_n})")
            print("\n Q1U3:")
            print(f"int_Q1U3 = {int_Q1U3:.6e} ± {int_Q1U3_se:.6e} (eff. n = {int_Q1U3_eff_n})")
            print(f"int_bare_Q1U3_vH = {int_bare_Q1U3_vH:.6e} ± {int_bare_Q1U3_vH_se:.6e} (eff. n = {int_bare_Q1U3_vH_eff_n})")
            print(f"int_bare_Q1U3_vQ1 = {int_bare_Q1U3_vQ1:.6e} ± {int_bare_Q1U3_vQ1_se:.6e} (eff. n = {int_bare_Q1U3_vQ1_eff_n})")
            print(f"int_bare_Q1U3_vU3 = {int_bare_Q1U3_vU3:.6e} ± {int_bare_Q1U3_vU3_se:.6e} (eff. n = {int_bare_Q1U3_vU3_eff_n})")
            print("\n Q2U3:")
            print(f"int_Q2U3 = {int_Q2U3:.6e} ± {int_Q2U3_se:.6e} (eff. n = {int_Q2U3_eff_n})")
            print(f"int_bare_Q2U3_vH = {int_bare_Q2U3_vH:.6e} ± {int_bare_Q2U3_vH_se:.6e} (eff. n = {int_bare_Q2U3_vH_eff_n})")
            print(f"int_bare_Q2U3_vQ2 = {int_bare_Q2U3_vQ2:.6e} ± {int_bare_Q2U3_vQ2_se:.6e} (eff. n = {int_bare_Q2U3_vQ2_eff_n})")
            print(f"int_bare_Q2U3_vU3 = {int_bare_Q2U3_vU3:.6e} ± {int_bare_Q2U3_vU3_se:.6e} (eff. n = {int_bare_Q2U3_vU3_eff_n})")
            print("\n Q3U1:")
            print(f"int_Q3U1 = {int_Q3U1:.6e} ± {int_Q3U1_se:.6e} (eff. n = {int_Q3U1_eff_n})")
            print(f"integrand_bare_Q3U1_vH = {int_bare_Q3U1_vH:.6e} ± {int_bare_Q3U1_vH_se:.6e} (eff. n = {int_bare_Q3U1_vH_eff_n})")
            print(f"integrand_bare_Q3U1_vQ3 = {int_bare_Q3U1_vQ3:.6e} ± {int_bare_Q3U1_vQ3_se:.6e} (eff. n = {int_bare_Q3U1_vQ3_eff_n})")
            print(f"integrand_bare_Q3U1_vU1 = {int_bare_Q3U1_vU1:.6e} ± {int_bare_Q3U1_vU1_se:.6e} (eff. n = {int_bare_Q3U1_vU1_eff_n})")
            
            
            print("--------------------------------\n\n\n\n\n\n\n\n")
           
        # vals = []
       
        # vals = []
        # mAll = []
        # mAll.append(m)
        # vals.append(np.abs([HuHu,Q1Q1,Q2Q2,Q3Q3,U1U1,U2U2,U3U3,Q1Q2,U1U2]))

        import scipy
        # # 4 is |G|
        # 0.5 is the factor of 2 outside the KIJ integral, which comes from 1/4*2 = 1/2 apparently????
        mfieldmatrixfactors = (1/2)* 0.5/4
        Hmat = mfieldmatrixfactors*np.array([[HuHu]])
        #Q1Q2 has Q2 barred, so that should multiply Q2bar which is on the LHS, so it should be on the first column
        Qmat = mfieldmatrixfactors*np.array([[Q1Q1,Q2Q1,0],[Q1Q2,Q2Q2,0],[0,0,Q3Q3]])
        Umat = mfieldmatrixfactors*np.array([[U1U1,U2U1,0],[U1U2,U2U2,0],[0,0,U3U3]])
        # Store effective sample sizes for each integral
        Qneffs = np.array([
            [Q1Q1_eff_n, Q2Q1_eff_n, 0],
            [Q1Q2_eff_n, Q2Q2_eff_n, 0],
            [0, 0, Q3Q3_eff_n]
        ])
        Uneffs = np.array([
            [U1U1_eff_n, U2U1_eff_n, 0],
            [U1U2_eff_n, U2U2_eff_n, 0],
            [0, 0, U3U3_eff_n]
        ])
        Hneffs = np.array([[HuHu_eff_n]])

        print("Hneffs, Qneffs, Uneffs: ")
        print(np.round(Hneffs,1))
        print(np.round(Qneffs,1))
        print(np.round(Uneffs,1))
        # Calculate error matrices
        Hmat_errors = mfieldmatrixfactors*np.array([[integral_stats['HuHu']['std_error']]])
        Qmat_errors = mfieldmatrixfactors*np.array([
            [integral_stats['Q1Q1']['std_error'], integral_stats['Q2Q1']['std_error'], 0],
            [integral_stats['Q1Q2']['std_error'], integral_stats['Q2Q2']['std_error'], 0],
            [0, 0, integral_stats['Q3Q3']['std_error']]
        ])
        Umat_errors = mfieldmatrixfactors*np.array([
            [integral_stats['U1U1']['std_error'], integral_stats['U2U1']['std_error'], 0],
            [integral_stats['U1U2']['std_error'], integral_stats['U2U2']['std_error'], 0],
            [0, 0, integral_stats['U3U3']['std_error']]
        ])

        # Store matrix errors
        matrix_stats = {
            'Hmat': {'value': Hmat, 'std_error': Hmat_errors},
            'Qmat': {'value': Qmat, 'std_error': Qmat_errors},
            'Umat': {'value': Umat, 'std_error': Umat_errors}
        }

        NormH=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Hmat).astype(complex),complex_dtype))
        NormQ=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Qmat).astype(complex),complex_dtype))
        NormU=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Umat).astype(complex),complex_dtype))
        # second index is the U index
        #physical_yukawas=  (1/(8*np.sqrt(30))) *   2*np.sqrt(2)/4*NormH[0][0].numpy()*np.einsum('ab,cd,bd->ac',NormQ,NormU,m)


        # Calculate normalization matrices with proper error propagation
        NormH_errors = propagate_errors_through_normalization(Hmat, Hmat_errors)
        NormQ_errors = propagate_errors_through_normalization(Qmat, Qmat_errors)
        NormU_errors = propagate_errors_through_normalization(Umat, Umat_errors)

        # Calculate m (holomorphic yukawa) error matrix
        m_errors = np.zeros((3, 3), dtype=complex)
        m = np.zeros((3, 3), dtype=complex)
        m_errorswoH = np.zeros((3, 3), dtype=complex)
        mwoH = np.zeros((3, 3), dtype=complex)
        m_neffs = np.zeros((3, 3), dtype=complex)
        mwoH_neffs = np.zeros((3, 3), dtype=complex)

        # Calculate standard error for each element of the holomorphic yukawa matrix
        # #2 sqrt2 comes out, 4 is |G|
        # # 8 Sqrt30 is the group theoretic factor
        # m= (1/(8*np.sqrt(30))) *   2*np.sqrt(2)/4* np.array(m)
        factor = (1/(8*np.sqrt(30))) * 2*np.sqrt(2)/4

        # For each element, calculate standard error
        for i in range(3):
            for j in range(3):
                # Get the corresponding integrand
                if i == 0 and j == 0:
                    integrand = tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU1)*omega_normalised_to_one
                    integrandwoH = tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ1,vU1)*omega_normalised_to_one
                elif i == 0 and j == 1:
                    integrand = tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU2)*omega_normalised_to_one
                    integrandwoH = tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ1,vU2)*omega_normalised_to_one
                elif i == 0 and j == 2:
                    integrand = tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vQ1,vU3)*omega_normalised_to_one
                    integrandwoH = tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ1,vU3)*omega_normalised_to_one
                elif i == 1 and j == 0:
                    integrand = tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU1)*omega_normalised_to_one
                    integrandwoH = tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ2,vU1)*omega_normalised_to_one
                elif i == 1 and j == 1:
                    integrand = tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU2)*omega_normalised_to_one
                    integrandwoH = tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ2,vU2)*omega_normalised_to_one
                elif i == 1 and j == 2:
                    integrand = tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vQ2,vU3)*omega_normalised_to_one
                    integrandwoH = tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ2,vU3)*omega_normalised_to_one
                elif i == 2 and j == 0:
                    integrand = tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU1)*omega_normalised_to_one
                    integrandwoH = tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ3,vU1)*omega_normalised_to_one
                elif i == 2 and j == 1:
                    integrand = tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU2)*omega_normalised_to_one
                    integrandwoH = tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ3,vU2)*omega_normalised_to_one
                elif i == 2 and j == 2:
                    integrand = tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H2),vH,vQ3,vU3)*omega_normalised_to_one
                    integrandwoH = tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ3,vU3)*omega_normalised_to_one
                # Apply factor and calculate weighted mean and standard error
                scaled_integrand = factor * integrand
                scaled_integrandwoH = factor * integrandwoH
                m_value, m_se, m_eff_n, m_stats = weighted_mean_and_standard_error(scaled_integrand, aux_weights,is_top_form=True)
                m_valueewoH, m_sewoH, m_eff_nwoH, m_statswoH = weighted_mean_and_standard_error(scaled_integrandwoH, aux_weights,is_top_form=True)
                m_errors[i, j] = m_se
                m[i, j] = m_value
                m_errorswoH[i, j] = m_sewoH
                mwoH[i, j] = m_valueewoH
                m_neffs[i, j] = m_eff_n
                mwoH_neffs[i, j] = m_eff_nwoH
                # Store statistics for this matrix element
                if 'm_stats' not in integral_stats:
                    integral_stats['m_stats'] = {}
                if 'm_statswoH' not in integral_stats:
                    integral_stats['m_statswoH'] = {}

                matrix_key = f'm_{i+1}{j+1}'
                integral_stats['m_stats'][matrix_key] = m_stats
                integral_stats['m_statswoH'][matrix_key] = m_statswoH

        holomorphic_Yukawas_trained_and_ref.append(m)
        print("without H * 10**6")
        print(np.round(np.array(mwoH)*10**6,1))
        print("holomorphic Yukawa errors *10**6 (absolute value)")
        print(np.round(m_errorswoH*10**6,1))

        
        print("neffs without H")
        print(np.round(mwoH_neffs,1))

        print('proper calculation*10**6')
        print(np.round(np.array(m)*10**6,1))
  
         # Print holomorphic Yukawa matrix errors
        print("holomorphic Yukawa errors *10**6 (absolute value)")
        print(np.round(m_errors*10**6,1))
        print("neffs")
        print(np.round(m_neffs,1))


        # Use the new function to calculate physical Yukawas and their errors
        physical_yukawas, physical_yukawas_errors = propagate_errors_to_physical_yukawas(
            NormH, NormH_errors, NormQ, NormQ_errors, NormU, NormU_errors, m, m_errors
        )

       
        print("physical_yukawas*10**6")
        print(np.round(physical_yukawas*10**6,1))
        print("physical_yukawas_errors*10**6 (absolute value)")
        print(np.round(np.abs(physical_yukawas_errors)*10**6,1))
        print("physical_yukawas_errors*10**6 (real part)")
        print(np.round(np.real(physical_yukawas_errors)*10**6,1))
        print("physical_yukawas_errors*10**6 (imaginary part)")
        print(np.round(np.imag(physical_yukawas_errors)*10**6,1))
        mats.append(physical_yukawas)
        # no conjugating required, no transposing!
        #check good bundle mterics, compatible
        prodbundles=tf.reduce_mean(H1*H2*H3)
        prodbundlesstddev=tf.math.reduce_std(H1*H2*H3)
        #print("fixed GT factor except for 8/30, fixed the factors of 2 from the integral")
        print("fixed all factors -finished")
        print("good bundle metric" + str(prodbundles) + " and std ddev " + str(prodbundlesstddev))
        print("masses")
        u,s,v = np.linalg.svd(physical_yukawas)
        print(s)
        masses_trained_and_ref.append(s)

        # Calculate singular values and their errors
        singular_value_errors = propagate_errors_to_singular_values(physical_yukawas, physical_yukawas_errors)

        print("masses ± errors")
        for i in range(len(s)):
            print(f"{s[i]:.6e} ± {singular_value_errors[i]:.6e}")

        ## Calculate effective sample size for the full dataset
        #full_eff_n = effective_sample_size(aux_weights[0:n_p]).numpy()
        #print(f"Overall effective sample size: {full_eff_n} (out of {n_p} total points)")

    # Save training results to CSV files


    
    # # Calculate average real and imaginary accuracy metrics
    # avg_rel_real_error = np.mean(np.abs(np.real(physical_yukawas_errors)) / np.abs(physical_yukawas))
    # avg_rel_imag_error = np.mean(np.abs(np.imag(physical_yukawas_errors)) / np.abs(physical_yukawas))

    # # Calculate average real and imaginary accuracy metrics for holomorphic yukawas
    # avg_rel_real_error_holo = np.mean(np.abs(np.real(m_errors)) / np.abs(m))
    # avg_rel_imag_error_holo = np.mean(np.abs(np.imag(m_errors)) / np.abs(m))

    # Save results for this run to unique npz file
    results = {

        'network_params': network_params,
        # Results matrices
        'physical_yukawas': np.array(mats),
        'singular_values': np.array(masses_trained_and_ref),
        'holomorphic_yukawas': np.array(holomorphic_Yukawas_trained_and_ref),
        'holomorphic_yukawas_errors': m_errors,

        # Statistical error information
        'integral_stats': integral_stats,
        'matrix_stats': matrix_stats,
        'physical_yukawas_errors': physical_yukawas_errors,
        'singular_values_errors': singular_value_errors,
        'm_neffs': m_neffs,
        'mwoH_neffs': mwoH_neffs,
        'Qneffs': Qneffs,
        'Uneffs': Uneffs,
        'Hneffs': Hneffs
    }

# Create unique filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{unique_id_or_coeff}"  # Add process ID for uniqueness

    # Save to unique file in results directory
    os.makedirs(foldername + '_results', exist_ok=True)
    npzsavelocation = foldername + '_results/run_' + run_id + '.npz'
    np.savez(npzsavelocation, **results)

    # Save masses to CSV
    import csv
    csv_file = f'{foldername}_results/masses.csv'
    print("saving csv to " + npzsavelocation)

    # Create header if file doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run_id', 'learned_mass1', 'learned_mass2', 'learned_mass3', 
                             'learned_mass1_error', 'learned_mass2_error', 'learned_mass3_error',
                             'ref_mass1', 'ref_mass2', 'ref_mass3',
                             'ref_mass1_error', 'ref_mass2_error', 'ref_mass3_error',
                             'coefficient', 'n_to_integrate'])

    # Append masses for this run
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run_id] + 
                        list(masses_trained_and_ref[0]) + 
                        list(singular_value_errors) +
                        list(masses_trained_and_ref[1]) + 
                        list(singular_value_errors) +  # Using same error estimate for both learned and reference
                        [unique_id_or_coeff] + 
                        [n_p])

    if do_extra_stuff:    
        pass

    return