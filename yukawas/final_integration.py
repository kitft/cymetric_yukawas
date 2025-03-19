import os
import numpy as np
import tensorflow as tf
from cymetric.config import real_dtype, complex_dtype


from auxiliary_funcs import *
from laplacian_funcs import *
import gc
import uuid
from datetime import datetime
from cymetric.models.tfhelper import prepare_tf_basis, train_model
import wandb


def convert_to_nested_tensor_dict(data):
   if isinstance(data, dict) or hasattr(data, 'items'):
      result = {}
      for key, value in data.items():
         if isinstance(value, np.ndarray):
            if value.dtype in [np.complex64, np.complex128] or 'complex' in str(value.dtype):
                print(f'converting {key} to complex')
                result[key] = tf.convert_to_tensor(value, dtype=complex_dtype)
            elif value.dtype in [ np.float32, np.float64,float] or 'float' in str(value.dtype):
                print(f'converting {key} to real')
                result[key] = tf.convert_to_tensor(value, dtype=real_dtype)
            elif value.dtype == np.int32 or 'int' in str(value.dtype):
                print(f'converting {key} to int')
                result[key] = tf.convert_to_tensor(value, dtype=value.dtype)
            else:
               print(f"Unknown dtype for conversion: {value.dtype}, {key}")
               result[key] = value  # Keep original if not a handled dtype
         elif isinstance(value, dict):
            result[key] = convert_to_nested_tensor_dict(value)
         else:
            result[key] = value
      return result
   else:
      return data

def do_integrals(manifold_name_and_data, pg, dataEval, phimodel, betamodel_LB1, betamodel_LB2, betamodel_LB3, HFmodel_vH, HFmodel_vQ3,
                  HFmodel_vU3, HFmodel_vQ1, HFmodel_vQ2, HFmodel_vU1, HFmodel_vU2, network_params, do_extra_stuff = None, run_args=None, dirnameEval=None, result_files_path=None, addtofilename=None):
    #savevecs = not ("nosavevecs" in run_args or "no_save_vecs" in run_args or "no_savevecs" in run_args or "nosave_vecs" in run_args) 
    savevecs =("savevecs" in run_args or "save_vecs" in run_args) 
    loadvecs = ("loadvecs" in run_args or "load_vecs" in run_args)
    loadpullbacks = ("loadpullbacks" in run_args or "load_pullbacks" in run_args)# default true
    savepullbacks =  ("savepullbacks" in run_args or "save_pullbacks" in run_args)# default false
    loadextra = False#("loadextra" in run_args or "load_extra" in run_args) and not ("no_extra" in run_args or "noextra" in run_args)
    saveextra = ("saveextra" in run_args or "save_extra" in run_args)

    (_, kmoduli, _, _, type_folder, unique_id_or_coeff, manifold_name, data_path) = manifold_name_and_data
    points64=tf.concat((dataEval['X_train'], dataEval['X_val']),axis=0)
    ytrain64 = tf.concat((dataEval['y_train'], dataEval['y_val']),axis=0)
    n_p = len(points64)# len(dataEval['X_train'])# len(points64)
    real_pts=tf.cast(points64[:n_p],real_dtype)
    volCY_from_Om=tf.reduce_mean(ytrain64[:n_p,0])/6 #this is the actual volume of the CY computed from Omega.since J^J^J^ = K Om^Ombar?? not totally sure here:
    pointsComplex=tf.cast(point_vec_to_complex(points64[:n_p]),complex_dtype)# we have to cast to complex64 from complex128 sometimes. Just for safety. point_vec_to_complex does the obvious thing
    weights=tf.cast(ytrain64[:n_p,0],real_dtype)
    omegasquared=tf.cast(ytrain64[:n_p,1],real_dtype)
    mulweightsby=omegasquared/tf.reduce_mean(omegasquared)# arbitrary normalisation

    print("\n do analysis: \n\n\n")

    #Vol_from_dijk_J=pg.get_volume_from_intersections(kmoduli)
    #Vol_reference_dijk_with_k_is_1=pg.get_volume_from_intersections(np.ones_like(kmoduli)) 
    #volCY_from_Om=tf.reduce_mean(dataEval['y_train'][:n_p,0])
    print("Compute holomorphic Yukawas")
    #consider omega normalisation
    if not loadvecs:
        omega = tf.cast(batch_process_helper_func(pg.holomorphic_volume_form, [pointsComplex], batch_indices=[0], batch_size=100000),complex_dtype)
    else:
        filename = os.path.join(data_path,type_folder, f"vectors_fc_{unique_id_or_coeff}_{n_p}_trained_{False}.npz")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        data = np.load(filename)
        omega = tf.cast(data['omega'],complex_dtype)
    # Verify that dataEval['y_train'][:n_p,1] equals |omega|^2
    omega_abs_squared_calculated = tf.math.real(omega * tf.math.conj(omega))
    assert tf.reduce_all(tf.abs(omega_abs_squared_calculated[:100] - omegasquared[:100]) < 1e-5), "First elements of dataEval['y_train'][:,1] should equal |omega|^2"
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

     
        # Compute or load pullbacks
    pullbacks_filename = os.path.join(data_path, type_folder, f"vectors_fc_{unique_id_or_coeff}_{n_p}_trained_{str(False)}_pullbacks.npz")
    if loadpullbacks:
        try:
            dataextra = np.load(pullbacks_filename, allow_pickle=True)
            dataextra = convert_to_nested_tensor_dict(dataextra)
            pullbacks = dataextra['pullbacks']
            if len(pullbacks)!=len(pointsComplex):
                raise ValueError("Pullbacks have the wrong length")
            print("Loaded pullbacks from saved file")
        except FileNotFoundError:
            print("File not found. Computing pullbacks...")
            #raise Exception("Pullbacks file not found.")
            pullbacks = tf.cast(batch_process_helper_func(pg.pullbacks, [pointsComplex], batch_indices=[0], batch_size=100000), complex_dtype)
            
        # Save pullbacks to file
        np.savez(pullbacks_filename, pullbacks=pullbacks)
        print(f"Saved pullbacks to {pullbacks_filename}")
    else:
        # Compute pullbacks if not loading from file
        pullbacks = tf.cast(batch_process_helper_func(pg.pullbacks, [pointsComplex], batch_indices=[0], batch_size=100000), complex_dtype)
        if savepullbacks:
            np.savez(pullbacks_filename, pullbacks=pullbacks)
            print(f"Saved pullbacks to {pullbacks_filename}")
    

    if not loadvecs:
        mets_bare = batch_process_helper_func(phimodel.fubini_study_pb, (real_pts, pullbacks), batch_indices=(0,1), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch, kwargs={'ts':tf.cast(kmoduli,complex_dtype)})
        vH_bare = batch_process_helper_func(HFmodel_vH.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch, kwargs={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
        vQ3_bare = batch_process_helper_func(HFmodel_vQ3.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch, kwargs={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
        vU3_bare = batch_process_helper_func(HFmodel_vU3.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch, kwargs={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
        vQ1_bare = batch_process_helper_func(HFmodel_vQ1.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch, kwargs={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
        vQ2_bare = batch_process_helper_func(HFmodel_vQ2.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch, kwargs={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
        vU1_bare = batch_process_helper_func(HFmodel_vU1.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch, kwargs={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
        vU2_bare = batch_process_helper_func(HFmodel_vU2.uncorrected_FS_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True, actually_batch=actually_batch, kwargs={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])


    mats=[]
    masses_trained_and_ref=[]
    use_trained = False
    holomorphic_Yukawas_trained_and_ref=[]
    holomorphic_Yukawas_trained_and_ref_errors=[]
    topological_data={}
    for use_trained in [True,False]:
        if loadvecs:
            filename = os.path.join(data_path,type_folder, f"vectors_fc_{unique_id_or_coeff}_{n_p}_trained_{use_trained}.npz")
            data = np.load(filename, allow_pickle=True)
            data = convert_to_nested_tensor_dict(data)
            vH_bare, vQ3_bare, vU3_bare, vQ1_bare, vQ2_bare, vU1_bare, vU2_bare = data['vH.barevec'], data['vQ3.barevec'], data['vU3.barevec'], data['vQ1.barevec'], data['vQ2.barevec'], data['vU1.barevec'], data['vU2.barevec']
            H1, H2, H3, LB1c, LB2c, LB3c, vH, vQ3, vU3, vQ1, vQ2, vU1, vU2, mets = data['H1'], data['H2'], data['H3'], data['LB1c'], data['LB2c'], data['LB3c'], data['vH.vec'], data['vQ3.vec'], data['vU3.vec'], data['vQ1.vec'], data['vQ2.vec'], data['vU1.vec'], data['vU2.vec'], data['mets']
            hvHb, hvQ3b, hvU3b, hvQ1b, hvQ2b, hvU1b, hvU2b = data['vH.Hconj'], data['vQ3.Hconj'], data['vU3.Hconj'], data['vQ1.Hconj'], data['vQ2.Hconj'], data['vU1.Hconj'], data['vU2.Hconj']
            print("loaded data", flush=True)
        else:
            print("using trained? " + str(use_trained))
            if use_trained:
                mets = batch_process_helper_func(phimodel, (real_pts,), batch_indices=(0,), batch_size=10000, compile_func=True,kwargs = {'pb':pullbacks}, batch_kwargs_keys=['pb'])
                print('got mets', flush=True)
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
                vH = batch_process_helper_func(HFmodel_vH.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True,kwargs ={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
                hvHb = tf.einsum('x,xb->xb', LB1c, tf.math.conj(vH))
                print('got vH', flush=True)

                vQ3 = batch_process_helper_func(HFmodel_vQ3.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True,kwargs ={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
                hvQ3b = tf.einsum('x,xb->xb', LB2c, tf.math.conj(vQ3))
                print('got vQ3', flush=True)

                vU3 = batch_process_helper_func(HFmodel_vU3.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True,kwargs ={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
                hvU3b = tf.einsum('x,xb->xb', LB2c, tf.math.conj(vU3))
                print('got vU3', flush=True)

                vQ1 = batch_process_helper_func(HFmodel_vQ1.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True,kwargs ={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
                hvQ1b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vQ1))
                print('got vQ1', flush=True)

                vQ2 = batch_process_helper_func(HFmodel_vQ2.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True,kwargs ={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
                hvQ2b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vQ2))
                print('got vQ2', flush=True)

                vU1 = batch_process_helper_func(HFmodel_vU1.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True,kwargs ={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
                hvU1b = tf.einsum('x,xb->xb', LB3c, tf.math.conj(vU1))
                print('got vU1', flush=True)

                vU2 = batch_process_helper_func(HFmodel_vU2.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True,kwargs ={'pullbacks_holo':pullbacks}, batch_kwargs_keys=['pullbacks_holo'])
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
        
        # Save vectors to file if requested
        if savevecs:
            print("Saving vectors to file...")
            save_data = {
                'H1': H1.numpy(),
                'H2': H2.numpy(),
                'H3': H3.numpy(),
                'LB1c': LB1c.numpy(),
                'LB2c': LB2c.numpy(),
                'LB3c': LB3c.numpy(),
                'vH.vec': vH.numpy(),
                'vH.Hconj': hvHb.numpy(),
                'vH.barevec': vH_bare.numpy(),
                'vQ3.vec': vQ3.numpy(),
                'vQ3.Hconj': hvQ3b.numpy(),
                'vQ3.barevec': vQ3_bare.numpy(),
                'vU3.vec': vU3.numpy(),
                'vU3.Hconj': hvU3b.numpy(),
                'vU3.barevec': vU3_bare.numpy(),
                'vQ1.vec': vQ1.numpy(),
                'vQ1.Hconj': hvQ1b.numpy(),
                'vQ1.barevec': vQ1_bare.numpy(),
                'vQ2.vec': vQ2.numpy(),
                'vQ2.Hconj': hvQ2b.numpy(),
                'vQ2.barevec': vQ2_bare.numpy(),
                'vU1.vec': vU1.numpy(),
                'vU1.Hconj': hvU1b.numpy(),
                'vU1.barevec': vU1_bare.numpy(),
                'vU2.vec': vU2.numpy(),
                'vU2.Hconj': hvU2b.numpy(),
                'vU2.barevec': vU2_bare.numpy(),
                'mets': mets.numpy(),
                'mets_bare': mets_bare.numpy(),
                'omega': omega.numpy(),
            }

            # Create filename with unique identifier
            filename = os.path.join(data_path,type_folder, f"vectors_fc_{unique_id_or_coeff}_{n_p}_trained_{use_trained}.npz")
            np.savez(filename, **save_data)
            print(f"(Trained? {use_trained}) vectors saved to {filename}")

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
        HuHu, HuHu_se, HuHu_eff_n, integral_stats['HuHu'] = weighted_mean_and_standard_error(HuHu_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
        print("(Hu,Hu) = " + str(HuHu) + " ± " + str(HuHu_se) + " (eff. n = " + str(HuHu_eff_n) + ")")
        print(f"  Real part: {tf.math.real(HuHu).numpy():.6e} ± {tf.math.real(HuHu_se).numpy():.6e}")
        print(f"  Imag part: {tf.math.imag(HuHu).numpy():.6e} ± {tf.math.imag(HuHu_se).numpy():.6e}")

        # Calculate Q3Q3 and its standard error
        Q3Q3_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ3,hvQ3b,lc_c,lc_c)
        Q3Q3, Q3Q3_se, Q3Q3_eff_n, integral_stats['Q3Q3'] = weighted_mean_and_standard_error(Q3Q3_integrand, aux_weights, is_top_form=True,mulweightsby=mulweightsby)
        print("(Q3,Q3) = " + str(Q3Q3) + " ± " + str(Q3Q3_se) + " (eff. n = " + str(Q3Q3_eff_n) + ")")

        # Calculate U3U3 and its standard error
        U3U3_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU3,hvU3b,lc_c,lc_c)
        U3U3, U3U3_se, U3U3_eff_n, integral_stats['U3U3'] = weighted_mean_and_standard_error(U3U3_integrand, aux_weights, is_top_form=True, mulweightsby=mulweightsby)
        print("(U3,U3) = " + str(U3U3) + " ± " + str(U3U3_se) + " (eff. n = " + str(U3U3_eff_n) + ")")

        # Calculate U1U1 and its standard error
        U1U1_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU1,hvU1b,lc_c,lc_c)
        U1U1, U1U1_se, U1U1_eff_n, integral_stats['U1U1'] = weighted_mean_and_standard_error(U1U1_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
        print("(U1,U1) = " + str(U1U1) + " ± " + str(U1U1_se) + " (eff. n = " + str(U1U1_eff_n) + ")")

        # Calculate U2U2 and its standard error
        U2U2_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU2,hvU2b,lc_c,lc_c)
        U2U2, U2U2_se, U2U2_eff_n, integral_stats['U2U2'] = weighted_mean_and_standard_error(U2U2_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
        print("(U2,U2) = " + str(U2U2) + " ± " + str(U2U2_se) + " (eff. n = " + str(U2U2_eff_n) + ")")

        # Calculate Q1Q1 and its standard error
        Q1Q1_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ1,hvQ1b,lc_c,lc_c)
        Q1Q1, Q1Q1_se, Q1Q1_eff_n, integral_stats['Q1Q1'] = weighted_mean_and_standard_error(Q1Q1_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
        print("(Q1,Q1) = " + str(Q1Q1) + " ± " + str(Q1Q1_se) + " (eff. n = " + str(Q1Q1_eff_n) + ")")

        # Calculate Q2Q2 and its standard error
        Q2Q2_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ2,hvQ2b,lc_c,lc_c)
        Q2Q2, Q2Q2_se, Q2Q2_eff_n, integral_stats['Q2Q2'] = weighted_mean_and_standard_error(Q2Q2_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
        print("(Q2,Q2) = " + str(Q2Q2) + " ± " + str(Q2Q2_se) + " (eff. n = " + str(Q2Q2_eff_n) + ")")

        print("The field mixings:")

        # Calculate Q1Q2 and its standard error
        Q1Q2_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ1,hvQ2b,lc_c,lc_c)
        Q1Q2, Q1Q2_se, Q1Q2_eff_n, integral_stats['Q1Q2'] = weighted_mean_and_standard_error(Q1Q2_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
        print("(Q1,Q2) = " + str(Q1Q2) + " ± " + str(Q1Q2_se) + " (eff. n = " + str(Q1Q2_eff_n) + ")")

        # Calculate Q2Q1 and its standard error
        Q2Q1_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vQ2,hvQ1b,lc_c,lc_c)
        Q2Q1, Q2Q1_se, Q2Q1_eff_n, integral_stats['Q2Q1'] = weighted_mean_and_standard_error(Q2Q1_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
        print("(Q2,Q1) = " + str(Q2Q1) + " ± " + str(Q2Q1_se) + " (eff. n = " + str(Q2Q1_eff_n) + ")")

        # Calculate U1U2 and its standard error
        U1U2_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU1,hvU2b,lc_c,lc_c)
        U1U2, U1U2_se, U1U2_eff_n, integral_stats['U1U2'] = weighted_mean_and_standard_error(U1U2_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
        print("(U1,U2) = " + str(U1U2) + " ± " + str(U1U2_se) + " (eff. n = " + str(U1U2_eff_n) + ")")

        # Calculate U2U1 and its standard error
        U2U1_integrand = (-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets,mets,vU2,hvU1b,lc_c,lc_c)
        U2U1, U2U1_se, U2U1_eff_n, integral_stats['U2U1'] = weighted_mean_and_standard_error(U2U1_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
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
            integrand_Q3U2 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3_bare,vU2_bare)*omega_normalised_to_one
            integrand_bare_Q3U2_vH = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH-vH_bare,vQ3_bare,vU2_bare)*omega_normalised_to_one
            integrand_bare_Q3U2_vQ3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3-vQ3_bare,vU2_bare)*omega_normalised_to_one
            integrand_bare_Q3U2_vU2 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3_bare,vU2-vU2_bare)*omega_normalised_to_one

            # Q1U3 combination
            integrand_Q1U3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ1_bare,vU3_bare)*omega_normalised_to_one
            integrand_bare_Q1U3_vH = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH-vH_bare,vQ1_bare,vU3_bare)*omega_normalised_to_one
            integrand_bare_Q1U3_vQ1 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ1-vQ1_bare,vU3_bare)*omega_normalised_to_one
            integrand_bare_Q1U3_vU3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ1_bare,vU3-vU3_bare)*omega_normalised_to_one

            # Q2U3 combination
            integrand_Q2U3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ2_bare,vU3_bare)*omega_normalised_to_one
            integrand_bare_Q2U3_vH = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH-vH_bare,vQ2_bare,vU3_bare)*omega_normalised_to_one
            integrand_bare_Q2U3_vQ2 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ2-vQ2_bare,vU3_bare)*omega_normalised_to_one
            integrand_bare_Q2U3_vU3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH_bare,vQ2_bare,vU3-vU3_bare)*omega_normalised_to_one

            # Q3U1 combination
            integrand_Q3U1 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3_bare,vU1_bare)*omega_normalised_to_one
            integrand_bare_Q3U1_vH = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH-vH_bare,vQ3_bare,vU1_bare)*omega_normalised_to_one
            integrand_bare_Q3U1_vQ3 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3-vQ3_bare,vU1_bare)*omega_normalised_to_one
            integrand_bare_Q3U1_vU1 = factor * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH_bare,vQ3_bare,vU1-vU1_bare)*omega_normalised_to_one

            # Calculate statistics for all integrands
            int_Q3U2, int_Q3U2_se, int_Q3U2_eff_n, int_Q3U2_stats = weighted_mean_and_standard_error(integrand_Q3U2, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q3U2_vH, int_bare_Q3U2_vH_se, int_bare_Q3U2_vH_eff_n, int_bare_Q3U2_vH_stats = weighted_mean_and_standard_error(integrand_bare_Q3U2_vH, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q3U2_vQ3, int_bare_Q3U2_vQ3_se, int_bare_Q3U2_vQ3_eff_n, int_bare_Q3U2_vQ3_stats = weighted_mean_and_standard_error(integrand_bare_Q3U2_vQ3, aux_weights, mulweightsby=mulweightsby)  
            int_bare_Q3U2_vU2, int_bare_Q3U2_vU2_se, int_bare_Q3U2_vU2_eff_n, int_bare_Q3U2_vU2_stats = weighted_mean_and_standard_error(integrand_bare_Q3U2_vU2, aux_weights, mulweightsby=mulweightsby)

            int_Q1U3, int_Q1U3_se, int_Q1U3_eff_n, int_Q1U3_stats = weighted_mean_and_standard_error(integrand_Q1U3, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q1U3_vH, int_bare_Q1U3_vH_se, int_bare_Q1U3_vH_eff_n, int_bare_Q1U3_vH_stats = weighted_mean_and_standard_error(integrand_bare_Q1U3_vH, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q1U3_vQ1, int_bare_Q1U3_vQ1_se, int_bare_Q1U3_vQ1_eff_n, int_bare_Q1U3_vQ1_stats = weighted_mean_and_standard_error(integrand_bare_Q1U3_vQ1, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q1U3_vU3, int_bare_Q1U3_vU3_se, int_bare_Q1U3_vU3_eff_n, int_bare_Q1U3_vU3_stats = weighted_mean_and_standard_error(integrand_bare_Q1U3_vU3, aux_weights, mulweightsby=mulweightsby)

            int_Q2U3, int_Q2U3_se, int_Q2U3_eff_n, int_Q2U3_stats = weighted_mean_and_standard_error(integrand_Q2U3, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q2U3_vH, int_bare_Q2U3_vH_se, int_bare_Q2U3_vH_eff_n, int_bare_Q2U3_vH_stats = weighted_mean_and_standard_error(integrand_bare_Q2U3_vH, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q2U3_vQ2, int_bare_Q2U3_vQ2_se, int_bare_Q2U3_vQ2_eff_n, int_bare_Q2U3_vQ2_stats = weighted_mean_and_standard_error(integrand_bare_Q2U3_vQ2, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q2U3_vU3, int_bare_Q2U3_vU3_se, int_bare_Q2U3_vU3_eff_n, int_bare_Q2U3_vU3_stats = weighted_mean_and_standard_error(integrand_bare_Q2U3_vU3, aux_weights, mulweightsby=mulweightsby)

            int_Q3U1, int_Q3U1_se, int_Q3U1_eff_n, int_Q3U1_stats = weighted_mean_and_standard_error(integrand_Q3U1, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q3U1_vH, int_bare_Q3U1_vH_se, int_bare_Q3U1_vH_eff_n, int_bare_Q3U1_vH_stats = weighted_mean_and_standard_error(integrand_bare_Q3U1_vH, aux_weights, mulweightsby=mulweightsby)
            int_bare_Q3U1_vQ3, int_bare_Q3U1_vQ3_se, int_bare_Q3U1_vQ3_eff_n, int_bare_Q3U1_vQ3_stats = weighted_mean_and_standard_error(integrand_bare_Q3U1_vQ3, aux_weights, mulweightsby=mulweightsby)  
            int_bare_Q3U1_vU1, int_bare_Q3U1_vU1_se, int_bare_Q3U1_vU1_eff_n, int_bare_Q3U1_vU1_stats = weighted_mean_and_standard_error(integrand_bare_Q3U1_vU1, aux_weights, mulweightsby=mulweightsby)
            print("\n Q3U2:")
            print(f"int_Q3U2 = {int_Q3U2:.6e} ± {int_Q3U2_se:.6e} ({np.round(int_Q3U2_stats['z_score'],2)}σ)")
            print(f"int_bare_Q3U2_vH = {int_bare_Q3U2_vH:.6e} ± {int_bare_Q3U2_vH_se:.6e} ({np.round(int_bare_Q3U2_vH_stats['z_score'],2)}σ)")
            print(f"int_bare_Q3U2_vQ3 = {int_bare_Q3U2_vQ3:.6e} ± {int_bare_Q3U2_vQ3_se:.6e} ({np.round(int_bare_Q3U2_vQ3_stats['z_score'],2)}σ)")
            print(f"int_bare_Q3U2_vU2 = {int_bare_Q3U2_vU2:.6e} ± {int_bare_Q3U2_vU2_se:.6e} ({np.round(int_bare_Q3U2_vU2_stats['z_score'],2)}σ)")
            print("\n Q1U3:")
            print(f"int_Q1U3 = {int_Q1U3:.6e} ± {int_Q1U3_se:.6e} ({np.round(int_Q1U3_stats['z_score'],2)}σ)")
            print(f"int_bare_Q1U3_vH = {int_bare_Q1U3_vH:.6e} ± {int_bare_Q1U3_vH_se:.6e} ({np.round(int_bare_Q1U3_vH_stats['z_score'],2)}σ)")
            print(f"int_bare_Q1U3_vQ1 = {int_bare_Q1U3_vQ1:.6e} ± {int_bare_Q1U3_vQ1_se:.6e} ({np.round(int_bare_Q1U3_vQ1_stats['z_score'],2)}σ)")
            print(f"int_bare_Q1U3_vU3 = {int_bare_Q1U3_vU3:.6e} ± {int_bare_Q1U3_vU3_se:.6e} ({np.round(int_bare_Q1U3_vU3_stats['z_score'],2)}σ)") 
            print("\n Q2U3:")
            print(f"int_Q2U3 = {int_Q2U3:.6e} ± {int_Q2U3_se:.6e} ({np.round(int_Q2U3_stats['z_score'],2)}σ)")
            print(f"int_bare_Q2U3_vH = {int_bare_Q2U3_vH:.6e} ± {int_bare_Q2U3_vH_se:.6e} ({np.round(int_bare_Q2U3_vH_stats['z_score'],2)}σ)")
            print(f"int_bare_Q2U3_vQ2 = {int_bare_Q2U3_vQ2:.6e} ± {int_bare_Q2U3_vQ2_se:.6e} ({np.round(int_bare_Q2U3_vQ2_stats['z_score'],2)}σ)")
            print(f"int_bare_Q2U3_vU3 = {int_bare_Q2U3_vU3:.6e} ± {int_bare_Q2U3_vU3_se:.6e} ({np.round(int_bare_Q2U3_vU3_stats['z_score'],2)}σ)")
            print("\n Q3U1:")
            print(f"int_Q3U1 = {int_Q3U1:.6e} ± {int_Q3U1_se:.6e} ({np.round(int_Q3U1_stats['z_score'],2)}σ)")
            print(f"int_bare_Q3U1_vH = {int_bare_Q3U1_vH:.6e} ± {int_bare_Q3U1_vH_se:.6e} ({np.round(int_bare_Q3U1_vH_stats['z_score'],2)}σ)")
            print(f"int_bare_Q3U1_vQ3 = {int_bare_Q3U1_vQ3:.6e} ± {int_bare_Q3U1_vQ3_se:.6e} ({np.round(int_bare_Q3U1_vQ3_stats['z_score'],2)}σ)")
            print(f"int_bare_Q3U1_vU1 = {int_bare_Q3U1_vU1:.6e} ± {int_bare_Q3U1_vU1_se:.6e} ({np.round(int_bare_Q3U1_vU1_stats['z_score'],2)}σ)")
            
            # Log all means and standard errors to wandb
            if wandb.run is not None:
                # Split complex values into real and imaginary parts for wandb logging
                # Use absolute value for error terms
                if use_trained:
                    prefix = "trained"
                else:
                    prefix = "bare"
                wandb.log({
                    f"int_Q3U2_real_{prefix}": float(np.real(int_Q3U2)),
                    f"int_Q3U2_imag_{prefix}": float(np.imag(int_Q3U2)),
                    f"int_Q3U2_se_{prefix}": float(np.abs(int_Q3U2_se)),
                    f"int_Q3U2_z_score_{prefix}": float(int_Q3U2_stats['z_score']),
                    
                    f"int_bare_Q3U2_vH_real_{prefix}": float(np.real(int_bare_Q3U2_vH)),
                    f"int_bare_Q3U2_vH_imag_{prefix}": float(np.imag(int_bare_Q3U2_vH)),
                    f"int_bare_Q3U2_vH_se_{prefix}": float(np.abs(int_bare_Q3U2_vH_se)),
                    f"int_bare_Q3U2_vH_z_score_{prefix}": float(int_bare_Q3U2_vH_stats['z_score']),
                    
                    f"int_bare_Q3U2_vQ3_real_{prefix}": float(np.real(int_bare_Q3U2_vQ3)),
                    f"int_bare_Q3U2_vQ3_imag_{prefix}": float(np.imag(int_bare_Q3U2_vQ3)),
                    f"int_bare_Q3U2_vQ3_se_{prefix}": float(np.abs(int_bare_Q3U2_vQ3_se)),
                    f"int_bare_Q3U2_vQ3_z_score_{prefix}": float(int_bare_Q3U2_vQ3_stats['z_score']),
                    
                    f"int_bare_Q3U2_vU2_real_{prefix}": float(np.real(int_bare_Q3U2_vU2)),
                    f"int_bare_Q3U2_vU2_imag_{prefix}": float(np.imag(int_bare_Q3U2_vU2)),
                    f"int_bare_Q3U2_vU2_se_{prefix}": float(np.abs(int_bare_Q3U2_vU2_se)),
                    f"int_bare_Q3U2_vU2_z_score_{prefix}": float(int_bare_Q3U2_vU2_stats['z_score']),
                    
                    f"int_Q1U3_real_{prefix}": float(np.real(int_Q1U3)),
                    f"int_Q1U3_imag_{prefix}": float(np.imag(int_Q1U3)),
                    f"int_Q1U3_se_{prefix}": float(np.abs(int_Q1U3_se)),
                    f"int_Q1U3_z_score_{prefix}": float(int_Q1U3_stats['z_score']),
                    
                    f"int_bare_Q1U3_vH_real_{prefix}": float(np.real(int_bare_Q1U3_vH)),
                    f"int_bare_Q1U3_vH_imag_{prefix}": float(np.imag(int_bare_Q1U3_vH)),
                    f"int_bare_Q1U3_vH_se_{prefix}": float(np.abs(int_bare_Q1U3_vH_se)),
                    f"int_bare_Q1U3_vH_z_score_{prefix}": float(int_bare_Q1U3_vH_stats['z_score']),
                    
                    f"int_bare_Q1U3_vQ1_real_{prefix}": float(np.real(int_bare_Q1U3_vQ1)),
                    f"int_bare_Q1U3_vQ1_imag_{prefix}": float(np.imag(int_bare_Q1U3_vQ1)),
                    f"int_bare_Q1U3_vQ1_se_{prefix}": float(np.abs(int_bare_Q1U3_vQ1_se)),
                    f"int_bare_Q1U3_vQ1_z_score_{prefix}": float(int_bare_Q1U3_vQ1_stats['z_score']),
                    
                    f"int_bare_Q1U3_vU3_real_{prefix}": float(np.real(int_bare_Q1U3_vU3)),
                    f"int_bare_Q1U3_vU3_imag_{prefix}": float(np.imag(int_bare_Q1U3_vU3)),
                    f"int_bare_Q1U3_vU3_se_{prefix}": float(np.abs(int_bare_Q1U3_vU3_se)),
                    f"int_bare_Q1U3_vU3_z_score_{prefix}": float(int_bare_Q1U3_vU3_stats['z_score']),
                    
                    f"int_Q2U3_real_{prefix}": float(np.real(int_Q2U3)),
                    f"int_Q2U3_imag_{prefix}": float(np.imag(int_Q2U3)),
                    f"int_Q2U3_se_{prefix}": float(np.abs(int_Q2U3_se)),
                    f"int_Q2U3_z_score_{prefix}": float(int_Q2U3_stats['z_score']),
                    
                    f"int_bare_Q2U3_vH_real_{prefix}": float(np.real(int_bare_Q2U3_vH)),
                    f"int_bare_Q2U3_vH_imag_{prefix}": float(np.imag(int_bare_Q2U3_vH)),
                    f"int_bare_Q2U3_vH_se_{prefix}": float(np.abs(int_bare_Q2U3_vH_se)),
                    f"int_bare_Q2U3_vH_z_score_{prefix}": float(int_bare_Q2U3_vH_stats['z_score']),
                    
                    f"int_bare_Q2U3_vQ2_real_{prefix}": float(np.real(int_bare_Q2U3_vQ2)),
                    f"int_bare_Q2U3_vQ2_imag_{prefix}": float(np.imag(int_bare_Q2U3_vQ2)),
                    f"int_bare_Q2U3_vQ2_se_{prefix}": float(np.abs(int_bare_Q2U3_vQ2_se)),
                    f"int_bare_Q2U3_vQ2_z_score_{prefix}": float(int_bare_Q2U3_vQ2_stats['z_score']),
                    
                    f"int_bare_Q2U3_vU3_real_{prefix}": float(np.real(int_bare_Q2U3_vU3)),
                    f"int_bare_Q2U3_vU3_imag_{prefix}": float(np.imag(int_bare_Q2U3_vU3)),
                    f"int_bare_Q2U3_vU3_se_{prefix}": float(np.abs(int_bare_Q2U3_vU3_se)),
                    f"int_bare_Q2U3_vU3_z_score_{prefix}": float(int_bare_Q2U3_vU3_stats['z_score']),
                    
                    f"int_Q3U1_real_{prefix}": float(np.real(int_Q3U1)),
                    f"int_Q3U1_imag_{prefix}": float(np.imag(int_Q3U1)),
                    f"int_Q3U1_se_{prefix}": float(np.abs(int_Q3U1_se)),
                    f"int_Q3U1_z_score_{prefix}": float(int_Q3U1_stats['z_score']),
                    
                    f"int_bare_Q3U1_vH_real_{prefix}": float(np.real(int_bare_Q3U1_vH)),
                    f"int_bare_Q3U1_vH_imag_{prefix}": float(np.imag(int_bare_Q3U1_vH)),
                    f"int_bare_Q3U1_vH_se_{prefix}": float(np.abs(int_bare_Q3U1_vH_se)),
                    f"int_bare_Q3U1_vH_z_score_{prefix}": float(int_bare_Q3U1_vH_stats['z_score']),
                    
                    f"int_bare_Q3U1_vQ3_real_{prefix}": float(np.real(int_bare_Q3U1_vQ3)),
                    f"int_bare_Q3U1_vQ3_imag_{prefix}": float(np.imag(int_bare_Q3U1_vQ3)),
                    f"int_bare_Q3U1_vQ3_se_{prefix}": float(np.abs(int_bare_Q3U1_vQ3_se)),
                    f"int_bare_Q3U1_vQ3_z_score_{prefix}": float(int_bare_Q3U1_vQ3_stats['z_score']),
                    
                    f"int_bare_Q3U1_vU1_real_{prefix}": float(np.real(int_bare_Q3U1_vU1)),
                    f"int_bare_Q3U1_vU1_imag_{prefix}": float(np.imag(int_bare_Q3U1_vU1)),
                    f"int_bare_Q3U1_vU1_se_{prefix}": float(np.abs(int_bare_Q3U1_vU1_se)),
                    f"int_bare_Q3U1_vU1_z_score_{prefix}": float(int_bare_Q3U1_vU1_stats['z_score'])
                })
                    "actual": [int_Q1U3, int_Q1U3_se, int_Q1U3_stats['z_score']],
                    "vHsection": [int_bare_Q1U3_vH, int_bare_Q1U3_vH_se, int_bare_Q1U3_vH_stats['z_score']],
                    "Q1section": [int_bare_Q1U3_vQ1, int_bare_Q1U3_vQ1_se, int_bare_Q1U3_vQ1_stats['z_score']],
                    "U3section": [int_bare_Q1U3_vU3, int_bare_Q1U3_vU3_se, int_bare_Q1U3_vU3_stats['z_score']]
                },
                "Q2U3": {
                    "actual": [int_Q2U3, int_Q2U3_se, int_Q2U3_stats['z_score']],
                    "vHsection": [int_bare_Q2U3_vH, int_bare_Q2U3_vH_se, int_bare_Q2U3_vH_stats['z_score']],
                    "Q2section": [int_bare_Q2U3_vQ2, int_bare_Q2U3_vQ2_se, int_bare_Q2U3_vQ2_stats['z_score']],
                    "U3section": [int_bare_Q2U3_vU3, int_bare_Q2U3_vU3_se, int_bare_Q2U3_vU3_stats['z_score']]
                },
                "Q3U1": {
                    "actual": [int_Q3U1, int_Q3U1_se, int_Q3U1_stats['z_score']],
                    "vHsection": [int_bare_Q3U1_vH, int_bare_Q3U1_vH_se, int_bare_Q3U1_vH_stats['z_score']],
                    "Q3section": [int_bare_Q3U1_vQ3, int_bare_Q3U1_vQ3_se, int_bare_Q3U1_vQ3_stats['z_score']],
                    "U1section": [int_bare_Q3U1_vU1, int_bare_Q3U1_vU1_se, int_bare_Q3U1_vU1_stats['z_score']]
                }
            
            topological_data[prefix] = topological_data_toadd
            

            print("--------------------------------\n\n\n\n\n\n\n\n")
            # Calculate laplacians for each model
            print("\nComputing and integrating laplacians for each model...")
           
            # Calculate laplacian for vH model
            
            from custom_networks import BiholoModelFuncGENERALforHYMinv3
            tf.random.set_seed(0)
            #mock_model = BiholoModelFuncGENERALforHYMinv3([10,2], pg.BASIS,stddev=0.1,use_zero_network=False,use_symmetry_reduced_TQ=False)
            invmetrics = tf.linalg.inv(mets)
            laplacians_filename = os.path.join(data_path, type_folder, f"vectors_fc_{unique_id_or_coeff}_{n_p}_trained_{use_trained}_laplacians.npz")
            network_shape = [256,16,16,1]
            BASIS = prepare_tf_basis(np.load(os.path.join(dirnameEval, 'basis.pickle'), allow_pickle=True))
            import random
            seed = 0
            tf.keras.utils.set_random_seed(seed)# equivalent to np, random, torch all in one
            #mock_model = BiholoModelFuncGENERALforHYMinv3(network_shape, BASIS, stddev=0.1, use_zero_network=False, use_symmetry_reduced_TQ=False)
            def mock_model(x, **kwargs):
                cpoints = point_vec_to_complex(x)
                kappa1 = tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,0:2])**2, axis=-1),real_dtype)
                x0,x1 = cpoints[:,0], cpoints[:,1]
                x0bar,x1bar = tf.math.conj(x0), tf.math.conj(x1)

                return tf.math.abs(x0*x0bar)/kappa1
            def mock_model_2(x, **kwargs):
                cpoints = point_vec_to_complex(x)
                kappa2 = tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,2:4])**2, axis=-1),real_dtype)
                y0,y1 = cpoints[:,2], cpoints[:,3]
                y0bar,y1bar = tf.math.conj(y0), tf.math.conj(y1)

                return tf.math.abs(y0*y0bar)/kappa2

            def mock_model_3(x, **kwargs):
                cpoints = point_vec_to_complex(x)
                kappa3 = tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,4:6])**2, axis=-1),real_dtype)
                z0,z1 = cpoints[:,4], cpoints[:,5]
                z0bar,z1bar = tf.math.conj(z0), tf.math.conj(z1)

                return tf.math.abs(z0*z0bar)/kappa3
            def mock_model_4(x, **kwargs):
                cpoints = point_vec_to_complex(x)
                kappa4 = tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,6:8])**2, axis=-1),real_dtype) 
                w0,w1 = cpoints[:,6], cpoints[:,7]
                w0bar,w1bar = tf.math.conj(w0), tf.math.conj(w1)

                return tf.math.abs(w0*w0bar)/kappa4

            def mock_model_complex1(x, **kwargs):
                cpoints = point_vec_to_complex(x)
                kappa1 = tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,0:2])**2, axis=-1),complex_dtype) 
                x0,x1 = cpoints[:,0], cpoints[:,1]
                x0bar,x1bar = tf.math.conj(x0), tf.math.conj(x1)

                return x0*x1bar/kappa1
            def mock_model_complex2(x, **kwargs):
                cpoints = point_vec_to_complex(x)
                kappa2 = tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,2:4])**2, axis=-1),complex_dtype) 
                y0,y1 = cpoints[:,2], cpoints[:,3]
                y0bar,y1bar = tf.math.conj(y0), tf.math.conj(y1)

                return y0*y1bar/kappa2
            def mock_model_complex3(x, **kwargs):
                cpoints = point_vec_to_complex(x)
                kappa3 = tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,4:6])**2, axis=-1),complex_dtype) 
                z0,z1 = cpoints[:,4], cpoints[:,5]
                z0bar,z1bar = tf.math.conj(z0), tf.math.conj(z1)        

                return z0*z1bar/kappa3
            def mock_model_complex4(x, **kwargs):
                cpoints = point_vec_to_complex(x)
                kappa4 = tf.cast(tf.reduce_sum(tf.math.abs(cpoints[:,6:8])**2, axis=-1),complex_dtype) 
                w0,w1 = cpoints[:,6], cpoints[:,7]
                w0bar,w1bar = tf.math.conj(w0), tf.math.conj(w1)

                return w0*w1bar/kappa4

            tf.keras.utils.set_random_seed(seed+1)
            #mock_model_2 = BiholoModelFuncGENERALforHYMinv3(network_shape, BASIS, stddev=0.1, use_zero_network=False, use_symmetry_reduced_TQ=False)
            print('model1: example output',mock_model(real_pts[0:3]))
            print('model1: scale of first 100 outputs',tf.math.reduce_mean(tf.abs(mock_model(real_pts[0:100]))))
            print('model2: example output',mock_model_2(real_pts[0:3]))
            print('model2: scale of first 100 outputs',tf.math.reduce_mean(tf.abs(mock_model_2(real_pts[0:100]))))
            print('model3: example output',mock_model_3(real_pts[0:3]))
            print('model3: scale of first 100 outputs',tf.math.reduce_mean(tf.abs(mock_model_3(real_pts[0:100]))))
            print('model4: example output',mock_model_4(real_pts[0:3]))
            print('model4: scale of first 100 outputs',tf.math.reduce_mean(tf.abs(mock_model_4(real_pts[0:100]))))
            if loadextra:
                try:
                    dataextra = np.load(laplacians_filename, allow_pickle=True)
                    dataextra = convert_to_nested_tensor_dict(dataextra)
                    laplacian_mock = dataextra['laplacian_mock']
                    laplacian_mock_2 = dataextra['laplacian_mock_2']
                    mock_model_vals = dataextra['mock_model_vals']
                    mock_model_2_vals = dataextra['mock_model_2_vals']
                    if not tf.reduce_all(tf.abs(mock_model_vals[0:100] - mock_model(real_pts[0:100])) < 1e-5):
                        print(f"Loaded incorrectly! Not compatible, {tf.reduce_max(tf.abs(mock_model_vals[0:100] - mock_model(real_pts[0:100])))}")
                        print(f"types: {type(mock_model_vals)}, {type(mock_model(real_pts))}")
                        print(f"loaded: { mock_model_vals.dtype}, {mock_model_vals.shape}, {mock_model_vals[0:1]}")
                        print(f"actual: {mock_model(real_pts).dtype}, {mock_model(real_pts).shape}, {mock_model(real_pts[:1])}")
                        raise ValueError("Loaded incorrectly! Not compatible.")
                    print("Loaded laplacians from saved file")
                except FileNotFoundError:
                    raise ValueError("Laplacians file not found. Computing laplacians...")
                    
            else:
                print("Computing laplacians as not requested to load.")
                laplacian_mock = batch_process_helper_func(laplacian, [mock_model, real_pts, pullbacks, invmetrics], batch_indices=[1, 2, 3], kwargs={'training': False})
                laplacian_mock_2 = batch_process_helper_func(laplacian, [mock_model_2, real_pts, pullbacks, invmetrics], batch_indices=[1, 2, 3], kwargs={'training': False})
                laplacian_mock_3 = batch_process_helper_func(laplacian, [mock_model_3, real_pts, pullbacks, invmetrics], batch_indices=[1, 2, 3], kwargs={'training': False})
                laplacian_mock_4 = batch_process_helper_func(laplacian, [mock_model_4, real_pts, pullbacks, invmetrics], batch_indices=[1, 2, 3], kwargs={'training': False})
                laplacian_mock_c = batch_process_helper_func(laplacianWithH, [mock_model_complex1, real_pts, pullbacks, invmetrics,None], batch_indices=[1, 2, 3], kwargs={'training': False})
                laplacian_mock_2_c = batch_process_helper_func(laplacianWithH, [mock_model_complex2, real_pts, pullbacks, invmetrics,None], batch_indices=[1, 2, 3], kwargs={'training': False})
                laplacian_mock_3_c = batch_process_helper_func(laplacianWithH, [mock_model_complex3, real_pts, pullbacks, invmetrics,None], batch_indices=[1, 2, 3], kwargs={'training': False})
                laplacian_mock_4_c = batch_process_helper_func(laplacianWithH, [mock_model_complex4, real_pts, pullbacks, invmetrics,None], batch_indices=[1, 2, 3], kwargs={'training': False})
                mock_model_vals = mock_model(real_pts)
                mock_model_2_vals = mock_model_2(real_pts)
                mock_model_3_vals = mock_model_3(real_pts)
                mock_model_4_vals = mock_model_4(real_pts)
                mock_model_c_vals = mock_model_complex1(real_pts)
                mock_model_2_c_vals = mock_model_complex2(real_pts)
                mock_model_3_c_vals = mock_model_complex3(real_pts)
                mock_model_4_c_vals = mock_model_complex4(real_pts)
                
                # Save laplacians to file
                if saveextra:
                    np.savez(laplacians_filename, 
                         laplacian_mock=laplacian_mock.numpy(),
                         laplacian_mock_2=laplacian_mock_2.numpy(),
                         laplacian_mock_3=laplacian_mock_3.numpy(),
                         laplacian_mock_4=laplacian_mock_4.numpy(),
                         laplacian_mock_c=laplacian_mock_c.numpy(),
                         laplacian_mock_2_c=laplacian_mock_2_c.numpy(),
                         laplacian_mock_3_c=laplacian_mock_3_c.numpy(),
                         laplacian_mock_4_c=laplacian_mock_4_c.numpy(),

                         mock_model_vals=mock_model_vals.numpy(),
                         mock_model_2_vals=mock_model_2_vals.numpy(),
                         mock_model_3_vals=mock_model_3_vals.numpy(),
                         mock_model_4_vals=mock_model_4_vals.numpy(),
                         mock_model_c_vals=mock_model_c_vals.numpy(),
                         mock_model_2_c_vals=mock_model_2_c_vals.numpy(),
                         mock_model_3_c_vals=mock_model_3_c_vals.numpy(),
                         mock_model_4_c_vals=mock_model_4_c_vals.numpy())
                print(f"Saved laplacians to {laplacians_filename}")
                
                # laplacian_mock = batch_process_helper_func(laplacian, [mock_model, real_pts, pullbacks, invmetrics], batch_indices=[1, 2, 3], kwargs={'training': False})
                # laplacian_mock_2 = batch_process_helper_func(laplacian, [mock_model_2, real_pts, pullbacks, invmetrics], batch_indices=[1, 2, 3], kwargs={'training': False})
                # mock_model_vals = mock_model(real_pts)
                # mock_model_2_vals = mock_model_2(real_pts)
                # np

            #sectionval_mock = mock_model(real_pts)
            #sectionval_mock_2 = mock_model_2(real_pts)

            # Calculate absolute values for mock models
            laplacian_mock_real_abs = tf.abs(tf.math.real(laplacian_mock))
            laplacian_mock_imag_abs = tf.abs(tf.math.imag(laplacian_mock))
            
            laplacian_mock_2_real_abs = tf.abs(tf.math.real(laplacian_mock_2))
            laplacian_mock_2_imag_abs = tf.abs(tf.math.imag(laplacian_mock_2))
            
            laplacian_mock_3_real_abs = tf.abs(tf.math.real(laplacian_mock_3))
            laplacian_mock_3_imag_abs = tf.abs(tf.math.imag(laplacian_mock_3))
            
            laplacian_mock_4_real_abs = tf.abs(tf.math.real(laplacian_mock_4))
            laplacian_mock_4_imag_abs = tf.abs(tf.math.imag(laplacian_mock_4))

            laplacian_mock_c_real_abs = tf.abs(tf.math.real(laplacian_mock_c))
            laplacian_mock_c_imag_abs = tf.abs(tf.math.imag(laplacian_mock_c))
            
            laplacian_mock_2_c_real_abs = tf.abs(tf.math.real(laplacian_mock_2_c))
            laplacian_mock_2_c_imag_abs = tf.abs(tf.math.imag(laplacian_mock_2_c))

            laplacian_mock_3_c_real_abs = tf.abs(tf.math.real(laplacian_mock_3_c))
            laplacian_mock_3_c_imag_abs = tf.abs(tf.math.imag(laplacian_mock_3_c))

            laplacian_mock_4_c_real_abs = tf.abs(tf.math.real(laplacian_mock_4_c))
            laplacian_mock_4_c_imag_abs = tf.abs(tf.math.imag(laplacian_mock_4_c))

            # Use g_cy_weights for integration
            g_cy_weights = aux_weights * tf.math.real(tf.linalg.det(mets))

            vol, se, _, _ = weighted_mean_and_standard_error(g_cy_weights**0, g_cy_weights)
            print(f"check volume: {vol} +- {se}")

            print(f"Shapes: {mock_model_vals.shape}, {mock_model_2_vals.shape}, {laplacian_mock.shape}, {laplacian_mock_2.shape}")

            # Ensure mock models are real
            mock_model_vals_real = tf.math.real(mock_model_vals)
            mock_model_2_vals_real = tf.math.real(mock_model_2_vals)
            mock_model_3_vals_real = tf.math.real(mock_model_3_vals)
            mock_model_4_vals_real = tf.math.real(mock_model_4_vals)

            # Integrate laplacians
            integate_mock, integate_mock_se, integate_mock_eff_n, _ = weighted_mean_and_standard_error(mock_model_vals_real, g_cy_weights)
            integate_mock_2, integate_mock_2_se, integate_mock_2_eff_n, _ = weighted_mean_and_standard_error(mock_model_2_vals_real, g_cy_weights)
            integate_mock_3, integate_mock_3_se, integate_mock_3_eff_n, _ = weighted_mean_and_standard_error(mock_model_3_vals_real, g_cy_weights)
            integate_mock_4, integate_mock_4_se, integate_mock_4_eff_n, _ = weighted_mean_and_standard_error(mock_model_4_vals_real, g_cy_weights)
            integate_mock_c, integate_mock_c_se, integate_mock_c_eff_n, _ = weighted_mean_and_standard_error(mock_model_c_vals, g_cy_weights)  
            integate_mock_2_c, integate_mock_2_c_se, integate_mock_2_c_eff_n, _ = weighted_mean_and_standard_error(mock_model_2_c_vals, g_cy_weights)
            integate_mock_3_c, integate_mock_3_c_se, integate_mock_3_c_eff_n, _ = weighted_mean_and_standard_error(mock_model_3_c_vals, g_cy_weights)
            integate_mock_4_c, integate_mock_4_c_se, integate_mock_4_c_eff_n, _ = weighted_mean_and_standard_error(mock_model_4_c_vals, g_cy_weights)

            
            
            int_lap_mock, int_lap_mock_se, int_lap_mock_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock, g_cy_weights)
            int_lap_mock_real, int_lap_mock_real_se, int_lap_mock_real_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_real_abs, g_cy_weights)
            int_lap_mock_imag, int_lap_mock_imag_se, int_lap_mock_imag_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_imag_abs, g_cy_weights)
            
            int_lap_mock_2, int_lap_mock_2_se, int_lap_mock_2_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_2, g_cy_weights)
            int_lap_mock_2_real, int_lap_mock_2_real_se, int_lap_mock_2_real_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_2_real_abs, g_cy_weights)
            int_lap_mock_2_imag, int_lap_mock_2_imag_se, int_lap_mock_2_imag_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_2_imag_abs, g_cy_weights)
            
            int_lap_mock_3, int_lap_mock_3_se, int_lap_mock_3_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_3, g_cy_weights)
            int_lap_mock_3_real, int_lap_mock_3_real_se, int_lap_mock_3_real_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_3_real_abs, g_cy_weights)
            int_lap_mock_3_imag, int_lap_mock_3_imag_se, int_lap_mock_3_imag_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_3_imag_abs, g_cy_weights)
            
            int_lap_mock_4, int_lap_mock_4_se, int_lap_mock_4_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_4, g_cy_weights)
            int_lap_mock_4_real, int_lap_mock_4_real_se, int_lap_mock_4_real_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_4_real_abs, g_cy_weights)
            int_lap_mock_4_imag, int_lap_mock_4_imag_se, int_lap_mock_4_imag_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_4_imag_abs, g_cy_weights)


            int_lap_mock_c, int_lap_mock_c_se, int_lap_mock_c_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_c, g_cy_weights)
            int_lap_mock_c_real, int_lap_mock_c_real_se, int_lap_mock_c_real_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_c_real_abs, g_cy_weights)
            int_lap_mock_c_imag, int_lap_mock_c_imag_se, int_lap_mock_c_imag_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_c_imag_abs, g_cy_weights)   

            int_lap_mock_2_c, int_lap_mock_2_c_se, int_lap_mock_2_c_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_2_c, g_cy_weights)
            int_lap_mock_2_c_real, int_lap_mock_2_c_real_se, int_lap_mock_2_c_real_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_2_c_real_abs, g_cy_weights)
            int_lap_mock_2_c_imag, int_lap_mock_2_c_imag_se, int_lap_mock_2_c_imag_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_2_c_imag_abs, g_cy_weights)

            int_lap_mock_3_c, int_lap_mock_3_c_se, int_lap_mock_3_c_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_3_c, g_cy_weights)
            int_lap_mock_3_c_real, int_lap_mock_3_c_real_se, int_lap_mock_3_c_real_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_3_c_real_abs, g_cy_weights)
            int_lap_mock_3_c_imag, int_lap_mock_3_c_imag_se, int_lap_mock_3_c_imag_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_3_c_imag_abs, g_cy_weights)

            int_lap_mock_4_c, int_lap_mock_4_c_se, int_lap_mock_4_c_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_4_c, g_cy_weights)
            int_lap_mock_4_c_real, int_lap_mock_4_c_real_se, int_lap_mock_4_c_real_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_4_c_real_abs, g_cy_weights)
            int_lap_mock_4_c_imag, int_lap_mock_4_c_imag_se, int_lap_mock_4_c_imag_eff_n, _ = weighted_mean_and_standard_error(laplacian_mock_4_c_imag_abs, g_cy_weights)


            
            
            
            
            
            
            
            
            
            # Print integration results
            print(f"Mock model 1: {integate_mock} ± {integate_mock_se}")
            print(f"Mock model 1 laplacian: {int_lap_mock} ± {int_lap_mock_se}")
            print(f"Mock model 1 |Re(lap)|: {int_lap_mock_real} ± {int_lap_mock_real_se}")
            #print(f"Mock model 1 |Im(lap)|: {int_lap_mock_imag} ± {int_lap_mock_imag_se}")
            
            print(f"Mock model 2: {integate_mock_2} ± {integate_mock_2_se}")
            print(f"Mock model 2 laplacian: {int_lap_mock_2} ± {int_lap_mock_2_se}")
            print(f"Mock model 2 |Re(lap)|: {int_lap_mock_2_real} ± {int_lap_mock_2_real_se}")
            #print(f"Mock model 2 |Im(lap)|: {int_lap_mock_2_imag} ± {int_lap_mock_2_imag_se}")
            
            print(f"Mock model 3: {integate_mock_3} ± {integate_mock_3_se}")
            print(f"Mock model 3 laplacian: {int_lap_mock_3} ± {int_lap_mock_3_se}")
            print(f"Mock model 3 |Re(lap)|: {int_lap_mock_3_real} ± {int_lap_mock_3_real_se}")
            #print(f"Mock model 3 |Im(lap)|: {int_lap_mock_3_imag} ± {int_lap_mock_3_imag_se}")
            
            print(f"Mock model 4: {integate_mock_4} ± {integate_mock_4_se}")
            print(f"Mock model 4 laplacian: {int_lap_mock_4} ± {int_lap_mock_4_se}")
            print(f"Mock model 4 |Re(lap)|: {int_lap_mock_4_real} ± {int_lap_mock_4_real_se}")
            #print(f"Mock model 4 |Im(lap)|: {int_lap_mock_4_imag} ± {int_lap_mock_4_imag_se}")

            print(f"Mock model c: {integate_mock_c} ± {integate_mock_c_se}")
            print(f"Mock model c laplacian: {int_lap_mock_c} ± {int_lap_mock_c_se}")
            print(f"Mock model c |Re(lap)|: {int_lap_mock_c_real} ± {int_lap_mock_c_real_se}")
            #print(f"Mock model c |Im(lap)|: {int_lap_mock_c_imag} ± {int_lap_mock_c_imag_se}")

            print(f"Mock model 2 c: {integate_mock_2_c} ± {integate_mock_2_c_se}")
            print(f"Mock model 2 c laplacian: {int_lap_mock_2_c} ± {int_lap_mock_2_c_se}")
            print(f"Mock model 2 c |Re(lap)|: {int_lap_mock_2_c_real} ± {int_lap_mock_2_c_real_se}")
            #print(f"Mock model 2 c |Im(lap)|: {int_lap_mock_2_c_imag} ± {int_lap_mock_2_c_imag_se}")
            
            print(f"Mock model 3 c: {integate_mock_3_c} ± {integate_mock_3_c_se}")
            print(f"Mock model 3 c laplacian: {int_lap_mock_3_c} ± {int_lap_mock_3_c_se}")
            print(f"Mock model 3 c |Re(lap)|: {int_lap_mock_3_c_real} ± {int_lap_mock_3_c_real_se}")
            #print(f"Mock model 3 c |Im(lap)|: {int_lap_mock_3_c_imag} ± {int_lap_mock_3_c_imag_se}")
            
            print(f"Mock model 4 c: {integate_mock_4_c} ± {integate_mock_4_c_se}")
            print(f"Mock model 4 c laplacian: {int_lap_mock_4_c} ± {int_lap_mock_4_c_se}")
            print(f"Mock model 4 c |Re(lap)|: {int_lap_mock_4_c_real} ± {int_lap_mock_4_c_real_se}")
            #print(f"Mock model 4 c |Im(lap)|: {int_lap_mock_4_c_imag} ± {int_lap_mock_4_c_imag_se}")
            
            
            
            
            
            


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

        # print("Hneffs, Qneffs, Uneffs: ")
        # print(np.round(Hneffs,1))
        # print(np.round(Qneffs,1))
        # print(np.round(Uneffs,1))
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
                m_value, m_se, m_eff_n, m_stats = weighted_mean_and_standard_error(scaled_integrand, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
                m_valueewoH, m_sewoH, m_eff_nwoH, m_statswoH = weighted_mean_and_standard_error(scaled_integrandwoH, aux_weights,is_top_form=True, mulweightsby=mulweightsby)
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
        holomorphic_Yukawas_trained_and_ref_errors.append(m_errors)
        print("without H * 10**6")
        print(np.round(np.array(mwoH)*10**6,1))
        print("holomorphic Yukawa errors *10**6 (absolute value)")
        print(np.round(m_errorswoH*10**6,1))

        
        # print("neffs without H")
        # print(np.round(mwoH_neffs,1))

        print('proper calculation*10**6')
        print(np.round(np.array(m)*10**6,1))
  
         # Print holomorphic Yukawa matrix errors
        print("holomorphic Yukawa errors *10**6 (absolute value)")
        print(np.round(m_errors*10**6,1))
        # print("neffs")
        # print(np.round(m_neffs,1))


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
        print("\n")

        # Log masses and their errors to wandb
        if wandb.run is not None:
            # Create a dictionary to log masses and their errors
            mass_data = {}
            # Add trained or untrained prefix to the logs
            prefix = "trained_" if use_trained else "ref_"

            for i in range(len(s)):
                mass_data[f"{prefix}mass_{i+1}"] = s[i]
                mass_data[f"{prefix}mass_{i+1}_error"] = singular_value_errors[i]

            # Log the data to wandb
            wandb.log(mass_data)
            # Create tables for physical yukawa matrices (trained and reference)
            physical_yukawa_data = []
            # Log individual physical yukawa matrix elements instead of as a table
            log_data = {}
            for i in range(3):
                for j in range(3):
                    # Add trained/ref values with specific keys for each component
                    log_data[f"{prefix[:-1]}_physical_yukawa_matrix_{i}{j}_real"] = np.real(physical_yukawas[i,j])
                    log_data[f"{prefix[:-1]}_physical_yukawa_matrix_{i}{j}_imag"] = np.imag(physical_yukawas[i,j])
                    log_data[f"{prefix[:-1]}_physical_yukawa_matrix_{i}{j}_error"] = np.abs(physical_yukawas_errors[i,j])
            
            # Log all the data at once
            wandb.log(log_data)
            
            # Create table for holomorphic yukawa matrices (trained and reference)
            holomorphic_yukawa_data = []
            
            # Log individual holomorphic yukawa matrix elements
            holo_log_data = {}
            for i in range(3):
                for j in range(3):
                    # Add trained/ref values with specific keys for each component
                    holo_log_data[f"{prefix[:-1]}_holomorphic_yukawa_matrix_{i}{j}_real"] = np.real(holomorphic_Yukawas_trained_and_ref[-1][i,j])
                    holo_log_data[f"{prefix[:-1]}_holomorphic_yukawa_matrix_{i}{j}_imag"] = np.imag(holomorphic_Yukawas_trained_and_ref[-1][i,j])
                    holo_log_data[f"{prefix[:-1]}_holomorphic_yukawa_matrix_{i}{j}_error"] = np.abs(holomorphic_Yukawas_trained_and_ref_errors[-1][i,j])
                    
                    # Also add to table format for visualization
                    holomorphic_yukawa_data.append([
                        i, j, prefix[:-1], 
                        np.real(holomorphic_Yukawas_trained_and_ref[-1][i,j]),
                        np.imag(holomorphic_Yukawas_trained_and_ref[-1][i,j]), 
                        np.abs(holomorphic_Yukawas_trained_and_ref_errors[-1][i,j])
                    ])
            
            # Log holomorphic yukawa data
            wandb.log(holo_log_data)
            
            # Log holomorphic yukawa matrix as table
            wandb.log({
                "holomorphic_yukawa_matrix": wandb.Table(
                    data=holomorphic_yukawa_data,
                    columns=["row", "col", "type", "real_value", "imag_value", "abs_error"]
                )
            })

        ## Calculate effective sample size for the full dataset
        #full_eff_n = effective_sample_size(aux_weights[0:n_p]).numpy()
        #print(f"Overall effective sample size: {full_eff_n} (out of {n_p} total points)")

    # Save training results to CSV files


    trained_holo13 = holomorphic_Yukawas_trained_and_ref[0][0,2]
    trained_holo23 = holomorphic_Yukawas_trained_and_ref[0][1,2]
    trained_holo31 = holomorphic_Yukawas_trained_and_ref[0][2,0]
    trained_holo32 = holomorphic_Yukawas_trained_and_ref[0][2,1]
    ref_holo13 = holomorphic_Yukawas_trained_and_ref[1][0,2]
    ref_holo23 = holomorphic_Yukawas_trained_and_ref[1][1,2]
    ref_holo31 = holomorphic_Yukawas_trained_and_ref[1][2,0]
    ref_holo32 = holomorphic_Yukawas_trained_and_ref[1][2,1]
    trained_holo13_score= np.abs(trained_holo13)/np.abs(holomorphic_Yukawas_trained_and_ref_errors[0][0,2])/np.abs(trained_holo13)
    trained_holo23_score=np.abs(trained_holo23)/np.abs(holomorphic_Yukawas_trained_and_ref_errors[0][1,2])/np.abs(trained_holo23)    
    trained_holo31_score = np.abs(trained_holo31)/np.abs(holomorphic_Yukawas_trained_and_ref_errors[0][2,0])/np.abs(trained_holo31)
    trained_holo32_score = np.abs(trained_holo32)/np.abs(holomorphic_Yukawas_trained_and_ref_errors[0][2,1])/np.abs(trained_holo32)
    ref_holo13_score = np.abs(ref_holo13)/np.abs(holomorphic_Yukawas_trained_and_ref_errors[1][0,2])/np.abs(ref_holo13)
    ref_holo23_score = np.abs(ref_holo23)/np.abs(holomorphic_Yukawas_trained_and_ref_errors[1][1,2])/np.abs(ref_holo23)   
    ref_holo31_score = np.abs(ref_holo31)/np.abs(holomorphic_Yukawas_trained_and_ref_errors[1][2,0])/np.abs(ref_holo31)
    ref_holo32_score = np.abs(ref_holo32)/np.abs(holomorphic_Yukawas_trained_and_ref_errors[1][2,1])/np.abs(ref_holo32)
    
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
        'masses_values': np.array(masses_trained_and_ref),
        'holomorphic_yukawas': np.array(holomorphic_Yukawas_trained_and_ref),
        'holomorphic_yukawas_errors': m_errors,

        # Statistical error information
        'integral_stats': integral_stats,
        'matrix_stats': matrix_stats,
        'physical_yukawas_errors': physical_yukawas_errors,
        'masses_errors': singular_value_errors,
        'm_neffs': m_neffs,
        'mwoH_neffs': mwoH_neffs,
        'Qneffs': Qneffs,
        'Uneffs': Uneffs,
        'Hneffs': Hneffs,
        'topological_data': topological_data
    }

    # Create unique filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{unique_id_or_coeff}"  

    # Save to unique file in results directory
    os.makedirs(os.path.join(result_files_path,f'{manifold_name}_{type_folder}_results_{addtofilename}'), exist_ok=True)
    npzsavelocation = os.path.join(result_files_path,f'{manifold_name}_{type_folder}_results_{addtofilename}/run_' + run_id + '.npz')
    np.savez(npzsavelocation, **results)
    print("saving npz of run to " + npzsavelocation)

    # Save masses to CSV
    import csv
    os.makedirs(os.path.join(result_files_path,'masses'), exist_ok=True)
    csv_file = os.path.join(result_files_path,'masses',f'{manifold_name}_{type_folder}_masses_{addtofilename}.csv')
    print("saving csv to " + npzsavelocation, "saving npz to " + npzsavelocation)

    doubleprecision = network_params['doubleprecision']
    orbit = network_params['orbit']
    
    
    # Create header if file doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run_id', 'learned_mass1', 'learned_mass2', 'learned_mass3', 
                             'learned_mass1_error', 'learned_mass2_error', 'learned_mass3_error',
                             'ref_mass1', 'ref_mass2', 'ref_mass3',
                             'ref_mass1_error', 'ref_mass2_error', 'ref_mass3_error',
                             'coefficient', 'n_to_integrate', 'run_args','doubleprecision','orbit','t_holo13','t_holo23','t_holo31','t_holo21','r_holo13','r_holo23','r_holo31','r_holo21',
                             't_holo13_score','t_holo23_score','t_holo31_score','t_holo21_score','r_holo13_score','r_holo23_score','r_holo31_score','r_holo21_score'])

    # Append masses for this run
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run_id] + 
                        list(masses_trained_and_ref[0]) + 
                        list(singular_value_errors) +
                        list(masses_trained_and_ref[1]) + 
                        list(singular_value_errors) +  # Using same error estimate for both learned and reference
                        [unique_id_or_coeff] + 
                        [n_p]+ [run_args]+ [doubleprecision]+ [orbit]+[trained_holo13,trained_holo23,trained_holo31,trained_holo32] + [ref_holo13,ref_holo23,ref_holo31,ref_holo32] +
                        [trained_holo13_score,trained_holo23_score,trained_holo31_score,trained_holo32_score] + [ref_holo13_score,ref_holo23_score,ref_holo31_score,ref_holo32_score])

    if do_extra_stuff:    
        pass

    return np.array(masses_trained_and_ref), np.array(singular_value_errors)