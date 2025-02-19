from cymetric.config import float_dtype, complex_dtype
import numpy as np
import gc
import sys
import os
import re
import logging
import pickle
import sys
#sys.path.append("/Users/kit/Documents/Phys_Working/MF metric")

logging.basicConfig(stream=sys.stdout)

from cymetric.pointgen.pointgen import PointGenerator
from cymetric.pointgen.nphelper import prepare_dataset, prepare_basis_pickle

import tensorflow as tf
import tensorflow.keras as tfk

tf.get_logger().setLevel('ERROR')


from cymetric.models.tfmodels import PhiFSModel, MultFSModel, FreeModel, MatrixFSModel, AddFSModel, PhiFSModelToric, MatrixFSModelToric
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.callbacks import SigmaCallback, KaehlerCallback, TransitionCallback, RicciCallback, VolkCallback, AlphaCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, RicciLoss, VolkLoss, TotalLoss

from NewCustomMetrics import *
from HarmonicFormModel import *
from BetaModel import *
from laplacian_funcs import *
from OneAndTwoFormsForLineBundles import *
from generate_and_train_all_nnsHOLO import *


# nPoints=300000
# free_coefficient=1.# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
# nEpochsPhi=100
# nEpochsBeta=60
# nEpochsSigma=50


nPoints=10000
#free_coefficient=1.000000004# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
import sys
#free_coefficient = float(sys.argv[1])
#free_coefficient=1.# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
nEpochsPhi=100
nEpochsBeta=100
nEpochsSigma=75
depthPhi=3
widthPhi=2#128 4 in the 1.0s
depthBeta=3
widthBeta=2
depthSigma=3
widthSigma=2# up from 256
alphabeta=[1,10]
alphasigma1=[1,10] # down form 50
alphasigma2=[1,80] # down from 200
#
#nEpochsPhi=150
#nEpochsBeta=150
#nEpochsSigma=100
#depthPhi=4
#widthPhi=128
#depthBeta=4
#widthBeta=128#96
#depthSigma=3
#widthSigma=128
#
#nEpochsPhi=150
#nEpochsBeta=100
#nEpochsSigma=50
#depthPhi=4
#widthPhi=128
#depthBeta=3
#widthBeta=128#96
#depthSigma=3
#widthSigma=128
#
# %%capture outputfor00000001

#free_coefficient = np.random.randint(-20,21,size=21)
#free_coefficient_name=hash(tuple(free_coefficient))
#print('set of coeffs')
#print(free_coefficient)
#print("hash")
#print(hash(tuple(free_coefficient)))
print('ok')
free_coefficient = float(sys.argv[1])
print('free coefficient')
print(free_coefficient)


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


if __name__ ==  '__main__':
    generate_points_and_save_using_defaults(free_coefficient,nPoints)
    phimodel1,training_history,_=load_nn_phimodel(free_coefficient,depthPhi,widthPhi,nEpochsPhi,[64,50000],set_weights_to_zero=True,skip_measures=True)

    linebundleforHYM_02m20=np.array([0,2,-2,0]) 
    linebundleforHYM_110m2=np.array([1,1,0,-2]) 
    linebundleforHYM_m1m322=np.array([-1,-3,2,2]) 
    gc.collect()
    tf.keras.backend.clear_session()


    generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM_02m20,nPoints,phimodel1,force_generate=False)
    betamodel_02m20,training_historyBeta_02m20,_=load_nn_HYM(free_coefficient,linebundleforHYM_02m20,depthBeta,widthBeta,nEpochsBeta,[64,50000],set_weights_to_zero=True,skip_measures=True)
    gc.collect()
    tf.keras.backend.clear_session()


    
    
    
    delete_all_dicts_except('dataEval')
    n_integrate=1000000
    pg,kmoduli=generate_points_and_save_using_defaults_for_eval(free_coefficient,n_integrate)
    dataEval=np.load(os.path.join('data/tetraquadric_pg_for_eval_with_'+str(free_coefficient), 'dataset.npz'))
    n_p = n_integrate
    phi = phimodel1
    points64=dataEval['X_train'][0:n_p]
    real_pts=tf.cast(dataEval['X_train'][0:n_p],np.float32)
    pointsComplex=tf.cast(point_vec_to_complex(points64),complex_dtype)# we have to cast to complex64 from complex128 sometimes. Just for safety. point_vec_to_complex does the obvious thing
    #mets=phimodel.fubini_study_pb(real_pts,ts=tf.cast(kmoduli,complex_dtype))
    
    H1=betamodel_02m20(real_pts)
    #mH1=mbetamodel_02m20(real_pts)
    #
    #prodbundles=tf.reduce_mean(H1*mH1)
    #prodbundlesstddev=tf.math.reduce_std(H1*mH1)
    #print("test if good bundle metric inverse" + str(prodbundles) + " and std ddev " + str(prodbundlesstddev))

    generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM_110m2,nPoints,phimodel1,force_generate=False)
    betamodel_110m2,training_historyBeta_110m2,_=load_nn_HYM(free_coefficient,linebundleforHYM_110m2,depthBeta,widthBeta,nEpochsBeta,[64,50000],set_weights_to_zero=True,skip_measures=True)
    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM_m1m322,nPoints,phimodel1,force_generate=False)
    betamodel_m1m322,training_historyBeta_m1m322,_=load_nn_HYM(free_coefficient,linebundleforHYM_m1m322,depthBeta,widthBeta,nEpochsBeta,[64,50000],set_weights_to_zero=True,skip_measures=True)

    gc.collect()
    tf.keras.backend.clear_session()
    H2=betamodel_110m2(real_pts)
    H3=betamodel_m1m322(real_pts)


    prodbundles=tf.reduce_mean(H1*H2*H3)
    prodbundlesstddev=tf.math.reduce_std(H1*H2*H3)
    print("test if good bundle metric" + str(prodbundles) + " and std ddev " + str(prodbundlesstddev))


    phi=phimodel1
    def vFS_Q1(x):
        return getTypeIIs(x,phi,'vQ1')
    def vFS_Q2(x):
        return getTypeIIs(x,phi,'vQ2')
    def vFS_U1(x):
        return getTypeIIs(x,phi,'vU1')
    def vFS_U2(x):
        return getTypeIIs(x,phi,'vU2')

    gc.collect()
    tf.keras.backend.clear_session()


    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_02m20,functionforbaseharmonicform_jbar_for_vH,phi,betamodel_02m20,nPoints,force_generate=False)
    HFmodel_vH,trainingHistoryHF_vH,_=load_nn_HF(free_coefficient,linebundleforHYM_02m20,betamodel_02m20,functionforbaseharmonicform_jbar_for_vH,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma1,set_weights_to_zero=True,skip_measures=True)
    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_110m2,functionforbaseharmonicform_jbar_for_vQ3,phi,betamodel_110m2,nPoints,force_generate=False)
    HFmodel_vQ3,trainingHistoryHF_vQ3,_=load_nn_HF(free_coefficient,linebundleforHYM_110m2,betamodel_110m2,functionforbaseharmonicform_jbar_for_vQ3,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma1,set_weights_to_zero=True,skip_measures=True)
    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_110m2,functionforbaseharmonicform_jbar_for_vU3,phi,betamodel_110m2,nPoints,force_generate=False)
    HFmodel_vU3,trainingHistoryHF_vU3,_=load_nn_HF(free_coefficient,linebundleforHYM_110m2,betamodel_110m2,functionforbaseharmonicform_jbar_for_vU3,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma1,set_weights_to_zero=True,skip_measures=True)

    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_m1m322,vFS_Q1,phi,betamodel_m1m322,nPoints,force_generate=False)
    HFmodel_vQ1,trainingHistoryHF_vQ1,_=load_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_Q1,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma2,set_weights_to_zero=True,skip_measures=True)
    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_m1m322,vFS_Q2,phi,betamodel_m1m322,nPoints,force_generate=False)
    HFmodel_vQ2,trainingHistoryHF_vQ2,_=load_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_Q2,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma2,set_weights_to_zero=True,skip_measures=True)
    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_m1m322,vFS_U1,phi,betamodel_m1m322,nPoints,force_generate=False)
    HFmodel_vU1,trainingHistoryHF_vU1,_=load_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_U1,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma2,set_weights_to_zero=True,skip_measures=True)
    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_m1m322,vFS_U2,phi,betamodel_m1m322,nPoints,force_generate=False)
    HFmodel_vU2,trainingHistoryHF_vU2,_=load_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_U2,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma2,set_weights_to_zero=True,skip_measures=True)
    gc.collect()
    tf.keras.backend.clear_session()

    print("\n do analysis: \n\n\n")

    Vol_from_dijk_J=pg.get_volume_from_intersections(kmoduli)
    Vol_reference_dijk_with_k_is_1=pg.get_volume_from_intersections(np.ones_like(kmoduli)) 
    volCY_from_Om=tf.reduce_mean(dataEval['y_train'][:n_p,0]/6

    aux_weights=(dataEval['y_train'][:n_p,0]/(dataEval['y_train'][:n_p,1]))*(1/(6))### these are the appropriate 'flat measure' so sum_i aux_weights f is int_X f(x)
    #convertcomptoreal=lambda complexvec: tf.concat([tf.math.real(complexvec),tf.math.imag(complexvec)],-1) # this converts from complex to real

    mats=[]
    masses_ref=[]
    masses_holo=[]
    use_trained = False
    for use_trained in [False]:
        if use_trained:
            print("NO")
            #mets=phi(real_pts)
            #dets=tf.linalg.det(mets)

            #vH =  HFmodel_vH.corrected_harmonicform(real_pts)
            #hvHb =  tf.einsum('x,xb->xb',tf.cast(betamodel_02m20(real_pts),complex_dtype),tf.math.conj(vH))
            #vQ3 = HFmodel_vQ3.corrected_harmonicform(real_pts)
            #hvQ3b = tf.einsum('x,xb->xb',tf.cast(betamodel_110m2(real_pts),complex_dtype),tf.math.conj(vQ3))
            #vU3 = HFmodel_vU3.corrected_harmonicform(real_pts)
            #hvU3b = tf.einsum('x,xb->xb',tf.cast(betamodel_110m2(real_pts),complex_dtype),tf.math.conj(vU3))

            #vQ1 = HFmodel_vQ1.corrected_harmonicform(real_pts)
            #hvQ1b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322(real_pts),complex_dtype),tf.math.conj(vQ1))
            #vQ2 = HFmodel_vQ2.corrected_harmonicform(real_pts)
            #hvQ2b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322(real_pts),complex_dtype),tf.math.conj(vQ2))

            #vU1 = HFmodel_vU1.corrected_harmonicform(real_pts)
            #hvU1b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322(real_pts),complex_dtype),tf.math.conj(vU1))
            #vU2 = HFmodel_vU2.corrected_harmonicform(real_pts)
            #hvU2b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322(real_pts),complex_dtype),tf.math.conj(vU2))

            #H1=betamodel_02m20(real_pts) 
            #H2=betamodel_110m2(real_pts) 
            #H3=betamodel_m1m322(real_pts) 
        elif not use_trained:
            mets = phi.fubini_study_pb(real_pts,ts=tf.cast(kmoduli,complex_dtype))
            dets = tf.linalg.det(mets)
            vH =  HFmodel_vH.uncorrected_FS_harmonicform(real_pts)
            hvHb =  tf.einsum('x,xb->xb',tf.cast(betamodel_02m20.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vH))
            vQ3 = HFmodel_vQ3.uncorrected_FS_harmonicform(real_pts)
            hvQ3b = tf.einsum('x,xb->xb',tf.cast(betamodel_110m2.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vQ3))
            vU3 = HFmodel_vU3.uncorrected_FS_harmonicform(real_pts)
            hvU3b = tf.einsum('x,xb->xb',tf.cast(betamodel_110m2.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vU3))

            vQ1 = HFmodel_vQ1.uncorrected_FS_harmonicform(real_pts)
            hvQ1b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vQ1))
            vQ2 = HFmodel_vQ2.uncorrected_FS_harmonicform(real_pts)
            hvQ2b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vQ2))

            vU1 = HFmodel_vU1.uncorrected_FS_harmonicform(real_pts)
            hvU1b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vU1))
            vU2 = HFmodel_vU2.uncorrected_FS_harmonicform(real_pts)
            hvU2b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vU2))

            H1=betamodel_02m20.raw_FS_HYM_r(real_pts) 
            H2=betamodel_110m2.raw_FS_HYM_r(real_pts) 
            H3=betamodel_m1m322.raw_FS_HYM_r(real_pts) 

        #print("Now compute the integrals")
        ##(1j) for each FS form
        ##(-j)**3 to get the flat measure dx1dy1dx2dy2dx3dy3
        ##(-1) to rearrange dz wedge dzbar in the v wedge hvb in canonical order.
        #print("The field normalisations:")
        ##check this is constant!
        #HuHu=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vH[:n_p],hvHb[:n_p],pg.lc,pg.lc))
        #print("(Hu,Hu) = " + str(HuHu))
        #Q3Q3=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ3[:n_p],hvQ3b[:n_p],pg.lc,pg.lc))
        #print("(Q3,Q3) = " + str(Q3Q3))
        #U3U3=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU3[:n_p],hvU3b[:n_p],pg.lc,pg.lc))
        #print("(U3,U3) = " + str(U3U3))
        #U1U1=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU1[:n_p],hvU1b[:n_p],pg.lc,pg.lc))
        #print("(U1,U1) = " + str(U1U1))
        #U2U2=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU2[:n_p],hvU2b[:n_p],pg.lc,pg.lc))
        #print("(U2,U2) = " + str(U2U2))
        #Q1Q1=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ1[:n_p],hvQ1b[:n_p],pg.lc,pg.lc))
        #print("(Q1,Q1) = " + str(Q1Q1))
        #Q2Q2=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ2[:n_p],hvQ2b[:n_p],pg.lc,pg.lc))
        #print("(Q2,Q2) = " + str(Q2Q2))

        #print("The field mixings:")
        #Q1Q2=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ1[:n_p],hvQ2b[:n_p],pg.lc,pg.lc))
        #print("(Q1,Q2) = " + str(Q1Q2))
        #Q2Q1=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ2[:n_p],hvQ1b[:n_p],pg.lc,pg.lc))
        #print("(Q2,Q1) = " + str(Q2Q1))
        #U1U2=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU1[:n_p],hvU2b[:n_p],pg.lc,pg.lc))
        #print("(U1,U2) = " + str(U1U2))
        #U2U1=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU2[:n_p],hvU1b[:n_p],pg.lc,pg.lc))
        #print("(U2,U1) = " + str(U2U1))

        #print("Compute holomorphic Yukawas")
        ##consider omega normalisation
        #omega = pg.holomorphic_volume_form(pointsComplex)
        ##put the omega here, not the omegabar
        #omega_normalised_to_one=omega/tf.cast(np.sqrt(volCY_from_Om),complex_dtype) # this is the omega that's normalised to 1. Numerical factor missing for the physical Yukawas?
        #m = [[0,0,0],[0,0,0],[0,0,0]]
        #mwoH = [[0,0,0],[0,0,0],[0,0,0]]

        #tfsqrtandcast=lambda x: tf.cast(tf.math.sqrt(x),complex_dtype)

        #mwoH[0][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ1,vU1)*omega_normalised_to_one)
        #mwoH[0][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ1,vU2)*omega_normalised_to_one)
        #mwoH[0][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ1,vU3)*omega_normalised_to_one)
        #mwoH[1][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ2,vU1)*omega_normalised_to_one)
        #mwoH[1][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ2,vU2)*omega_normalised_to_one)
        #mwoH[1][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ2,vU3)*omega_normalised_to_one)
        #mwoH[2][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ3,vU1)*omega_normalised_to_one)
        #mwoH[2][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ3,vU2)*omega_normalised_to_one)
        #mwoH[2][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ3,vU3)*omega_normalised_to_one)
        #print("without H")
        #print(np.round(np.array(mwoH)*10**4,1))

        #m[0][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU1)*omega_normalised_to_one)
        #m[0][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU2)*omega_normalised_to_one)
        #m[0][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H2),vH,vQ1,vU3)*omega_normalised_to_one)
        #m[1][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU1)*omega_normalised_to_one)
        #m[1][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU2)*omega_normalised_to_one)
        #m[1][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H2),vH,vQ2,vU3)*omega_normalised_to_one)
        #m[2][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU1)*omega_normalised_to_one)
        #m[2][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU2)*omega_normalised_to_one)
        #m[2][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H2*H2),vH,vQ3,vU3)*omega_normalised_to_one)
        #print('proper calculation')
        #print(np.round(np.array(m)*10**4,1))
        ##volCY=tf.reduce_mean(omega)
        #vals = []
        #mAll = []
        #mAll.append(m)
        #vals.append(np.abs([HuHu,Q1Q1,Q2Q2,Q3Q3,U1U1,U2U2,U3U3,Q1Q2,U1U2]))
        #import scipy
        ## 4 is |G|
        #Hmat = 0.5/4*np.array([[HuHu]])
        ##Q1Q2 has Q2 barred, so that should multiply Q2bar which is on the LHS, so it should be on the first column
        #Qmat = 0.5/4*np.array([[Q1Q1,Q2Q1,0],[Q1Q2,Q2Q2,0],[0,0,Q3Q3]])
        #Umat = 0.5/4*np.array([[U1U1,U2U1,0],[U1U2,U2U2,0],[0,0,U3U3]])
        #NormH=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Hmat).astype(complex),complex_dtype))
        #NormQ=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Qmat).astype(complex),complex_dtype))
        #NormU=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Umat).astype(complex),complex_dtype))
        ## second index is the U index
        ##2 sqrt2 comes out, 4 is |G|
        #physical_yukawas=2*np.sqrt(2)/4*NormH[0][0].numpy()*np.einsum('ab,cd,bd->ac',NormQ,NormU,m)
        #print(np.round(physical_yukawas*10**4,1))
        #print(physical_yukawas)
        #mats.append(physical_yukawas)
        ## no conjugating required, no transposing!
        ##check good bundle mterics, compatible
        #prodbundles=tf.reduce_mean(H1*H2*H3)
        #prodbundlesstddev=tf.math.reduce_std(H1*H2*H3)
        #print("good bundle metric" + str(prodbundles) + " and std ddev " + str(prodbundlesstddev))

        #print("masses")
        #u,s,v = np.linalg.svd(physical_yukawas)
        #print(s)
 
        print("Now compute the integrals")
        #(1j) for each FS form #maybe should be (1j/2)??
        #(-j)**3 to get the flat measure dx1dy1dx2dy2dx3dy3 # and this should be (-2j)???
        #(-1) to rearrange dz wedge dzbar in the v wedge hvb in canonical order.
        # whence the -1j/2???? it comes from the hodge star (the one out the front)
        #overall have added a factor of 2
        print("The field normalisations:")
        #check this is constant!
        HuHu=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vH[:n_p],hvHb[:n_p],pg.lc,pg.lc))
        print("(Hu,Hu) = " + str(HuHu))
        Q3Q3=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ3[:n_p],hvQ3b[:n_p],pg.lc,pg.lc))
        print("(Q3,Q3) = " + str(Q3Q3))
        U3U3=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU3[:n_p],hvU3b[:n_p],pg.lc,pg.lc))
        print("(U3,U3) = " + str(U3U3))
        U1U1=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU1[:n_p],hvU1b[:n_p],pg.lc,pg.lc))
        print("(U1,U1) = " + str(U1U1))
        U2U2=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU2[:n_p],hvU2b[:n_p],pg.lc,pg.lc))
        print("(U2,U2) = " + str(U2U2))
        Q1Q1=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ1[:n_p],hvQ1b[:n_p],pg.lc,pg.lc))
        print("(Q1,Q1) = " + str(Q1Q1))
        Q2Q2=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ2[:n_p],hvQ2b[:n_p],pg.lc,pg.lc))
        print("(Q2,Q2) = " + str(Q2Q2))

        print("The field mixings:")
        Q1Q2=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ1[:n_p],hvQ2b[:n_p],pg.lc,pg.lc))
        print("(Q1,Q2) = " + str(Q1Q2))
        Q2Q1=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ2[:n_p],hvQ1b[:n_p],pg.lc,pg.lc))
        print("(Q2,Q1) = " + str(Q2Q1))
        U1U2=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU1[:n_p],hvU2b[:n_p],pg.lc,pg.lc))
        print("(U1,U2) = " + str(U1U2))
        U2U1=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU2[:n_p],hvU1b[:n_p],pg.lc,pg.lc))
        print("(U2,U1) = " + str(U2U1))

        print("Compute holomorphic Yukawas")
        #consider omega normalisation
        omega = pg.holomorphic_volume_form(pointsComplex)
        #put the omega here, not the omegabar
        omega_normalised_to_one=omega/tf.cast(np.sqrt(volCY_from_Om),complex_dtype) # this is the omega that's normalised to 1. Numerical factor missing for the physical Yukawas?
        m = [[0,0,0],[0,0,0],[0,0,0]]
        mwoH = [[0,0,0],[0,0,0],[0,0,0]]

        tfsqrtandcast=lambda x: tf.cast(tf.math.sqrt(x),complex_dtype)

        mwoH[0][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ1,vU1)*omega_normalised_to_one)
        mwoH[0][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ1,vU2)*omega_normalised_to_one)
        mwoH[0][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ1,vU3)*omega_normalised_to_one)
        mwoH[1][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ2,vU1)*omega_normalised_to_one)
        mwoH[1][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ2,vU2)*omega_normalised_to_one)
        mwoH[1][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ2,vU3)*omega_normalised_to_one)
        mwoH[2][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ3,vU1)*omega_normalised_to_one)
        mwoH[2][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ3,vU2)*omega_normalised_to_one)
        mwoH[2][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ3,vU3)*omega_normalised_to_one)


        mwoH = (1/(8*np.sqrt(30))) *   2*np.sqrt(2)/4* np.array(mwoH)
        print("without H")
        print(np.round(np.array(mwoH)*10**4,1))

        m[0][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU1)*omega_normalised_to_one)
        m[0][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU2)*omega_normalised_to_one)
        m[0][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H2),vH,vQ1,vU3)*omega_normalised_to_one)
        m[1][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU1)*omega_normalised_to_one)
        m[1][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU2)*omega_normalised_to_one)
        m[1][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H2),vH,vQ2,vU3)*omega_normalised_to_one)
        m[2][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU1)*omega_normalised_to_one)
        m[2][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU2)*omega_normalised_to_one)
        m[2][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H2*H2),vH,vQ3,vU3)*omega_normalised_to_one)

        #2 sqrt2 comes out, 4 is |G|
        # 8 Sqrt30 is the group theoretic factor
        # 8 Sqrt30 is the group theoretic factor
        m= (1/(8*np.sqrt(30))) *   2*np.sqrt(2)/4* np.array(m)
        print('proper calculation')
        print(np.round(np.array(m)*10**4,1))
        #volCY=tf.reduce_mean(omega)
        vals = []
        mAll = []
        mAll.append(m)
        vals.append(np.abs([HuHu,Q1Q1,Q2Q2,Q3Q3,U1U1,U2U2,U3U3,Q1Q2,U1U2]))
        import scipy
        # 4 is |G|
        # 0.5 is the factor of 2 outside the KIJ integral, which comes from 1/4*2 = 1/2 apparently????
        Hmat = (1/2)* 0.5/4*np.array([[HuHu]])
        #Q1Q2 has Q2 barred, so that should multiply Q2bar which is on the LHS, so it should be on the first column
        #1/2 is the Groups factor
        Qmat = (1/2)* 0.5/4*np.array([[Q1Q1,Q2Q1,0],[Q1Q2,Q2Q2,0],[0,0,Q3Q3]])
        Umat = (1/2)* 0.5/4*np.array([[U1U1,U2U1,0],[U1U2,U2U2,0],[0,0,U3U3]])
        NormH=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Hmat).astype(complex),complex_dtype))
        NormQ=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Qmat).astype(complex),complex_dtype))
        NormU=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Umat).astype(complex),complex_dtype))
        # second index is the U index
        #physical_yukawas=  (1/(8*np.sqrt(30))) *   2*np.sqrt(2)/4*NormH[0][0].numpy()*np.einsum('ab,cd,bd->ac',NormQ,NormU,m)
        physical_yukawas= NormH[0][0].numpy()*np.einsum('ab,cd,bd->ac',NormQ,NormU,m)
        print(np.round(physical_yukawas*10**4,1))
        print(physical_yukawas)
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
        u2,s2,v2 = np.linalg.svd(m)
        print("s ref: ",s)
        print("s holo: ",s2)
        masses_ref.append(s)
        masses_holo.append(s2)

    #write to a CSV file all relevant details: free coefficient, number of epochs for each type, size of the networks, number of points for each, the physical Yukawas, the singular values of the physical Yukawas,
    import csv
    filename = 'FS_results_csv'
    #measures = [measure_phi, measure_LB1, measure_LB2, measure_LB3, measure_vH, measure_vQ3, measure_vU3, measure_vQ1, measure_vQ2, measure_vU1, measure_vU2]
    with open(filename, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([free_coefficient,mats,masses_ref,masses_holo])
        file.close()
