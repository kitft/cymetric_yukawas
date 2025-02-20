from cymetric.config import real_dtype, complex_dtype
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
from generate_and_train_all_nnsADAPTED import *


# nPoints=300000
# free_coefficient=1.# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
# nEpochsPhi=100
# nEpochsBeta=60
# nEpochsSigma=50


nPoints=300000
nPointsHF=300000
#free_coefficient=1.000000004# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
import sys
#free_coefficient=1.# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
#nEpochsPhi=100
nEpochsPhi=100
nEpochsBeta=100
nEpochsSigma=50
depthPhi=3
widthPhi=196#128 4 in the 1.0s
depthBeta=3
widthBeta=196
depthSigma=3
widthSigma=256 # up from 256
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

train_phi=False
train_02m20 = False
train_110m2 = False
train_m1m322 = False
train_vH = False
train_vQ3 = False
train_vU3 = False
train_vQ1 = False
train_vQ2 = False
train_vU1 = True 
train_vU2 =  True


arg= sys.argv[1]
#free_coefficient = np.array([int(hi) for hi in arg[1:-1] if hi != ','])
free_coefficient = np.array([int(hi) for hi in arg[1:-1].split(',')])
#free_coefficient=np.array(sys.argv[1])
free_coefficient_name=hash(tuple(free_coefficient))
print('set of coeffs')
print(free_coefficient)
print("hash")
print(hash(tuple(free_coefficient)))
print('ok')


if __name__ ==  '__main__':
    generate_points_and_save_using_defaults(free_coefficient,nPoints)
    if train_phi:
        phimodel1,training_history=train_and_save_nn(free_coefficient,depthPhi,widthPhi,nEpochsPhi,[64,50000],0.001) 
    else:
        phimodel1,training_history=load_nn_phimodel(free_coefficient,depthPhi,widthPhi,nEpochsPhi,[64,50000],set_weights_to_zero=False)

    linebundleforHYM_02m20=np.array([0,2,-2,0]) 
    linebundleforHYM_110m2=np.array([1,1,0,-2]) 
    linebundleforHYM_m1m322=np.array([-1,-3,2,2]) 
    gc.collect()
    tf.keras.backend.clear_session()


    generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM_02m20,nPoints,phimodel1,force_generate=False)
    if train_phi:
        betamodel_02m20,training_historyBeta_02m20=train_and_save_nn_HYM(free_coefficient,linebundleforHYM_02m20,depthBeta,widthBeta,nEpochsBeta,[64,50000],0.001,alpha=alphabeta)
    else: 
        betamodel_02m20,training_historyBeta_02m20=load_nn_HYM(free_coefficient,linebundleforHYM_02m20,depthBeta,widthBeta,nEpochsBeta,[64,50000],set_weights_to_zero=False)
    gc.collect()
    tf.keras.backend.clear_session()


    #generate_points_and_save_using_defaultsHYM(free_coefficient,-linebundleforHYM_02m20,nPoints,phimodel1,force_generate=False)
    #mbetamodel_02m20,mtraining_historyBeta_02m20=train_and_save_nn_HYM(free_coefficient,-linebundleforHYM_02m20,depthBeta,widthBeta,nEpochsBeta,[64,50000],0.001,alpha=alphabeta)
    #mbetamodel_02m20,mtraining_historymBeta_02m20=load_nn_HYM(free_coefficient,-linebundleforHYM_02m20,depthBeta,widthBeta,nEpochsBeta,[64,50000],set_weights_to_zero=False)
    
    
    
    pg,kmoduli=generate_points_and_save_using_defaults_for_eval(free_coefficient,300000)
    dataEval=np.load(os.path.join('tetraquadric_pg_for_eval_with_'+str(free_coefficient_name), 'dataset.npz'))
    n_p = 300000
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
    if train_110m2:
        betamodel_110m2,training_historyBeta_110m2=train_and_save_nn_HYM(free_coefficient,linebundleforHYM_110m2,depthBeta,widthBeta,nEpochsBeta,[64,50000],0.001,alpha=alphabeta)
    else:
        betamodel_110m2,training_historyBeta_110m2=load_nn_HYM(free_coefficient,linebundleforHYM_110m2,depthBeta,widthBeta,nEpochsBeta,[64,50000],set_weights_to_zero=False)
    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM_m1m322,nPoints,phimodel1,force_generate=False)
    if train_m1m322:
        betamodel_m1m322,training_historyBeta_m1m322=train_and_save_nn_HYM(free_coefficient,linebundleforHYM_m1m322,depthBeta,widthBeta,nEpochsBeta,[64,50000],0.001,alpha=alphabeta)
    else:
        betamodel_m1m322,training_historyBeta_m1m322=load_nn_HYM(free_coefficient,linebundleforHYM_m1m322,depthBeta,widthBeta,nEpochsBeta,[64,50000],set_weights_to_zero=False)

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


    #generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_02m20,functionforbaseharmonicform_jbar_for_vH,phi,betamodel_02m20,nPointsHF,force_generate=False)
    #if train_vH:
    #    HFmodel_vH,trainingHistoryHF_vH=train_and_save_nn_HF(free_coefficient,linebundleforHYM_02m20,betamodel_02m20,functionforbaseharmonicform_jbar_for_vH,depthSigma,widthSigma,nEpochsSigma,[64,50000],0.001,alpha=alphasigma1)
    #else:
    #    HFmodel_vH,trainingHistoryHF_vH=load_nn_HF(free_coefficient,linebundleforHYM_02m20,betamodel_02m20,functionforbaseharmonicform_jbar_for_vH,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma1)
    #gc.collect()
    #tf.keras.backend.clear_session()

    #generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_110m2,functionforbaseharmonicform_jbar_for_vQ3,phi,betamodel_110m2,nPointsHF,force_generate=False)
    #if train_vQ3:
    #    HFmodel_vQ3,trainingHistoryHF_vQ3=train_and_save_nn_HF(free_coefficient,linebundleforHYM_110m2,betamodel_110m2,functionforbaseharmonicform_jbar_for_vQ3,depthSigma,widthSigma,nEpochsSigma,[64,50000],0.001,alpha=alphasigma1)
    #else:
    #    HFmodel_vQ3,trainingHistoryHF_vQ3=load_nn_HF(free_coefficient,linebundleforHYM_110m2,betamodel_110m2,functionforbaseharmonicform_jbar_for_vQ3,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma1)
    #gc.collect()
    #tf.keras.backend.clear_session()

    #generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_110m2,functionforbaseharmonicform_jbar_for_vU3,phi,betamodel_110m2,nPointsHF,force_generate=False)
    #if train_vU3:
    #    HFmodel_vU3,trainingHistoryHF_vU3=train_and_save_nn_HF(free_coefficient,linebundleforHYM_110m2,betamodel_110m2,functionforbaseharmonicform_jbar_for_vU3,depthSigma,widthSigma,nEpochsSigma,[64,50000],0.001,alpha=alphasigma1)
    #else:
    #    HFmodel_vU3,trainingHistoryHF_vU3=load_nn_HF(free_coefficient,linebundleforHYM_110m2,betamodel_110m2,functionforbaseharmonicform_jbar_for_vU3,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma1)

    #gc.collect()
    #tf.keras.backend.clear_session()






    #nEpochsSigma=75

    #generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_m1m322,vFS_Q1,phi,betamodel_m1m322,nPointsHF,force_generate=False)
    #if train_vQ1:
    #    HFmodel_vQ1,trainingHistoryHF_vQ1=train_and_save_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_Q1,depthSigma,widthSigma,nEpochsSigma,[64,50000],0.001,alpha=alphasigma2)
    #else:
    #    HFmodel_vQ1,trainingHistoryHF_vQ1=load_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_Q1,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma2)
    #gc.collect()
    #tf.keras.backend.clear_session()

    #generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_m1m322,vFS_Q2,phi,betamodel_m1m322,nPointsHF,force_generate=False)
    #if train_vQ2:   
    #    HFmodel_vQ2,trainingHistoryHF_vQ2=train_and_save_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_Q2,depthSigma,widthSigma,nEpochsSigma,[64,50000],0.001,alpha=alphasigma2)
    #else:
    #    HFmodel_vQ2,trainingHistoryHF_vQ2=load_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_Q2,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma2)
    #gc.collect()
    #tf.keras.backend.clear_session()

    #generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_m1m322,vFS_U1,phi,betamodel_m1m322,nPointsHF,force_generate=False)
    #if train_vU1:
    #    HFmodel_vU1,trainingHistoryHF_vU1=train_and_save_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_U1,depthSigma,widthSigma,nEpochsSigma,[64,50000],0.001,alpha=alphasigma2)
    #else:
    #    HFmodel_vU1,trainingHistoryHF_vU1=load_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_U1,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma2)
    #gc.collect()
    #tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_m1m322,vFS_U2,phi,betamodel_m1m322,nPointsHF,force_generate=False)
    if train_vU2:
        HFmodel_vU2,trainingHistoryHF_vU2=train_and_save_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_U2,depthSigma,widthSigma,nEpochsSigma,[64,50000],0.001,alpha=alphasigma2)
    else:
        HFmodel_vU2,trainingHistoryHF_vU2=load_nn_HF(free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_U2,depthSigma,widthSigma,nEpochsSigma,[64,50000],alpha=alphasigma2)
    gc.collect()
    tf.keras.backend.clear_session()

    print("\n do analysis: \n\n\n")

    Vol_from_dijk_J=pg.get_volume_from_intersections(kmoduli)
    Vol_reference_dijk_with_k_is_1=pg.get_volume_from_intersections(np.ones_like(kmoduli)) 
    volCY_from_Om=tf.reduce_mean(dataEval['y_train'][:n_p,0])

    aux_weights=(dataEval['y_train'][:n_p,0]/(dataEval['y_train'][:n_p,1]))*(1/(6))### these are the appropriate 'flat measure' so sum_i aux_weights f is int_X f(x)
    #convertcomptoreal=lambda complexvec: tf.concat([tf.math.real(complexvec),tf.math.imag(complexvec)],-1) # this converts from complex to real

    mats=[]
    use_trained = False
    for use_trained in [True,False]:
        if use_trained:
            mets=phi(real_pts)
            dets=tf.linalg.det(mets)

            vH =  HFmodel_vH.corrected_harmonicform(real_pts)
            hvHb =  tf.einsum('x,xb->xb',tf.cast(betamodel_02m20(real_pts),complex_dtype),tf.math.conj(vH))
            vQ3 = HFmodel_vQ3.corrected_harmonicform(real_pts)
            hvQ3b = tf.einsum('x,xb->xb',tf.cast(betamodel_110m2(real_pts),complex_dtype),tf.math.conj(vQ3))
            vU3 = HFmodel_vU3.corrected_harmonicform(real_pts)
            hvU3b = tf.einsum('x,xb->xb',tf.cast(betamodel_110m2(real_pts),complex_dtype),tf.math.conj(vU3))

            vQ1 = HFmodel_vQ1.corrected_harmonicform(real_pts)
            hvQ1b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322(real_pts),complex_dtype),tf.math.conj(vQ1))
            vQ2 = HFmodel_vQ2.corrected_harmonicform(real_pts)
            hvQ2b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322(real_pts),complex_dtype),tf.math.conj(vQ2))

            vU1 = HFmodel_vU1.corrected_harmonicform(real_pts)
            hvU1b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322(real_pts),complex_dtype),tf.math.conj(vU1))
            vU2 = HFmodel_vU2.corrected_harmonicform(real_pts)
            hvU2b = tf.einsum('x,xb->xb',tf.cast(betamodel_m1m322(real_pts),complex_dtype),tf.math.conj(vU2))

            H1=betamodel_02m20(real_pts) 
            H2=betamodel_110m2(real_pts) 
            H3=betamodel_m1m322(real_pts) 
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

        print("Now compute the integrals")
        #(1j) for each FS form
        #(-j)**3 to get the flat measure dx1dy1dx2dy2dx3dy3
        #(-1) to rearrange dz wedge dzbar in the v wedge hvb in canonical order.
        print("The field normalisations:")
        #check this is constant!
        HuHu=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vH[:n_p],hvHb[:n_p],pg.lc,pg.lc))
        print("(Hu,Hu) = " + str(HuHu))
        Q3Q3=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ3[:n_p],hvQ3b[:n_p],pg.lc,pg.lc))
        print("(Q3,Q3) = " + str(Q3Q3))
        U3U3=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU3[:n_p],hvU3b[:n_p],pg.lc,pg.lc))
        print("(U3,U3) = " + str(U3U3))
        U1U1=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU1[:n_p],hvU1b[:n_p],pg.lc,pg.lc))
        print("(U1,U1) = " + str(U1U1))
        U2U2=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU2[:n_p],hvU2b[:n_p],pg.lc,pg.lc))
        print("(U2,U2) = " + str(U2U2))
        Q1Q1=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ1[:n_p],hvQ1b[:n_p],pg.lc,pg.lc))
        print("(Q1,Q1) = " + str(Q1Q1))
        Q2Q2=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ2[:n_p],hvQ2b[:n_p],pg.lc,pg.lc))
        print("(Q2,Q2) = " + str(Q2Q2))

        print("The field mixings:")
        Q1Q2=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ1[:n_p],hvQ2b[:n_p],pg.lc,pg.lc))
        print("(Q1,Q2) = " + str(Q1Q2))
        Q2Q1=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ2[:n_p],hvQ1b[:n_p],pg.lc,pg.lc))
        print("(Q2,Q1) = " + str(Q2Q1))
        U1U2=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU1[:n_p],hvU2b[:n_p],pg.lc,pg.lc))
        print("(U1,U2) = " + str(U1U2))
        U2U1=(-1j/2)*(1j)**2*(-1j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU2[:n_p],hvU1b[:n_p],pg.lc,pg.lc))
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
        print('proper calculation')
        print(np.round(np.array(m)*10**4,1))
        #volCY=tf.reduce_mean(omega)
        vals = []
        mAll = []
        mAll.append(m)
        vals.append(np.abs([HuHu,Q1Q1,Q2Q2,Q3Q3,U1U1,U2U2,U3U3,Q1Q2,U1U2]))
        import scipy
        # 4 is |G|
        Hmat = 0.5/4*np.array([[HuHu]])
        #Q1Q2 has Q2 barred, so that should multiply Q2bar which is on the LHS, so it should be on the first column
        Qmat = 0.5/4*np.array([[Q1Q1,Q2Q1,0],[Q1Q2,Q2Q2,0],[0,0,Q3Q3]])
        Umat = 0.5/4*np.array([[U1U1,U2U1,0],[U1U2,U2U2,0],[0,0,U3U3]])
        NormH=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Hmat).astype(complex),complex_dtype))
        NormQ=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Qmat).astype(complex),complex_dtype))
        NormU=tf.linalg.inv(tf.cast(scipy.linalg.sqrtm(Umat).astype(complex),complex_dtype))
        # second index is the U index
        #2 sqrt2 comes out, 4 is |G|
        physical_yukawas=2*np.sqrt(2)/4*NormH[0][0].numpy()*np.einsum('ab,cd,bd->ac',NormQ,NormU,m)
        print(np.round(physical_yukawas*10**4,1))
        print(physical_yukawas)
        mats.append(physical_yukawas)
        # no conjugating required, no transposing!
        #check good bundle mterics, compatible
        prodbundles=tf.reduce_mean(H1*H2*H3)
        prodbundlesstddev=tf.math.reduce_std(H1*H2*H3)
        print("good bundle metric" + str(prodbundles) + " and std ddev " + str(prodbundlesstddev))
        print("masses")
        u,s,v = np.linalg.svd(physical_yukawas)
        print(s)
