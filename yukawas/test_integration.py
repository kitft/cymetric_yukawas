#if __name__ ==  '__main__':
#    from multiprocessing import set_start_method
#    set_start_method('spawn')
#import multiprocessing
import time
import os
start_time_of_process = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_PROFILER"] = "1"
from tensorflow.python.client import device_lib
if __name__ == '__main__':
    print(device_lib.list_local_devices())

import csv
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

if __name__ == '__main__':
    run_eagerly = False

    tf.config.run_functions_eagerly(run_eagerly)

    print("Running with eager execution:",run_eagerly)

    from cymetric.config import real_dtype, complex_dtype, set_double_precision

    if 'doubleprecision' in sys.argv[1:]:
        double_precision = True
    else:
        double_precision = False
    print("Running with double precision?:", double_precision)
    set_double_precision(double_precision)

    tf.get_logger().setLevel('ERROR')


from cymetric.models.tfmodels import PhiFSModel, MultFSModel, FreeModel, MatrixFSModel, AddFSModel, PhiFSModelToric, MatrixFSModelToric
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.callbacks import SigmaCallback, KaehlerCallback, TransitionCallback, RicciCallback, VolkCallback, AlphaCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, RicciLoss, VolkLoss, TotalLoss

from NewCustomMetrics import *
from HarmonicFormModel import *
from BetaModel import *
from laplacian_funcs import *
name_of_run = "model13"
from OneAndTwoFormsForLineBundlesModel13 import *
#from generate_and_train_all_nns import *
from generate_and_train_all_nnsHOLOModel13 import *
from auxiliary_funcs import *

print_memory_usage(start_time_of_process=start_time_of_process, name = __name__)

if __name__ == '__main__':
    tr_batchsize=64
    #free_coefficient=1.000000004# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
    import sys
    free_coefficient = float(sys.argv[1])
    seed_for_gen=int((int(free_coefficient*100000000000)+free_coefficient*1000000))%4294967294 # modulo largest seed
    print("seed for gen", seed_for_gen)


    alphabeta=[1,10]
    alphasigma1=[1,10] # down form 50
    alphasigma2=[1,80] # down from 200

    lRatePhi= 0.005
    lRateBeta=0.005
    lRateSigma=0.001# perhaps best as 0.03
    lRateSigma=0.001#//10# perhaps best as 0.03
    lRateSigma2=0.001#//10# perhaps best as 0.03

    print("Learning rate: phi:",lRatePhi)
    print("Learning rate: beta:",lRateBeta)
    print("Learning rate: sigma:",lRateSigma)
    print("Learning rate decays by a factor, typically 10, over the course of learning")


    # Use command-line arguments if provided, starting from the second argument
    start_from = sys.argv[2] if len(sys.argv) > 2 else 'phi'

    training_flags = determine_training_flags(start_from)
    
    # Unpack training flags
    train_phi, train_02m20, train_001m3, train_0m213, train_vH, train_vQ3, train_vU3, train_vQ1, train_vQ2, train_vU1, train_vU2 = (
        training_flags['phi'],
        training_flags['LB1'],
        training_flags['LB2'],
        training_flags['LB3'],
        training_flags['vH'],
        training_flags['vQ3'],
        training_flags['vU3'],
        training_flags['vQ1'],
        training_flags['vQ2'],
        training_flags['vU1'],
        training_flags['vU2']
    )
    
    
    stddev_Q12U12=1.0#0.5
    final_layer_scale_Q12U12=1.0#0.3#0.000001#0.01
    stddev_Q12U12=0.5#0.5
    final_layer_scale_Q12U12=0.01#0.3#0.000001#0.01
    #lRateSigma=0.0
    
    
    stddev_H33=0.5
    stddev_H33=1
    #stddev_H33=1
    #final_layer_scale_H33=0.1
    final_layer_scale_H33=0.01#0.01
    
    skip_measuresPhi=True
    skip_measuresBeta=True
    skip_measuresHF=True
    
    force_generate_phi=True
    force_generate_HYM=True
    force_generate_HF=True
    force_generate_HF_2=True
    
    return_zero_phi= True
    return_zero_HYM = True
    return_zero_HF = False
    return_zero_HF_2 = False
    
    SecondBSize=1000
    n_to_integrate=1000000
    #n_to_integrate=100000
    
    use_zero_network_phi = True
    
    nPoints = 1000
    nPointsHF = 1000
    if 'small' in sys.argv[1:]:
        n_to_integrate = 1000
    elif 'allbig' in sys.argv[1:]:
        nPoints = 100000
        nPointsHF = 100000
        n_to_integrate = 100000
    elif 'allhuge' in sys.argv[1:]:
        nPoints = 1000000
        nPointsHF = 1000000
        n_to_integrate = 1000000
    else:
        n_to_integrate = 1000000
    #tr_batchsize = 10
    #SecondBSize = 10
    nEpochsPhi = 1
    nEpochsBeta = 1
    nEpochsSigma = 1
    nEpochsSigma2 = 1
    
    depthPhi = 2
    widthPhi = 3
    depthBeta = 2
    widthBeta = 3
    depthSigma = 2
    widthSigma = 3
    depthSigma2 = 2
    widthSigma2 = 3
    
    return_random_phi = False
    return_random_HYM = False
    return_random_HF = True
    return_random_HF_2 = True
    
    print("Number of points: " + str(nPoints), "Number of points HF: " + str(nPointsHF), "Number of points to integrate: " + str(n_to_integrate))
    
    

if __name__ ==  '__main__':


    
    generate_points_and_save_using_defaults(free_coefficient,nPoints,seed_set=seed_for_gen)
    if train_phi:
        #phimodel1,training_history=train_and_save_nn(free_coefficient,depthPhi,widthPhi,nEpochsPhi,bSizes=[64,tr_batchsize],lRate=lRatePhi) 
        phimodel1,training_history, measure_phi=train_and_save_nn(free_coefficient,depthPhi,widthPhi,nEpochsPhi,stddev=0.05,bSizes=[64,SecondBSize],lRate=lRatePhi,use_zero_network=use_zero_network_phi)
    else:
        phimodel1,training_history, measure_phi=load_nn_phimodel(free_coefficient,depthPhi,widthPhi,nEpochsPhi,[64,SecondBSize],set_weights_to_zero=return_zero_phi,skip_measures=skip_measuresPhi,set_weights_to_random=return_random_phi)

    linebundleforHYM_02m20=np.array([0,2,-2,0]) 
    linebundleforHYM_001m3=np.array([0,0,1,-3]) 
    linebundleforHYM_0m213=np.array([0,-2,1,3]) 

    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()


    generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM_02m20,nPoints,phimodel1,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_02m20:
        betamodel_02m20,training_historyBeta_02m20, measure_LB1 = train_and_save_nn_HYM(free_coefficient,linebundleforHYM_02m20,depthBeta,widthBeta,nEpochsBeta,stddev=0.05,bSizes=[64],lRate=lRateBeta,alpha=alphabeta,load_network=False)
    else: 
        betamodel_02m20,training_historyBeta_02m20, measure_LB1=load_nn_HYM(free_coefficient,linebundleforHYM_02m20,depthBeta,widthBeta,nEpochsBeta,[64],set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM)
    
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])


    #generate_points_and_save_using_defaultsHYM(free_coefficient,-linebundleforHYM_02m20,nPoints,phimodel1,force_generate=True)
    #mbetamodel_02m20,mtraining_historyBeta_02m20, measure = train_and_save_nn_HYM(free_coefficient,-linebundleforHYM_02m20,depthBeta,widthBeta,nEpochsBeta,[64],0.001,alpha=alphabeta)
    #mbetamodel_02m20,mtraining_historymBeta_02m20=load_nn_HYM(free_coefficient,-linebundleforHYM_02m20,depthBeta,widthBeta,nEpochsBeta,[64],set_weights_to_zero=False)
    
    
    
    generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM_001m3,nPoints,phimodel1,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_001m3:
        betamodel_001m3,training_historyBeta_001m3, measure_LB2 = train_and_save_nn_HYM(free_coefficient,linebundleforHYM_001m3,depthBeta,widthBeta,nEpochsBeta,stddev=0.05,bSizes=[64],lRate=lRateBeta,alpha=alphabeta,load_network=False)
    else:
        betamodel_001m3,training_historyBeta_001m3, measure_LB2=load_nn_HYM(free_coefficient,linebundleforHYM_001m3,depthBeta,widthBeta,nEpochsBeta,[64],set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM)
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM_0m213,nPoints,phimodel1,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_0m213:
        betamodel_0m213,training_historyBeta_0m213, measure_LB3 = train_and_save_nn_HYM(free_coefficient,linebundleforHYM_0m213,depthBeta,widthBeta,nEpochsBeta,stddev=0.05,bSizes=[64],lRate=lRateBeta,alpha=alphabeta,load_network=False)
    else:
        betamodel_0m213,training_historyBeta_0m213, measure_LB3=load_nn_HYM(free_coefficient,linebundleforHYM_0m213,depthBeta,widthBeta,nEpochsBeta,[64],set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM)

    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    phi=phimodel1

    gc.collect()
    tf.keras.backend.clear_session()

    ##tf.profiler.experimental.start('/mnt/extraspace/kitft/tf_profile_holo_'+str(free_coefficient))
    #tf.profiler.experimental.start('/mnt/extraspace/kitft/tf_profile_holo_'+str(free_coefficient), options=tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,python_tracer_level=1, device_tracer_level=1))

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_02m20,functionforbaseharmonicform_jbar_for_vH,phi,betamodel_02m20,nPointsHF,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vH:
        #HFmodel_vH,trainingHistoryHF_vH, measure = train_and_save_nn_HF(free_coefficient,linebundleforHYM_02m20,betamodel_02m20,functionforbaseharmonicform_jbar_for_vH,depthSigma+2,widthSigma//2,nEpochsSigma,[64],lRate=lRateSigma/10,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
        HFmodel_vH,trainingHistoryHF_vH, measure_HF1 = train_and_save_nn_HF(free_coefficient,linebundleforHYM_02m20,betamodel_02m20,phimodel1,functionforbaseharmonicform_jbar_for_vH,depthSigma,widthSigma,nEpochsSigma,[64],lRate=lRateSigma,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
        #was /100
    else:
        HFmodel_vH,trainingHistoryHF_vH, measure_HF1=load_nn_HF(free_coefficient,linebundleforHYM_02m20,betamodel_02m20,phimodel1,functionforbaseharmonicform_jbar_for_vH,depthSigma,widthSigma,nEpochsSigma,[64],alpha=alphasigma1,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)


    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    #nEpochsSigma=50#changed below vH
    #nEpochsSigma=10

    # Stop the profiler
    #print("stopping profiler")
    #tf.profiler.experimental.stop()
    #print('next')



    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_001m3,functionforbaseharmonicform_jbar_for_vQ3,phi,betamodel_001m3,nPointsHF,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vQ3:
        HFmodel_vQ3,trainingHistoryHF_vQ3, measure_HF2 = train_and_save_nn_HF(free_coefficient,linebundleforHYM_001m3,betamodel_001m3,phimodel1,functionforbaseharmonicform_jbar_for_vQ3,depthSigma,widthSigma,nEpochsSigma,[64],lRate=lRateSigma,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
    else:
        HFmodel_vQ3,trainingHistoryHF_vQ3, measure_HF2=load_nn_HF(free_coefficient,linebundleforHYM_001m3,betamodel_001m3,phimodel1,functionforbaseharmonicform_jbar_for_vQ3,depthSigma,widthSigma,nEpochsSigma,[64],alpha=alphasigma1,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    #nEpochsSigma=10
    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_001m3,functionforbaseharmonicform_jbar_for_vU3,phi,betamodel_001m3,nPointsHF,force_generate=force_generate_HF,seed_set=seed_for_gen)

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    if train_vU3:
        HFmodel_vU3,trainingHistoryHF_vU3, measure_HF3 = train_and_save_nn_HF(free_coefficient,linebundleforHYM_001m3,betamodel_001m3,phimodel1,functionforbaseharmonicform_jbar_for_vU3,depthSigma,widthSigma,nEpochsSigma,[64],lRate=lRateSigma,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
    else:
        HFmodel_vU3,trainingHistoryHF_vU3, measure_HF3=load_nn_HF(free_coefficient,linebundleforHYM_001m3,betamodel_001m3,phimodel1,functionforbaseharmonicform_jbar_for_vU3,depthSigma,widthSigma,nEpochsSigma,[64],alpha=alphasigma1,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()






    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_0m213,functionforbaseharmonicform_jbar_for_vQ1,phi,betamodel_0m213,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ1:
        HFmodel_vQ1,trainingHistoryHF_vQ1, measure_HF4 = train_and_save_nn_HF(free_coefficient,linebundleforHYM_0m213,betamodel_0m213,phimodel1,functionforbaseharmonicform_jbar_for_vQ1,depthSigma2,widthSigma2,nEpochsSigma2,[64],lRate=lRateSigma2,alpha=alphasigma2,stddev=stddev_Q12U12,final_layer_scale=final_layer_scale_Q12U12)
    else:
        HFmodel_vQ1,trainingHistoryHF_vQ1, measure_HF4=load_nn_HF(free_coefficient,linebundleforHYM_0m213,betamodel_0m213,phimodel1,functionforbaseharmonicform_jbar_for_vQ1,depthSigma2,widthSigma2,nEpochsSigma2,[64],alpha=alphasigma2,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_0m213,functionforbaseharmonicform_jbar_for_vQ2,phi,betamodel_0m213,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ2:   
        HFmodel_vQ2,trainingHistoryHF_vQ2, measure_HF5 = train_and_save_nn_HF(free_coefficient,linebundleforHYM_0m213,betamodel_0m213,phimodel1,functionforbaseharmonicform_jbar_for_vQ2,depthSigma2,widthSigma2,nEpochsSigma2,[64],lRate=lRateSigma2,alpha=alphasigma2,stddev=stddev_Q12U12,final_layer_scale=final_layer_scale_Q12U12)
    else:
        HFmodel_vQ2,trainingHistoryHF_vQ2, measure_HF5=load_nn_HF(free_coefficient,linebundleforHYM_0m213,betamodel_0m213,phimodel1,functionforbaseharmonicform_jbar_for_vQ2,depthSigma2,widthSigma2,nEpochsSigma2,[64],alpha=alphasigma2,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_0m213,functionforbaseharmonicform_jbar_for_vU1,phi,betamodel_0m213,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU1:
        HFmodel_vU1,trainingHistoryHF_vU1, measure_HF6 = train_and_save_nn_HF(free_coefficient,linebundleforHYM_0m213,betamodel_0m213,phimodel1,functionforbaseharmonicform_jbar_for_vU1,depthSigma2,widthSigma2,nEpochsSigma2,[64],lRate=lRateSigma2,alpha=alphasigma2,stddev=stddev_Q12U12,final_layer_scale=final_layer_scale_Q12U12)
    else:
        HFmodel_vU1,trainingHistoryHF_vU1, measure_HF6=load_nn_HF(free_coefficient,linebundleforHYM_0m213,betamodel_0m213,phimodel1,functionforbaseharmonicform_jbar_for_vU1,depthSigma2,widthSigma2,nEpochsSigma2,[64],alpha=alphasigma2,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM_0m213,functionforbaseharmonicform_jbar_for_vU2,phi,betamodel_0m213,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU2:
        HFmodel_vU2,trainingHistoryHF_vU2, measure_HF7 = train_and_save_nn_HF(free_coefficient,linebundleforHYM_0m213,betamodel_0m213,phimodel1,functionforbaseharmonicform_jbar_for_vU2,depthSigma2,widthSigma2,nEpochsSigma2,[64],lRate=lRateSigma2,alpha=alphasigma2,stddev=stddev_Q12U12,final_layer_scale=final_layer_scale_Q12U12)
    else:
        HFmodel_vU2,trainingHistoryHF_vU2, measure_HF7=load_nn_HF(free_coefficient,linebundleforHYM_0m213,betamodel_0m213,phimodel1,functionforbaseharmonicform_jbar_for_vU2,depthSigma2,widthSigma2,nEpochsSigma2,[64],alpha=alphasigma2,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    
    
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])



    pg,kmoduli=generate_points_and_save_using_defaults_for_eval(free_coefficient,n_to_integrate,seed_set=seed_for_gen)
    dataEval=np.load(os.path.join('dataM13/tetraquadric_pg_for_eval_with_'+str(free_coefficient), 'dataset.npz'))
    n_p = n_to_integrate 
    phi = phimodel1
    points64=dataEval['X_train'][0:n_p]
    real_pts=tf.cast(dataEval['X_train'][0:n_p],real_dtype)
    pointsComplex=tf.cast(point_vec_to_complex(points64),complex_dtype)# we have to cast to complex64 from complex128 sometimes. Just for safety. point_vec_to_complex does the obvious thing



    # H1=betamodel_02m20(real_pts)
    # H2=betamodel_001m3(real_pts)
    # H3=betamodel_0m213(real_pts)


    # prodbundles=tf.reduce_mean(H1*H2*H3)
    # prodbundlesstddev=tf.math.reduce_std(H1*H2*H3)
    # print("test if good bundle metric" + str(prodbundles) + " and std ddev " + str(prodbundlesstddev))



    print("\n do analysis: \n\n\n")

    Vol_from_dijk_J=pg.get_volume_from_intersections(kmoduli)
    Vol_reference_dijk_with_k_is_1=pg.get_volume_from_intersections(np.ones_like(kmoduli)) 
    #volCY_from_Om=tf.reduce_mean(dataEval['y_train'][:n_p,0])
    print("Compute holomorphic Yukawas")
    volCY_from_Om=tf.reduce_mean(dataEval['y_train'][:n_p,0])/6 #this is the actual volume of the CY computed from Omega.since J^J^J^ = K Om^Ombar?? not totally sure here:
    #consider omega normalisation
    omega = tf.cast(batch_process_helper_func(pg.holomorphic_volume_form, [pointsComplex], batch_indices=[0], batch_size=100000),complex_dtype)
    #put the omega here, not the omegabar
    omega_normalised_to_one=omega/tf.cast(np.sqrt(volCY_from_Om),complex_dtype) # this is the omega that's normalised to 1. VERIFIED yes.

    # this is integral omega wedge omegabar? yeah.

    aux_weights=(dataEval['y_train'][:n_p,0]/(dataEval['y_train'][:n_p,1]))*(1/(6))### these are the appropriate 'flat measure' so sum_i aux_weights f is int_X f(x)
    #convertcomptoreal=lambda complexvec: tf.concat([tf.math.real(complexvec),tf.math.imag(complexvec)],-1) # this converts from complex to real

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    batch_size_for_processing=min(100000, len(real_pts))
    mats=[]
    masses_ref_and_learned=[]
    use_trained = False
    holomorphic_Yukawas=[]
    for use_trained in [True,False]:
        print("using trained? " + str(use_trained))
        if use_trained:
            mets = batch_process_helper_func(phi, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing)
            print('got mets', flush=True)
            dets = tf.linalg.det(mets)
            # Batch process corrected harmonic forms
            vH = batch_process_helper_func(HFmodel_vH.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing)
            hvHb = tf.einsum('x,xb->xb', tf.cast(betamodel_02m20(real_pts), complex_dtype), tf.math.conj(vH))
            print('got vH', flush=True)
            
            vQ3 = batch_process_helper_func(HFmodel_vQ3.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing)
            hvQ3b = tf.einsum('x,xb->xb', tf.cast(betamodel_001m3(real_pts), complex_dtype), tf.math.conj(vQ3))
            print('got vQ3', flush=True)
            
            vU3 = batch_process_helper_func(HFmodel_vU3.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing)
            hvU3b = tf.einsum('x,xb->xb', tf.cast(betamodel_001m3(real_pts), complex_dtype), tf.math.conj(vU3))
            print('got vU3', flush=True)
            
            vQ1 = batch_process_helper_func(HFmodel_vQ1.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing)
            hvQ1b = tf.einsum('x,xb->xb', tf.cast(betamodel_0m213(real_pts), complex_dtype), tf.math.conj(vQ1))
            print('got vQ1', flush=True)
            
            vQ2 = batch_process_helper_func(HFmodel_vQ2.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing)
            hvQ2b = tf.einsum('x,xb->xb', tf.cast(betamodel_0m213(real_pts), complex_dtype), tf.math.conj(vQ2))
            print('got vQ2', flush=True)
            
            vU1 = batch_process_helper_func(HFmodel_vU1.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing)
            hvU1b = tf.einsum('x,xb->xb', tf.cast(betamodel_0m213(real_pts), complex_dtype), tf.math.conj(vU1))
            print('got vU1', flush=True)
            
            vU2 = batch_process_helper_func(HFmodel_vU2.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing)
            hvU2b = tf.einsum('x,xb->xb', tf.cast(betamodel_0m213(real_pts), complex_dtype), tf.math.conj(vU2))
            print('got vU2', flush=True)
            H1=betamodel_02m20(real_pts) 
            H2=betamodel_001m3(real_pts) 
            H3=betamodel_0m213(real_pts) 
        elif not use_trained:
            mets = phi.fubini_study_pb(real_pts,ts=tf.cast(kmoduli,complex_dtype))
            dets = tf.linalg.det(mets)
            vH =  HFmodel_vH.uncorrected_FS_harmonicform(real_pts)
            hvHb =  tf.einsum('x,xb->xb',tf.cast(betamodel_02m20.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vH))
            vQ3 = HFmodel_vQ3.uncorrected_FS_harmonicform(real_pts)
            hvQ3b = tf.einsum('x,xb->xb',tf.cast(betamodel_001m3.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vQ3))
            vU3 = HFmodel_vU3.uncorrected_FS_harmonicform(real_pts)
            hvU3b = tf.einsum('x,xb->xb',tf.cast(betamodel_001m3.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vU3))

            vQ1 = HFmodel_vQ1.uncorrected_FS_harmonicform(real_pts)
            hvQ1b = tf.einsum('x,xb->xb',tf.cast(betamodel_0m213.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vQ1))
            vQ2 = HFmodel_vQ2.uncorrected_FS_harmonicform(real_pts)
            hvQ2b = tf.einsum('x,xb->xb',tf.cast(betamodel_0m213.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vQ2))

            vU1 = HFmodel_vU1.uncorrected_FS_harmonicform(real_pts)
            hvU1b = tf.einsum('x,xb->xb',tf.cast(betamodel_0m213.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vU1))
            vU2 = HFmodel_vU2.uncorrected_FS_harmonicform(real_pts)
            hvU2b = tf.einsum('x,xb->xb',tf.cast(betamodel_0m213.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vU2))

            H1=betamodel_02m20.raw_FS_HYM_r(real_pts) 
            H2=betamodel_001m3.raw_FS_HYM_r(real_pts) 
            H3=betamodel_0m213.raw_FS_HYM_r(real_pts) 

        print("Now compute the integrals")
        #(1j) for each FS form #maybe should be (1j/2)??
        #(-j)**3 to get the flat measure dx1dy1dx2dy2dx3dy3 # and this should be (-2j)???
        #(-1) to rearrange dz wedge dzbar in the v wedge hvb in canonical order.
        # whence the -1j/2???? it comes from the hodge star (the one out the front)
        #overall have added a factor of 2
        print("The field normalisations:")
        #check this is constant!
        lc_c = tf.cast(pg.lc,complex_dtype)
        HuHu=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vH[:n_p],hvHb[:n_p],lc_c,lc_c))
        print("(Hu,Hu) = " + str(HuHu))
        Q3Q3=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ3[:n_p],hvQ3b[:n_p],lc_c,lc_c))
        print("(Q3,Q3) = " + str(Q3Q3))
        U3U3=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU3[:n_p],hvU3b[:n_p],lc_c,lc_c))
        print("(U3,U3) = " + str(U3U3))
        U1U1=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU1[:n_p],hvU1b[:n_p],lc_c,lc_c))
        print("(U1,U1) = " + str(U1U1))
        U2U2=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU2[:n_p],hvU2b[:n_p],lc_c,lc_c))
        print("(U2,U2) = " + str(U2U2))
        Q1Q1=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ1[:n_p],hvQ1b[:n_p],lc_c,lc_c))
        print("(Q1,Q1) = " + str(Q1Q1))
        Q2Q2=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ2[:n_p],hvQ2b[:n_p],lc_c,lc_c))
        print("(Q2,Q2) = " + str(Q2Q2))

        print("The field mixings:")
        Q1Q2=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ1[:n_p],hvQ2b[:n_p],lc_c,lc_c))
        print("(Q1,Q2) = " + str(Q1Q2))
        Q2Q1=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vQ2[:n_p],hvQ1b[:n_p],lc_c,lc_c))
        print("(Q2,Q1) = " + str(Q2Q1))
        U1U2=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU1[:n_p],hvU2b[:n_p],lc_c,lc_c))
        print("(U1,U2) = " + str(U1U2))
        U2U1=(-1j/2)*(-1j/2)**2*(-2j)**3*(-1)*tf.reduce_mean(aux_weights[0:n_p]*tf.einsum('xab,xcd,xe,xf,acf,bde->x',mets[:n_p],mets[:n_p],vU2[:n_p],hvU1b[:n_p],lc_c,lc_c))
        print("(U2,U1) = " + str(U2U1))

        tfsqrtandcast=lambda x: tf.cast(tf.math.sqrt(x),complex_dtype)
        print("CONSIDERING A PARTICULAR ELEMENT:")
        elements_21= aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU2)*omega_normalised_to_one
        elements_12 =aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vQ2,vU3)*omega_normalised_to_one 
        print("Max absolute value of elements_21:", tf.reduce_max(tf.abs(elements_21)))
        print("Mean of elements_21:", tf.reduce_mean(elements_21))
        print("Mean of abs(elements_21):", tf.reduce_mean(tf.abs(elements_21)))
        print("Max absolute value of elements_12:", tf.reduce_max(tf.abs(elements_12)))
        print("Mean of elements_12:", tf.reduce_mean(elements_12))
        print("Mean of abs(elements_12):", tf.reduce_mean(tf.abs(elements_12)))
        
        #print("Integral of omega_normalised_to_one = ", tf.reduce_mean(aux_weights * omega_normalised_to_one* tf.math.conj(omega_normalised_to_one))) # verified that this is correct!!!yes
        #this is the holomorphic Yukawa
        print('doing einsums')
        m = [[0,0,0],[0,0,0],[0,0,0]]
        mwoH = [[0,0,0],[0,0,0],[0,0,0]]


        mwoH[0][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ1,vU1)*omega_normalised_to_one)
        mwoH[0][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ1,vU2)*omega_normalised_to_one)
        mwoH[0][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ1,vU3)*omega_normalised_to_one)
        mwoH[1][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ2,vU1)*omega_normalised_to_one)
        mwoH[1][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ2,vU2)*omega_normalised_to_one)
        mwoH[1][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ2,vU3)*omega_normalised_to_one)
        mwoH[2][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ3,vU1)*omega_normalised_to_one)
        mwoH[2][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ3,vU2)*omega_normalised_to_one)
        mwoH[2][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",lc_c,vH,vQ3,vU3)*omega_normalised_to_one)


        mwoH = (1/(8*np.sqrt(30))) *   2*np.sqrt(2)/4* np.array(mwoH)
        print("without H * 10**6")
        print(np.round(np.array(mwoH)*10**6,1))

        m[0][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU1)*omega_normalised_to_one)
        m[0][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU2)*omega_normalised_to_one)
        m[0][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vQ1,vU3)*omega_normalised_to_one)
        m[1][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU1)*omega_normalised_to_one)
        m[1][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU2)*omega_normalised_to_one)
        m[1][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vQ2,vU3)*omega_normalised_to_one)
        m[2][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU1)*omega_normalised_to_one)
        m[2][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU2)*omega_normalised_to_one)
        m[2][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H2),vH,vQ3,vU3)*omega_normalised_to_one)

        #2 sqrt2 comes out, 4 is |G|
        # 8 Sqrt30 is the group theoretic factor
        m= (1/(8*np.sqrt(30))) *   2*np.sqrt(2)/4* np.array(m)
        holomorphic_Yukawas.append(m)
        print('proper calculation*10**6')
        print(np.round(np.array(m)*10**6,1))
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
        print("physical_yukawas*10**6")
        print(np.round(physical_yukawas*10**6,1))
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
        print(s)
        masses_ref_and_learned.append(s)
    # Save training results to CSV files
    import numpy as np
    import os
    import time
    from datetime import datetime

    # Create unique filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{os.getpid()}"  # Add process ID for uniqueness
    
    # Save results for this run to unique npz file
    results = {
        # Network parameters
        'free_coefficient': free_coefficient,
        'epochs_phi': nEpochsPhi,
        'epochs_beta': nEpochsBeta, 
        'epochs_sigma': nEpochsSigma,
        'epochs_sigma2': nEpochsSigma2,
        'width_phi': widthPhi,
        'width_beta': widthBeta,
        'width_sigma': widthSigma, 
        'width_sigma2': widthSigma2,
        'depth_phi': depthPhi,
        'depth_beta': depthBeta,
        'depth_sigma': depthSigma,
        'depth_sigma2': depthSigma2,
        'points_training': nPoints,
        'points_harmonic_forms': nPointsHF,
        'points_integration': n_to_integrate,
        'measure_phi': measure_phi,
        'measure_LBs': [measure_LB1, measure_LB2, measure_LB3],
        'measure_HFs': [measure_HF1, measure_HF2, measure_HF3, measure_HF4, measure_HF5, measure_HF6, measure_HF7],
        
        # Results matrices
        'physical_yukawas': np.array(mats),
        'singular_values': np.array(masses_ref_and_learned),
        'holomorphic_yukawas': np.array(holomorphic_Yukawas)
    }

    # Save to unique file in results directory
    os.makedirs(name_of_run + '_results', exist_ok=True)
    np.savez(f'{name_of_run}_results/run_{run_id}.npz', **results)
    # Save masses to CSV
    import csv
    csv_file = f'{name_of_run}_results/masses.csv'
    
    # Create header if file doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run_id', 'learned_mass1', 'learned_mass2', 'learned_mass3', 'ref_mass1', 'ref_mass2', 'ref_mass3'])
    
    # Append masses for this run
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run_id] + list(masses_ref_and_learned[0]) + list(masses_ref_and_learned[1]))
