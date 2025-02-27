#if __name__ ==  '__main__':
#    from multiprocessing import set_start_method
#    set_start_method('spawn')
#import multiprocessing
import time
import os
start_time_of_process = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
if 'profile' in sys.argv[1:]:
    os.environ["USE_PROFILER"] = "1"
else:
    os.environ["USE_PROFILER"] = "0"
if __name__ == '__main__':
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


import csv
import numpy as np
import gc
import os
import re
import logging
import pickle
#sys.path.append("/Users/kit/Documents/Phys_Working/MF metric")

logging.basicConfig(stream=sys.stdout)

from cymetric.pointgen.pointgen import PointGenerator
from cymetric.pointgen.nphelper import prepare_dataset, prepare_basis_pickle

import tensorflow as tf
import tensorflow.keras as tfk

if __name__ == '__main__':
    # Default to not eager unless explicitly requested
    run_eagerly = False
    if len(sys.argv) > 5 and 'eager' in sys.argv:
        run_eagerly = True

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

foldername = "model13"
linebundleforHYM_LB1=np.array([0,2,-2,0]) 
linebundleforHYM_LB2=np.array([0,0,1,-3]) 
linebundleforHYM_LB3=np.array([0,-2,1,3]) 
ambientTQ = np.array([1,1,1,1])
monomialsTQ = np.array([[2, 0, 2, 0, 2, 0, 2, 0], [2, 0, 2, 0, 2, 0, 1, 1], [2, 0, 2, 0, 2, 
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

kmoduliTQ = np.array([1,(np.sqrt(7)-2)/3,(np.sqrt(7)-2)/3,1])
# kmoduliTQ = np.array([1,1,1,1])

from OneAndTwoFormsForLineBundlesModel13 import *
#from generate_and_train_all_nns import *
from generate_and_train_all_nnsHOLOModel13 import *
from auxiliary_funcs import *

print_memory_usage(start_time_of_process=start_time_of_process)

if __name__ == '__main__':
    #print('fixed directory')

    # nPoints=300000
    # free_coefficient=1.# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
    # nEpochsPhi=100
    # nEpochsBeta=60
    # nEpochsSigma=50


    nPoints=300000
    nPointsHF=300000
    #nPoints=10000
    #nPointsHF=10000
    #nPoints=1000
    #nPointsHF=1000
    #nPoints=10000
    #nPointsHF=10000

    tr_batchsize=64
    #free_coefficient=1.000000004# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
    import sys
    free_coefficient = float(sys.argv[1])
    seed_for_gen=int((int(free_coefficient*100000000000)+free_coefficient*1000000))%4294967294 # modulo largest seed
    print("seed for gen", seed_for_gen)
    #free_coefficient=1.# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
    #nEpochsPhi=100
    #nEpochsPhi=10
    #nEpochsBeta=10
    #nEpochsSigma=40#changed below vH


    nEpochsPhi=30
    nEpochsBeta=50
    nEpochsSigma=60#changed below vH
    nEpochsSigma2=80#changed below vH


    depthPhi=4
    widthPhi=196#128 4 in the 1.0s
    depthBeta=4
    widthBeta=196
    depthSigma=4
    widthSigma=130 # up from 256
    depthSigma2=4
    widthSigma2=130 # up from 256


    #depthPhi=4
    #widthPhi=128#128 4 in the 1.0s
    #depthBeta=3
    #widthBeta=128
    #depthSigma=3+2
    #widthSigma=128//2 # up from 256
    #depthSigma2=3+2
    #widthSigma2=196//2 # up from 256

    #nEpochsPhi=1
    #nEpochsBeta=1
    #nEpochsSigma=1#changed below vH
    #nEpochsSigma2=1#changed below vH
    #
    #
    #depthPhi=4
    #widthPhi=128#128 4 in the 1.0s
    #depthBeta=4
    #widthBeta=128
    #depthSigma=4
    #widthSigma=128 # up from 256
    #depthSigma2=4
    #widthSigma2=128 # up from 256



    alphabeta=[1,10]
    alphasigma1=[1,10] # down form 50
    alphasigma2=[1,80] # down from 200


    #lRatePhi= 0.0000005
    #lRateBeta=0.0000005
    #lRateSigma=0.0000005# perhaps best as 0.03

    lRatePhi= 0.005
    lRateBeta=0.005
    lRateSigma=0.005# perhaps best as 0.03
    lRateSigma=0.005//10# perhaps best as 0.03
    lRateSigma2=0.003//10# perhaps best as 0.03

    lRateSigma=0.001# perhaps best as 0.03
    lRateSigma=0.001#//10# perhaps best as 0.03
    lRateSigma2=0.001#//10# perhaps best as 0.03

    #lRatePhi= 0.00
    #lRateBeta=0.00
    #lRateSigma=0.0
    #lRateSigma2=0.0




    print("Learning rate: phi:",lRatePhi)
    print("Learning rate: beta:",lRateBeta)
    print("Learning rate: sigma:",lRateSigma)
    print("Learning rate decays by a factor, typically 10, over the course of learning")



    # Use command-line arguments if provided, starting from the second argument
    start_from = sys.argv[2] if len(sys.argv) > 2 else 'phi'
    training_flags = determine_training_flags(start_from)

    # Unpack training flags
    train_phi, train_LB1, train_LB2, train_LB3, train_vH, train_vQ3, train_vU3, train_vQ1, train_vQ2, train_vU1, train_vU2 = (
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


    if len(sys.argv) > 4 and str(sys.argv[4]) == 'skipall':
        print("Requested to skip all measures")
        skip_measuresPhi=True
        skip_measuresBeta=True
        skip_measuresHF=True
    elif len(sys.argv) > 4 and str(sys.argv[4]) == 'skipnone':
        skip_measuresPhi=False
        skip_measuresBeta=False
        skip_measuresHF=False
        print("Requested to skip none of the measures")
    else:
        print("Requested to skip some of the measures")
        skip_measuresPhi=True
        skip_measuresBeta=True
        skip_measuresHF=False

    print(f"Skipping measures? phi? {skip_measuresPhi}, beta? {skip_measuresBeta}, HF? {skip_measuresHF}")

        #skip_measuresHF

    force_generate_phi=False
    force_generate_HYM=False
    force_generate_HF=False
    force_generate_HF_2=False
    force_generate_eval=False

    return_zero_phi= True
    return_zero_HYM = True
    return_zero_HF = False
    return_zero_HF_2 = False

    SecondBSize=50000
    n_to_integrate=1000000
    #n_to_integrate=100000

    use_zero_network_phi = True


    print("sys.argv: ", sys.argv)
    if len(sys.argv) > 3 and str(sys.argv[3]) in ['test','testmid','testsmall', 'alltiny']:
        # Override with small test values
        if str(sys.argv[3]) == 'alltiny':
            nPoints = 30
            nPointsHF = 30
            n_to_integrate = 30
            skip_measuresPhi=True
            skip_measuresBeta=True
            skip_measuresHF=True
        elif str(sys.argv[3]) == 'testsmall':
            nPoints = 100
            nPointsHF = 100
            n_to_integrate = 100
        elif sys.argv[3] == 'testmid':
            nPoints = 100000
            nPointsHF = 100000
            n_to_integrate = 1000000
        else:
            nPoints = 300000
            nPointsHF = 300000
            n_to_integrate = 1000000
        #tr_batchsize = 10
        #SecondBSize = 10
        nEpochsPhi = 1
        nEpochsBeta = 1
        nEpochsSigma = 1
        nEpochsSigma2 = 1

        depthPhi = 2
        widthPhi = 10
        depthBeta = 2
        widthBeta = 10
        depthSigma = 2
        widthSigma = 10
        depthSigma2 = 2
        widthSigma2 = 10

        return_random_phi = False
        return_random_HYM = False
        return_random_HF = True
        return_random_HF_2 = True

        if len(sys.argv) > 4 and str(sys.argv[4]) == 'actual':
            nEpochsPhi = 1
            nEpochsBeta = 1
            if sys.argv[3] == 'testmid':
                nEpochsSigma = 2
                nEpochsSigma2 = 2
            else:
                nEpochsSigma = 5
                nEpochsSigma2 = 5
            depthPhi = 2
            depthBeta = 2
            depthSigma = 3
            depthSigma2 = 3
            widthPhi = 10
            widthBeta = 10
            widthSigma = 100
            widthSigma2 = 100
            return_random_phi = False
            return_random_HYM = False
            return_random_HF = False
            return_random_HF_2 = False   

        if 'largenetworks' in sys.argv[1:]:
            depthPhi = 4
            widthPhi = 100
            depthBeta = 4
            widthBeta = 100
            depthSigma = 4
            widthSigma = 100

    print("Number of points: " + str(nPoints), "Number of points HF: " + str(nPointsHF), "Number of points to integrate: " + str(n_to_integrate))
    print(f"shapes, phi: {depthPhi}x{widthPhi}, beta: {depthBeta}x{widthBeta}, HF: {depthSigma}x{widthSigma}, HF2: {depthSigma2}x{widthSigma2}")


    

if __name__ ==  '__main__':
    unique_id_or_coeff = free_coefficient
    manifold_name_and_data = (get_coefficients_m13(free_coefficient), kmoduliTQ, ambientTQ, monomialsTQ, foldername, unique_id_or_coeff)
    
    generate_points_and_save_using_defaults(manifold_name_and_data,nPoints,seed_set=seed_for_gen)
    if train_phi:
        #phimodel,training_history=train_and_save_nn(manifold_name_and_data,depthPhi,widthPhi,nEpochsPhi,bSizes=[64,tr_batchsize],lRate=lRatePhi) 
        phimodel,training_history, measure_phi=train_and_save_nn(manifold_name_and_data,depthPhi,widthPhi,nEpochsPhi,stddev=0.05,bSizes=[64,SecondBSize],lRate=lRatePhi,use_zero_network=use_zero_network_phi)
    else:
        phimodel,training_history, measure_phi=load_nn_phimodel(manifold_name_and_data,depthPhi,widthPhi,nEpochsPhi,[64,SecondBSize],set_weights_to_zero=return_zero_phi,skip_measures=skip_measuresPhi,set_weights_to_random=return_random_phi)


    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()


    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB1,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB1:
        betamodel_LB1,training_historyBeta_LB1, measure_LB1 = train_and_save_nn_HYM(manifold_name_and_data,linebundleforHYM_LB1,depthBeta,widthBeta,nEpochsBeta,stddev=0.05,bSizes=[64],lRate=lRateBeta,alpha=alphabeta,load_network=False)
    else: 
        betamodel_LB1,training_historyBeta_LB1, measure_LB1=load_nn_HYM(manifold_name_and_data,linebundleforHYM_LB1,depthBeta,widthBeta,nEpochsBeta,[64],set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM)
    
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])


    #generate_points_and_save_using_defaultsHYM(manifold_name_and_data,-linebundleforHYM_LB1,nPoints,phimodel,force_generate=True)
    #mbetamodel_LB1,mtraining_historyBeta_LB1, measure = train_and_save_nn_HYM(manifold_name_and_data,-linebundleforHYM_LB1,depthBeta,widthBeta,nEpochsBeta,[64],0.001,alpha=alphabeta)
    #mbetamodel_LB1,mtraining_historymBeta_LB1=load_nn_HYM(manifold_name_and_data,-linebundleforHYM_LB1,depthBeta,widthBeta,nEpochsBeta,[64],set_weights_to_zero=False)
    
    
    
    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB2,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB2:
        betamodel_LB2,training_historyBeta_LB2, measure_LB2 = train_and_save_nn_HYM(manifold_name_and_data,linebundleforHYM_LB2,depthBeta,widthBeta,nEpochsBeta,stddev=0.05,bSizes=[64],lRate=lRateBeta,alpha=alphabeta,load_network=False)
    else:
        betamodel_LB2,training_historyBeta_LB2, measure_LB2=load_nn_HYM(manifold_name_and_data,linebundleforHYM_LB2,depthBeta,widthBeta,nEpochsBeta,[64],set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM)
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB3,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB3:
        betamodel_LB3,training_historyBeta_LB3, measure_LB3 = train_and_save_nn_HYM(manifold_name_and_data,linebundleforHYM_LB3,depthBeta,widthBeta,nEpochsBeta,stddev=0.05,bSizes=[64],lRate=lRateBeta,alpha=alphabeta,load_network=False)
    else:
        betamodel_LB3,training_historyBeta_LB3, measure_LB3=load_nn_HYM(manifold_name_and_data,linebundleforHYM_LB3,depthBeta,widthBeta,nEpochsBeta,[64],set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM)

    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    phi=phimodel

    gc.collect()
    tf.keras.backend.clear_session()

    ##tf.profiler.experimental.start('/mnt/extraspace/kitft/tf_profile_holo_'+str(manifold_name_and_data))
    #tf.profiler.experimental.start('/mnt/extraspace/kitft/tf_profile_holo_'+str(manifold_name_and_data), options=tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,python_tracer_level=1, device_tracer_level=1))

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB1,functionforbaseharmonicform_jbar_for_vH,phi,betamodel_LB1,nPointsHF,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vH:
        #HFmodel_vH,trainingHistoryHF_vH, measure = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB1,betamodel_LB1,functionforbaseharmonicform_jbar_for_vH,depthSigma+2,widthSigma//2,nEpochsSigma,[64],lRate=lRateSigma/10,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
        HFmodel_vH,trainingHistoryHF_vH, measure_HF1 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB1,betamodel_LB1,phimodel,functionforbaseharmonicform_jbar_for_vH,depthSigma,widthSigma,nEpochsSigma,[64],lRate=lRateSigma,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
        #was /100
    else:
        HFmodel_vH,trainingHistoryHF_vH, measure_HF1=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB1,betamodel_LB1,phimodel,functionforbaseharmonicform_jbar_for_vH,depthSigma,widthSigma,nEpochsSigma,[64],alpha=alphasigma1,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)


    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    #nEpochsSigma=50#changed below vH
    #nEpochsSigma=10

    # Stop the profiler
    #print("stopping profiler")
    #tf.profiler.experimental.stop()
    #print('next')



    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB2,functionforbaseharmonicform_jbar_for_vQ3,phi,betamodel_LB2,nPointsHF,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vQ3:
        HFmodel_vQ3,trainingHistoryHF_vQ3, measure_HF2 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vQ3,depthSigma,widthSigma,nEpochsSigma,[64],lRate=lRateSigma,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
    else:
        HFmodel_vQ3,trainingHistoryHF_vQ3, measure_HF2=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vQ3,depthSigma,widthSigma,nEpochsSigma,[64],alpha=alphasigma1,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    #nEpochsSigma=10
    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB2,functionforbaseharmonicform_jbar_for_vU3,phi,betamodel_LB2,nPointsHF,force_generate=force_generate_HF,seed_set=seed_for_gen)

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    if train_vU3:
        HFmodel_vU3,trainingHistoryHF_vU3, measure_HF3 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vU3,depthSigma,widthSigma,nEpochsSigma,[64],lRate=lRateSigma,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
    else:
        HFmodel_vU3,trainingHistoryHF_vU3, measure_HF3=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vU3,depthSigma,widthSigma,nEpochsSigma,[64],alpha=alphasigma1,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()






    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vQ1,phi,betamodel_LB3,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ1:
        HFmodel_vQ1,trainingHistoryHF_vQ1, measure_HF4 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ1,depthSigma2,widthSigma2,nEpochsSigma2,[64],lRate=lRateSigma2,alpha=alphasigma2,stddev=stddev_Q12U12,final_layer_scale=final_layer_scale_Q12U12)
    else:
        HFmodel_vQ1,trainingHistoryHF_vQ1, measure_HF4=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ1,depthSigma2,widthSigma2,nEpochsSigma2,[64],alpha=alphasigma2,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vQ2,phi,betamodel_LB3,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ2:   
        HFmodel_vQ2,trainingHistoryHF_vQ2, measure_HF5 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ2,depthSigma2,widthSigma2,nEpochsSigma2,[64],lRate=lRateSigma2,alpha=alphasigma2,stddev=stddev_Q12U12,final_layer_scale=final_layer_scale_Q12U12)
    else:
        HFmodel_vQ2,trainingHistoryHF_vQ2, measure_HF5=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ2,depthSigma2,widthSigma2,nEpochsSigma2,[64],alpha=alphasigma2,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vU1,phi,betamodel_LB3,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU1:
        HFmodel_vU1,trainingHistoryHF_vU1, measure_HF6 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU1,depthSigma2,widthSigma2,nEpochsSigma2,[64],lRate=lRateSigma2,alpha=alphasigma2,stddev=stddev_Q12U12,final_layer_scale=final_layer_scale_Q12U12)
    else:
        HFmodel_vU1,trainingHistoryHF_vU1, measure_HF6=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU1,depthSigma2,widthSigma2,nEpochsSigma2,[64],alpha=alphasigma2,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vU2,phi,betamodel_LB3,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU2:
        HFmodel_vU2,trainingHistoryHF_vU2, measure_HF7 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU2,depthSigma2,widthSigma2,nEpochsSigma2,[64],lRate=lRateSigma2,alpha=alphasigma2,stddev=stddev_Q12U12,final_layer_scale=final_layer_scale_Q12U12)
    else:
        HFmodel_vU2,trainingHistoryHF_vU2, measure_HF7=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU2,depthSigma2,widthSigma2,nEpochsSigma2,[64],alpha=alphasigma2,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()
    
    
    
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])


    pg,kmoduli=generate_points_and_save_using_defaults_for_eval(manifold_name_and_data,n_to_integrate,seed_set=seed_for_gen,force_generate=force_generate_eval)
    dataEval=np.load(os.path.join(foldername,'tetraquadric_pg_for_eval_with_'+str(unique_id_or_coeff), 'dataset.npz'))
    n_p = n_to_integrate 
    points64=dataEval['X_train'][0:n_p]
    real_pts=tf.cast(dataEval['X_train'][0:n_p],real_dtype)
    pointsComplex=tf.cast(point_vec_to_complex(points64),complex_dtype)# we have to cast to complex64 from complex128 sometimes. Just for safety. point_vec_to_complex does the obvious thing

    print("\n do analysis: \n\n\n")

    Vol_from_dijk_J=pg.get_volume_from_intersections(kmoduli)
    Vol_reference_dijk_with_k_is_1=pg.get_volume_from_intersections(np.ones_like(kmoduli)) 
    #volCY_from_Om=tf.reduce_mean(dataEval['y_train'][:n_p,0])
    print("Compute holomorphic Yukawas")
    volCY_from_Om=tf.reduce_mean(dataEval['y_train'][:n_p,0])/6 #this is the actual volume of the CY computed from Omega.since J^J^J^ = K Om^Ombar?? not totally sure here:
    #consider omega normalisation
    omega = tf.cast(batch_process_helper_func(pg.holomorphic_volume_form, [pointsComplex], batch_indices=[0], batch_size=100000),complex_dtype)
    # Verify that dataEval['y_train'][:n_p,1] equals |omega|^2
    omega_abs_squared = tf.math.real(omega * tf.math.conj(omega))
    assert tf.reduce_all(tf.abs(omega_abs_squared[:100] - dataEval['y_train'][:100,1]) < 1e-5), "First elements of dataEval['y_train'][:,1] should equal |omega|^2"
    #put the omega here, not the omegabar
    omega_normalised_to_one=omega/tf.cast(np.sqrt(volCY_from_Om),complex_dtype) # this is the omega that's normalised to 1. VERIFIED yes.

    # this is integral omega wedge omegabar? yeah.
    # this is integral omega wedge omegabar? yeah.
    weights=dataEval['y_train'][:n_p,0]
    omegasquared=dataEval['y_train'][:n_p,1]

    weights_c = tf.cast(weights, complex_dtype)
    omegasquared_c = tf.cast(omegasquared, complex_dtype)
    aux_weights=tf.cast((weights/(omegasquared))*(1/(6)),real_dtype)### these are the appropriate 'flat measure' so sum_i aux_weights f is int_X f(x)
    aux_weights_c = tf.cast(aux_weights, complex_dtype)
    
    #convertcomptoreal=lambda complexvec: tf.concat([tf.math.real(complexvec),tf.math.imag(complexvec)],-1) # this converts from complex to real

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    batch_size_for_processing=min(100000, len(real_pts))
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
            # Batch process corrected harmonic forms
            vH = batch_process_helper_func(HFmodel_vH.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvHb = tf.einsum('x,xb->xb', tf.cast(betamodel_LB1(real_pts), complex_dtype), tf.math.conj(vH))
            print('got vH', flush=True)
            
            vQ3 = batch_process_helper_func(HFmodel_vQ3.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvQ3b = tf.einsum('x,xb->xb', tf.cast(betamodel_LB2(real_pts), complex_dtype), tf.math.conj(vQ3))
            print('got vQ3', flush=True)
            
            vU3 = batch_process_helper_func(HFmodel_vU3.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvU3b = tf.einsum('x,xb->xb', tf.cast(betamodel_LB2(real_pts), complex_dtype), tf.math.conj(vU3))
            print('got vU3', flush=True)
            
            vQ1 = batch_process_helper_func(HFmodel_vQ1.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvQ1b = tf.einsum('x,xb->xb', tf.cast(betamodel_LB3(real_pts), complex_dtype), tf.math.conj(vQ1))
            print('got vQ1', flush=True)
            
            vQ2 = batch_process_helper_func(HFmodel_vQ2.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvQ2b = tf.einsum('x,xb->xb', tf.cast(betamodel_LB3(real_pts), complex_dtype), tf.math.conj(vQ2))
            print('got vQ2', flush=True)
            
            vU1 = batch_process_helper_func(HFmodel_vU1.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvU1b = tf.einsum('x,xb->xb', tf.cast(betamodel_LB3(real_pts), complex_dtype), tf.math.conj(vU1))
            print('got vU1', flush=True)
            
            vU2 = batch_process_helper_func(HFmodel_vU2.corrected_harmonicform, (real_pts,), batch_indices=(0,), batch_size=batch_size_for_processing, compile_func=True)
            hvU2b = tf.einsum('x,xb->xb', tf.cast(betamodel_LB3(real_pts), complex_dtype), tf.math.conj(vU2))
            print('got vU2', flush=True)
            H1=betamodel_LB1(real_pts) 
            H2=betamodel_LB2(real_pts) 
            H3=betamodel_LB3(real_pts) 
        elif not use_trained:
            mets = phi.fubini_study_pb(real_pts,ts=tf.cast(kmoduli,complex_dtype))
            dets = tf.linalg.det(mets)
            vH =  HFmodel_vH.uncorrected_FS_harmonicform(real_pts)
            hvHb =  tf.einsum('x,xb->xb',tf.cast(betamodel_LB1.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vH))
            vQ3 = HFmodel_vQ3.uncorrected_FS_harmonicform(real_pts)
            hvQ3b = tf.einsum('x,xb->xb',tf.cast(betamodel_LB2.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vQ3))
            vU3 = HFmodel_vU3.uncorrected_FS_harmonicform(real_pts)
            hvU3b = tf.einsum('x,xb->xb',tf.cast(betamodel_LB2.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vU3))

            vQ1 = HFmodel_vQ1.uncorrected_FS_harmonicform(real_pts)
            hvQ1b = tf.einsum('x,xb->xb',tf.cast(betamodel_LB3.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vQ1))
            vQ2 = HFmodel_vQ2.uncorrected_FS_harmonicform(real_pts)
            hvQ2b = tf.einsum('x,xb->xb',tf.cast(betamodel_LB3.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vQ2))

            vU1 = HFmodel_vU1.uncorrected_FS_harmonicform(real_pts)
            hvU1b = tf.einsum('x,xb->xb',tf.cast(betamodel_LB3.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vU1))
            vU2 = HFmodel_vU2.uncorrected_FS_harmonicform(real_pts)
            hvU2b = tf.einsum('x,xb->xb',tf.cast(betamodel_LB3.raw_FS_HYM_r(real_pts),complex_dtype),tf.math.conj(vU2))

            H1=betamodel_LB1.raw_FS_HYM_r(real_pts) 
            H2=betamodel_LB2.raw_FS_HYM_r(real_pts) 
            H3=betamodel_LB3.raw_FS_HYM_r(real_pts) 

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
        print("Mean of elements_21:", tf.reduce_mean(elements_21))
        print("Mean of abs(elements_21):", tf.reduce_mean(tf.abs(elements_21)))
        print("Max absolute value of elements_12:", tf.reduce_max(tf.abs(elements_12)))
        print("Mean of elements_12:", tf.reduce_mean(elements_12))
        print("Mean of abs(elements_12):", tf.reduce_mean(tf.abs(elements_12)))

        print("--------CONSIDERING THE INVERSION PARTICULAR ELEMENT:")
        elements_Q3U2= factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H2*H3),vH,vQ3,vQ2)*omega_normalised_to_one
        elements_U2U3 =factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H2),vH,vU2,vU3)*omega_normalised_to_one 
        print("Max absolute value of elements_Q3U2:", tf.reduce_max(tf.abs(elements_21)))
        print("Mean of elements_Q3U2:", tf.reduce_mean(elements_Q3U2))
        print("Mean of abs(elements_Q3U2):", tf.reduce_mean(tf.abs(elements_Q3U2)))
        print("Max absolute value of elements_U2U3:", tf.reduce_max(tf.abs(elements_U2U3)))
        print("Mean of elements_U2U3:", tf.reduce_mean(elements_U2U3))
        print("Mean of abs(elements_U2U3):", tf.reduce_mean(tf.abs(elements_U2U3)))
        print("--------------------------------")
        
        
        
        print("--------CONSIDERING THE (0,0) ELEMENT:")
        elements_00 = factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU1)*omega_normalised_to_one
        print("Max absolute value of elements_00:", tf.reduce_max(tf.abs(elements_00)))
        print("Mean of elements_00:", tf.reduce_mean(elements_00))
        print("Mean of abs(elements_00):", tf.reduce_mean(tf.abs(elements_00)))
        print("--------------------------------")

  
        print("--------CONSIDERING THE (1,1) ELEMENT:")
        elements_11 = factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU2)*omega_normalised_to_one
        print("Max absolute value of elements_11:", tf.reduce_max(tf.abs(elements_11)))     
        print("Mean of elements_11:", tf.reduce_mean(elements_11))
        print("Mean of abs(elements_11):", tf.reduce_mean(tf.abs(elements_11)))
        print("--------------------------------")


        print("--------CONSIDERING THE (2,2) ELEMENT:")
        elements_22 = factor * aux_weights_c * tf.einsum("abc,x,xa,xb,xc->x",lc_c,tfsqrtandcast(H1*H3*H3),vH,vQ3,vU3)*omega_normalised_to_one
        print("Max absolute value of elements_22:", tf.reduce_max(tf.abs(elements_22)))     
        print("Mean of elements_22:", tf.reduce_mean(elements_22))
        print("Mean of abs(elements_22):", tf.reduce_mean(tf.abs(elements_22)))
        print("--------------------------------")
        #print("Integral of omega_normalised_to_one = ", tf.reduce_mean(aux_weights_c * omega_normalised_to_one* tf.math.conj(omega_normalised_to_one))) # verified that this is correct!!!yes

        # #volCY=tf.reduce_mean(omega)
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
                print("Effective sample size for this element {i}{j}: " + str(m_eff_n))
                print("Effective sample size for this element {i}{j} without H: " + str(m_eff_nwoH))
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
        print("neffs without H")
        print(np.round(mwoH_neffs,1))

        print('proper calculation*10**6')
        print(np.round(np.array(m)*10**6,1))
        print("neffs")
        print(np.round(m_neffs,1))
        
        # Use the new function to calculate physical Yukawas and their errors
        physical_yukawas, physical_yukawas_errors = propagate_errors_to_physical_yukawas(
            NormH, NormH_errors, NormQ, NormQ_errors, NormU, NormU_errors, m, m_errors
        )
        
        # Print holomorphic Yukawa matrix errors
        print("holomorphic Yukawa errors *10**6 (absolute value)")
        print(np.round(np.abs(m_errors)*10**6,1))
        
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
    import numpy as np
    import os
    import time
    from datetime import datetime

    # Create unique filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{os.getpid()}"  # Add process ID for uniqueness
    
    # Calculate average real and imaginary accuracy metrics
    avg_rel_real_error = np.mean(np.abs(np.real(physical_yukawas_errors)) / np.abs(physical_yukawas))
    avg_rel_imag_error = np.mean(np.abs(np.imag(physical_yukawas_errors)) / np.abs(physical_yukawas))
    
    # Calculate average real and imaginary accuracy metrics for holomorphic yukawas
    avg_rel_real_error_holo = np.mean(np.abs(np.real(m_errors)) / np.abs(m))
    avg_rel_imag_error_holo = np.mean(np.abs(np.imag(m_errors)) / np.abs(m))
    
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
        'singular_values': np.array(masses_trained_and_ref),
        'holomorphic_yukawas': np.array(holomorphic_Yukawas_trained_and_ref),
        'holomorphic_yukawas_errors': m_errors,
        
        # Statistical error information
        'integral_stats': integral_stats,
        'matrix_stats': matrix_stats,
        'physical_yukawas_errors': physical_yukawas_errors,
        'singular_values_errors': singular_value_errors,
        'avg_rel_real_error': avg_rel_real_error,
        'avg_rel_imag_error': avg_rel_imag_error,
        'avg_rel_real_error_holo': avg_rel_real_error_holo,
        'avg_rel_imag_error_holo': avg_rel_imag_error_holo,
        'm_neffs': m_neffs,
        'mwoH_neffs': mwoH_neffs,
        'Qneffs': Qneffs,
        'Uneffs': Uneffs,
        'Hneffs': Hneffs
    }

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
                             'coefficient', 'n_to_integrate', 
                             'real_accuracy', 'imag_accuracy',
                             'holo_real_accuracy', 'holo_imag_accuracy'])
    
    # Append masses for this run
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run_id] + 
                        list(masses_trained_and_ref[0]) + 
                        list(singular_value_errors) +
                        list(masses_trained_and_ref[1]) + 
                        list(singular_value_errors) +  # Using same error estimate for both learned and reference
                        [free_coefficient] + 
                        [n_to_integrate] + 
                        [avg_rel_real_error, avg_rel_imag_error] +
                        [avg_rel_real_error_holo, avg_rel_imag_error_holo])

