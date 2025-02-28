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
from final_integration import *
from yukawas.generate_and_train_all_nnsHOLO_all import *
from auxiliary_funcs import *

foldername = "datamodel13"
get_coefficients_here = get_coefficients_m13# vs get_coefficients_m1
from OneAndTwoFormsForLineBundlesModel13 import *
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

    use_zero_network_phi = False # what does this do?
    use_zero_network_beta = False

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

    print(f"Skipping measures? phi? {skip_measuresPhi}, beta? {skip_measuresBeta}, HF? {skip_measuresHF}")
    print("Number of points: " + str(nPoints), "Number of points HF: " + str(nPointsHF), "Number of points to integrate: " + str(n_to_integrate))
    print(f"shapes, phi: {depthPhi}x{widthPhi}, beta: {depthBeta}x{widthBeta}, HF: {depthSigma}x{widthSigma}, HF2: {depthSigma2}x{widthSigma2}")

   
    return_random_phi = False
    return_random_HYM = False
    return_random_HF = True
    return_random_HF_2 = True
    phi_model_load_function = None 
    beta_model_load_function = None
    sigma_model_load_function = None
    sigma2_model_load_function = None










    stddev_phi = 0.05
    stddev_beta = 0.05
    stddev_sigma = stddev_H33
    stddev_sigma2 = stddev_Q12U12
    norm_momentum_sigma = 0.999
    norm_momentum_sigma2 = 0.999

    activationphi = None
    activationbeta = None
    activationsigma = None
    activationsigma2 = None

    phimodel_config = {'depth': depthPhi, 'width': widthPhi, 'nEpochs': nEpochsPhi, 'lRate': lRatePhi, 'stddev': stddev_phi, 'bSizes': [tr_batchsize,SecondBSize], 'network_function': phi_model_load_function, 'activation': activationphi}
    betamodel_config = {'depth': depthBeta, 'width': widthBeta, 'nEpochs': nEpochsBeta, 'lRate': lRateBeta, 'alpha': alphabeta, 'stddev': stddev_beta, 'bSizes': [tr_batchsize], 'network_function': beta_model_load_function, 'activation':activationbeta}
    sigmamodel_config = {'depth': depthSigma, 'width': widthSigma, 'nEpochs': nEpochsSigma, 'lRate': lRateSigma, 'alpha': alphasigma1, 'stddev': stddev_sigma, 'bSizes': [tr_batchsize], 'final_layer_scale': final_layer_scale_H33, 'norm_momentum': norm_momentum_sigma, 'network_function': sigma_model_load_function,'activation':activationsigma}
    sigma2model_config = {'depth': depthSigma2, 'width': widthSigma2, 'nEpochs': nEpochsSigma2, 'lRate': lRateSigma2, 'alpha': alphasigma2, 'stddev': stddev_Q12U12, 'bSizes': [tr_batchsize], 'final_layer_scale': final_layer_scale_Q12U12, 'norm_momentum': norm_momentum_sigma2, 'network_function': sigma2_model_load_function,'activation':activationsigma2}
    print("phimodel_config: ", phimodel_config)
    print("betamodel_config: ", betamodel_config)
    print("sigmamodel_config: ", sigmamodel_config)
    print("sigma2model_config: ", sigma2model_config)
    

    print("Name of invoking script: ", sys.argv[0], "namespace of vH: ", functionforbaseharmonicform_jbar_for_vH.__module__)
    if '13' in sys.argv[0]:
        if get_coefficients_here != get_coefficients_m13 or functionforbaseharmonicform_jbar_for_vH.__module__ not in ['yukawas.OneAndTwoFormsForLineBundlesModel13','OneAndTwoFormsForLineBundlesModel13']:
            raise ValueError("invalid configuration for m13: ", get_coefficients_here, functionforbaseharmonicform_jbar_for_vH.__module__)
    elif '1' in sys.argv[0]:
        if get_coefficients_here != get_coefficients_m1 or  functionforbaseharmonicform_jbar_for_vH.__module__ not in ['yukawas.OneAndTwoFormsForLineBundlesModel1','OneAndTwoFormsForLineBundlesModel1']:
            raise ValueError("invalid configuration for m1: ", get_coefficients_here, functionforbaseharmonicform_jbar_for_vH.__module__)
    else:
        raise ValueError("Invalid model specified")




def purge_dicts_and_mem():
    delete_all_dicts_except('dataEval','manifold_name_and_data', 'phimodel_config', 'betamodel_config', 'sigmamodel_config', 'sigma2model_config')
    gc.collect()
    tf.keras.backend.clear_session()
 
do_extra_stuff_for_integration = False

if __name__ ==  '__main__':
    free_coefficient = float(sys.argv[1])
    seed_for_gen=int((int(free_coefficient*100000000000)+free_coefficient*1000000))%4294967294 # modulo largest seed
    print("seed for gen", seed_for_gen)

    unique_id_or_coeff = free_coefficient
    coefficientsTQ = get_coefficients_here(free_coefficient)
    manifold_name_and_data = (coefficientsTQ, kmoduliTQ, ambientTQ, monomialsTQ, foldername, unique_id_or_coeff)
    
    
    generate_points_and_save_using_defaults(manifold_name_and_data,nPoints,seed_set=seed_for_gen)
    if train_phi:
        #phimodel,training_history=train_and_save_nn(manifold_name_and_data,depthPhi,widthPhi,nEpochsPhi,bSizes=[64,tr_batchsize],lRate=lRatePhi) 
        phimodel,training_history, measure_phi=train_and_save_nn(manifold_name_and_data,phimodel_config,use_zero_network=use_zero_network_phi)
    else:
        phimodel,training_history, measure_phi=load_nn_phimodel(manifold_name_and_data,phimodel_config,set_weights_to_zero=return_zero_phi,skip_measures=skip_measuresPhi,set_weights_to_random=return_random_phi)


    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB1,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB1:
        betamodel_LB1,training_historyBeta_LB1, measure_LB1 = train_and_save_nn_HYM(manifold_name_and_data,linebundleforHYM_LB1,betamodel_config,load_network=False,use_zero_network=use_zero_network_beta)
    else: 
        betamodel_LB1,training_historyBeta_LB1, measure_LB1=load_nn_HYM(manifold_name_and_data,linebundleforHYM_LB1,betamodel_config,set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM)
    
    purge_dicts_and_mem()
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])


    #generate_points_and_save_using_defaultsHYM(manifold_name_and_data,-linebundleforHYM_LB1,nPoints,phimodel,force_generate=True)
    #mbetamodel_LB1,mtraining_historyBeta_LB1, measure = train_and_save_nn_HYM(manifold_name_and_data,-linebundleforHYM_LB1,depthBeta,widthBeta,nEpochsBeta,[64],0.001,alpha=alphabeta)
    #mbetamodel_LB1,mtraining_historymBeta_LB1=load_nn_HYM(manifold_name_and_data,-linebundleforHYM_LB1,depthBeta,widthBeta,nEpochsBeta,[64],set_weights_to_zero=False)
    
    
    
    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB2,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB2:
        betamodel_LB2,training_historyBeta_LB2, measure_LB2 = train_and_save_nn_HYM(manifold_name_and_data,linebundleforHYM_LB2,betamodel_config,load_network=False,use_zero_network=use_zero_network_beta)
    else:
        betamodel_LB2,training_historyBeta_LB2, measure_LB2=load_nn_HYM(manifold_name_and_data,linebundleforHYM_LB2,betamodel_config,set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM)
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    purge_dicts_and_mem()
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB3,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB3:
        betamodel_LB3,training_historyBeta_LB3, measure_LB3 = train_and_save_nn_HYM(manifold_name_and_data,linebundleforHYM_LB3,betamodel_config,load_network=False,use_zero_network=use_zero_network_beta)
    else:
        betamodel_LB3,training_historyBeta_LB3, measure_LB3=load_nn_HYM(manifold_name_and_data,linebundleforHYM_LB3,betamodel_config,set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM)

    purge_dicts_and_mem()

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    ##tf.profiler.experimental.start('/mnt/extraspace/kitft/tf_profile_holo_'+str(manifold_name_and_data))
    #tf.profiler.experimental.start('/mnt/extraspace/kitft/tf_profile_holo_'+str(manifold_name_and_data), options=tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,python_tracer_level=1, device_tracer_level=1))

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB1,functionforbaseharmonicform_jbar_for_vH,phimodel,betamodel_LB1,nPointsHF,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vH:
        #HFmodel_vH,trainingHistoryHF_vH, measure = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB1,betamodel_LB1,functionforbaseharmonicform_jbar_for_vH,depthSigma+2,widthSigma//2,nEpochsSigma,[64],lRate=lRateSigma/10,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
        HFmodel_vH,trainingHistoryHF_vH, measure_HF1 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB1,betamodel_LB1,phimodel,functionforbaseharmonicform_jbar_for_vH,sigmamodel_config)
        #was /100
    else:
        HFmodel_vH,trainingHistoryHF_vH, measure_HF1=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB1,betamodel_LB1,phimodel,functionforbaseharmonicform_jbar_for_vH,sigmamodel_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)


    purge_dicts_and_mem()
    #nEpochsSigma=50#changed below vH
    #nEpochsSigma=10

    # Stop the profiler
    #print("stopping profiler")
    #tf.profiler.experimental.stop()
    #print('next')



    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB2,functionforbaseharmonicform_jbar_for_vQ3,phimodel,betamodel_LB2,nPointsHF,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vQ3:
        HFmodel_vQ3,trainingHistoryHF_vQ3, measure_HF2 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vQ3,sigmamodel_config)
    else:
        HFmodel_vQ3,trainingHistoryHF_vQ3, measure_HF2=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vQ3,sigmamodel_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    purge_dicts_and_mem()

    #nEpochsSigma=10
    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB2,functionforbaseharmonicform_jbar_for_vU3,phimodel,betamodel_LB2,nPointsHF,force_generate=force_generate_HF,seed_set=seed_for_gen)

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    if train_vU3:
        HFmodel_vU3,trainingHistoryHF_vU3, measure_HF3 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vU3,sigmamodel_config)
    else:
        HFmodel_vU3,trainingHistoryHF_vU3, measure_HF3=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vU3,sigmamodel_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])
    purge_dicts_and_mem()






    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vQ1,phimodel,betamodel_LB3,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ1:
        HFmodel_vQ1,trainingHistoryHF_vQ1, measure_HF4 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ1,sigma2model_config)
    else:
        HFmodel_vQ1,trainingHistoryHF_vQ1, measure_HF4=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ1,sigma2model_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    purge_dicts_and_mem()
    

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vQ2,phimodel,betamodel_LB3,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ2:   
        HFmodel_vQ2,trainingHistoryHF_vQ2, measure_HF5 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ2,sigma2model_config)
    else:
        HFmodel_vQ2,trainingHistoryHF_vQ2, measure_HF5=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ2,sigma2model_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    purge_dicts_and_mem()   

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vU1,phimodel,betamodel_LB3,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU1:
        HFmodel_vU1,trainingHistoryHF_vU1, measure_HF6 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU1,sigma2model_config)
    else:
        HFmodel_vU1,trainingHistoryHF_vU1, measure_HF6=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU1,sigma2model_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vU2,phimodel,betamodel_LB3,nPointsHF,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU2:
        HFmodel_vU2,trainingHistoryHF_vU2, measure_HF7 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU2,sigma2model_config)
    else:
        HFmodel_vU2,trainingHistoryHF_vU2, measure_HF7=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU2,sigma2model_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF)
    purge_dicts_and_mem()
    
    
    #mem= = tracker.SummaryTracker()
    #print(sorted(mem(.create_summary(), reverse=True, key=itemgetter(2))[:10])

    

    pg,kmoduli=generate_points_and_save_using_defaults_for_eval(manifold_name_and_data,n_to_integrate,seed_set=seed_for_gen,force_generate=force_generate_eval)
    dataEval=np.load(os.path.join(foldername,'tetraquadric_pg_for_eval_with_'+str(unique_id_or_coeff), 'dataset.npz'))


    network_params = {        # Network parameters
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
        'config_phi': phimodel_config,
        'config_beta': betamodel_config,
        'config_sigma': sigmamodel_config,
        'config_sigma2': sigma2model_config,
    }

    do_integrals(manifold_name_and_data, pg, dataEval, phimodel, betamodel_LB1, betamodel_LB2, betamodel_LB3, HFmodel_vH, HFmodel_vQ3, HFmodel_vU3, HFmodel_vQ1, HFmodel_vQ2, HFmodel_vU1, HFmodel_vU2, network_params, do_extra_stuff = do_extra_stuff_for_integration)

