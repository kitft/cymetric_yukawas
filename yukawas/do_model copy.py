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


name_of_invoking_script = sys.argv[0]
integrate_or_run = sys.argv[1] # integrate or run
modeltype = sys.argv[2] # m13 or m1
free_coefficient = float(sys.argv[3]) # free coefficient
start_from = sys.argv[4] if len(sys.argv) > 3+1 else 'phi' # start from phi, LB1, etc...
npoints_config = sys.argv[5] if len(sys.argv) > 4+1 else '1hundred'
job_id = sys.argv[-1] # job id

if integrate_or_run not in ['integrate', 'run']:
    raise ValueError("Invalid command: " + integrate_or_run, "you ran " + " ".join(sys.argv))

import csv
import numpy as np
import gc
import sys
import os
import re
import logging
import wandb
# Extract job ID if the last argument is an integer
try:
    print(f"Running with job ID: {int(job_id)}")
    job_id = str(job_id)
except (ValueError, IndexError):
    job_id = "direct" 
    print(f"Running with job ID: {job_id}")

# Default wandb behavior: disabled for 'integrate', enabled for 'run'
use_wandb = True

if os.path.exists(os.path.join(os.getcwd(), '.wandb_key')):
    # Override default with explicit command line flag
    if 'wandb' in sys.argv[1:]:
        use_wandb = True
    elif 'nowandb' in sys.argv[1:]:
        use_wandb = False
        
    if use_wandb:
        with open(os.path.join(os.getcwd(), '.wandb_key'), 'r') as f:
            wandb_key = f.read().strip()
            wandb.login(key=wandb_key)
    else:
        os.environ["WANDB_MODE"] = "disabled"
        print("Wandb disabled")
else:
    use_wandb = False
    os.environ["WANDB_MODE"] = "disabled"
    print("Wandb disabled, need to set wandb_key")

        
import pickle
#sys.path.append("/Users/kit/Documents/Phys_Working/MF metric")

logging.basicConfig(stream=sys.stdout)

from cymetric.pointgen.pointgen import PointGenerator
from cymetric.pointgen.nphelper import prepare_dataset, prepare_basis_pickle

import tensorflow as tf
import tensorflow.keras as tfk
tf.get_logger().setLevel('ERROR')

if __name__ == '__main__':
    # Default to not eager unless explicitly requested
    run_eagerly = False
    if 'eager' in sys.argv[1:]:
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

    # if 'savevecs' in sys.argv[1:]:
    #     savevecs = True
    # else:
    #     savevecs = False
    # if 'loadvecs' in sys.argv[1:]:
    #     loadvecs = True
    # else:
    #     loadvecs = False



from cymetric.models.tfmodels import PhiFSModel, MultFSModel, FreeModel, MatrixFSModel, AddFSModel, PhiFSModelToric, MatrixFSModelToric
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.callbacks import SigmaCallback, KaehlerCallback, TransitionCallback, RicciCallback, VolkCallback, AlphaCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, RicciLoss, VolkLoss, TotalLoss

# Import the new network config module
from network_config import (
    NetworkConfig, 
    phi_config_to_dict, 
    dict_to_phi_config,
    use_with_original_train_and_save_nn,
    use_with_original_load_nn_phimodel,
    train_and_save_phi_model,
    load_phi_model,
    train_and_save_beta_model,
    load_beta_model,
    train_and_save_sigma_model,
    load_sigma_model
)

from NewCustomMetrics import *
from HarmonicFormModel import *
from BetaModel import *
from laplacian_funcs import *
from auxiliary_funcs import *
from final_integration import *
from yukawas.generate_and_train_all_nnsHOLO_all import *

use_default_dir = False if 'local' in sys.argv[1:] else True
manifold_name = 'TQ'
import subprocess
# Check if hostname matches comp*/gpu*/hydra* pattern
try:
    result = subprocess.run(['hostnamectl'], capture_output=True, text=True)
    hostname_output = result.stdout
    first_line = hostname_output.split('\n')[0] if hostname_output else ""
    
    if use_default_dir and any(pattern in first_line for pattern in ["Static hostname: comp", "Static hostname: gpu", "Static hostname: hydra"]):
        data_path = "/mnt/extraspace/kitft/cy_yukawas/data"
    else:
        data_path = "data"
except Exception:
    # Default to local path if command fails
    data_path = "data"
print("Saving all files to data path: ",data_path)
if modeltype == "m13":
    type_folder = integrate_or_run+"model13"
    from yukawas.OneAndTwoFormsForLineBundlesModel13 import *
    get_coefficients_here = get_coefficients_m13# vs get_coefficients_m1
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
elif modeltype == "m1":
    type_folder = integrate_or_run+"model1"
    from yukawas.OneAndTwoFormsForLineBundlesModel1 import *
    get_coefficients_here = get_coefficients_m1

    linebundleforHYM_LB1=np.array([0,2,-2,0]) 
    linebundleforHYM_LB2=np.array([1,1,0,-2]) 
    linebundleforHYM_LB3=np.array([-1,-3,2,2]) 
    
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

    kmoduliTQ = np.array([1,1,1,1])

    def functionforbaseharmonicform_jbar_for_vQ1(x):
        return getTypeIIs(x,monomialsTQ,coefficientsTQ,'vQ1')
    def functionforbaseharmonicform_jbar_for_vQ2(x):
        return getTypeIIs(x,monomialsTQ,coefficientsTQ,'vQ2')
    def functionforbaseharmonicform_jbar_for_vU1(x):
        return getTypeIIs(x,monomialsTQ,coefficientsTQ,'vU1')
    def functionforbaseharmonicform_jbar_for_vU2(x):
        return getTypeIIs(x,monomialsTQ,coefficientsTQ,'vU2')

    functionforbaseharmonicform_jbar_for_vQ1.line_bundle = np.array([-1,-3,2,2]) 
    functionforbaseharmonicform_jbar_for_vQ2.line_bundle = np.array([-1,-3,2,2]) 
    functionforbaseharmonicform_jbar_for_vU1.line_bundle = np.array([-1,-3,2,2]) 
    functionforbaseharmonicform_jbar_for_vU2.line_bundle = np.array([-1,-3,2,2]) 



print_memory_usage(start_time_of_process=start_time_of_process, name = __name__)

if __name__ == '__main__':
    tr_batchsize=64
    #free_coefficient=1.000000004# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
    import sys
    seed_for_gen=int((int(free_coefficient*100000000000)+free_coefficient*1000000))%4294967294 # modulo largest seed
    print("seed for gen", seed_for_gen)
    coefficientsTQ = get_coefficients_here(free_coefficient)


    alphabeta=[1,10]
    alphasigma1=[1,10] # down form 50
    alphasigma2=[1,80] # down from 200

    lRatePhi= 0.005
    lRateBeta=0.005
    lRateSigma=0.001# perhaps best as 0.03
    lRateSigma=0.001#//10# perhaps best as 0.03
    lRateSigma2=0.001#//10# perhaps best as 0.03


    # Use command-line arguments if provided, starting from the second argument

    
   

    stddev_phi = 0.05
    stddev_beta = 0.05 
    stddev_sigma = 1
    stddev_sigma2=0.5#0.5
    final_layer_scale_sigma=0.01#0.01
    final_layer_scale_sigma2=0.01#0.3#0.000001#0.01

    if 'skipall' in sys.argv[1:]:
        print("Requested to skip all measures")
        skip_measuresPhi=True
        skip_measuresBeta=True
        skip_measuresHF=True
    elif 'skipnone' in sys.argv[1:]:
        skip_measuresPhi=False
        skip_measuresBeta=False
        skip_measuresHF=False
        print("Requested to skip none of the measures")
    elif 'skipphibeta' in sys.argv[1:]:
        skip_measuresPhi=True
        skip_measuresBeta=True
        skip_measuresHF=False
        print("Requested to skip phi and beta")
    else:
        if integrate_or_run == 'run':
            print("defaulting to skipping phi and beta")
            skip_measuresPhi=False
            skip_measuresBeta=False
            skip_measuresHF=False
        else:
            print("defaulting to skipping all")
            skip_measuresPhi=True
            skip_measuresBeta=True
            skip_measuresHF=True
        #skip_measuresHF


    return_zero_phi= True
    return_zero_HYM = True
    return_zero_HF = False
    return_zero_HF_2 = False
    print("RETURNING ZERO PHI AND HYM")
    
    SecondBSize=1000
    n_to_integrate=1000000
    #n_to_integrate=100000
    
    use_zero_network_phi = True
    use_zero_network_beta = True
    
    nPoints = 1000
    if integrate_or_run == 'integrate':
        if '1hundred' == npoints_config:
            nPoints = 100
            n_to_integrate = 100
        elif '1million' == npoints_config:
            nPoints = 100
            n_to_integrate = 1_000_000
        elif '10million' == npoints_config:
            nPoints = 100
            n_to_integrate = 10_000_000
        elif '20million' == npoints_config:
            nPoints = 100
            n_to_integrate = 20_000_000
        elif 'all100k' == npoints_config:
            nPoints = 100_000
            n_to_integrate = 100_000
        elif 'allhuge' == npoints_config:
            nPoints = 1_000_000
            n_to_integrate = 1_000_000
        elif 'allvast' == npoints_config:
            nPoints = 10_000_000
            n_to_integrate = 10_000_000
        else:
            raise ValueError("Invalid data size: " + npoints_config)
    elif integrate_or_run == 'run':
        if '1hundred_1hundred' == npoints_config:
            nPoints = 100
            n_to_integrate = 100
        elif '300k_1million' == npoints_config:
            nPoints = 300_000
            n_to_integrate = 1_000_000
        elif '300k_10million' == npoints_config:
            nPoints = 300_000
            n_to_integrate = 10_000_000
        elif '100k_100k' == npoints_config:
            nPoints = 100_000
            n_to_integrate = 100_000
        elif '1million_1million' == npoints_config:
            nPoints = 1_000_000
            n_to_integrate = 1_000_000
        elif '1hundred_1million' == npoints_config:
            nPoints = 100
            n_to_integrate = 1_000_000
        elif '1hundred_10million' == npoints_config:
            nPoints = 100
            n_to_integrate = 10_000_000
        else:
            raise ValueError("Invalid data size: " + npoints_config)
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
    if 'largenetworks' in sys.argv[1:]:
        depthPhi = 4
        widthPhi = 100
        depthBeta = 4
        widthBeta = 100
        depthSigma = 4
        widthSigma = 100
        depthSigma2 = 4
        widthSigma2 = 100


    
    return_random_phi = False
    return_random_HYM = False
    return_random_HF = True
    return_random_HF_2 = True
    




    stddev_phi = 0.05
    stddev_beta = 0.05
    norm_momentum_sigma = 0.999
    norm_momentum_sigma2 = 0.999



    phi_model_load_function = BiholoModelFuncGENERAL
    beta_model_load_function = BiholoModelFuncGENERALforHYMinv3
    sigma_model_load_function = BiholoModelFuncGENERALforSigma2_m13
    sigma2_model_load_function = BiholoModelFuncGENERALforSigma2_m13
    activationphi = tf.square
    activationbeta = tf.square
    activationsigma = tf.square
    activationsigma2 = tf.square


        
    if 'loadalldata' in sys.argv[1:]:
        force_generate_phi=False
        force_generate_HYM=False
        force_generate_HF=False
        force_generate_HF_2=False
        force_generate_eval=False
    elif 'loadphibeta' in sys.argv[1:]:
        force_generate_phi=False
        force_generate_HYM=False
        force_generate_HF=True
        force_generate_HF_2=True
        force_generate_eval=True
    else:
        force_generate_phi=True
        force_generate_HYM=True
        force_generate_HF=True
        force_generate_HF_2=True
        force_generate_eval=True


    unique_name_phi = 'phi'
    unique_name_LB1 = 'LB1'
    unique_name_LB2 = 'LB2'
    unique_name_LB3 = 'LB3'
    unique_name_vH = 'vH'
    unique_name_vQ3 = 'vQ3'
    unique_name_vU3 = 'vU3'
    unique_name_vQ1 = 'vQ1'
    unique_name_vQ2 = 'vQ2'
    unique_name_vU1 = 'vU1'
    unique_name_vU2 = 'vU2'

    # Create configurations using the new NetworkConfig class
    phi_config = NetworkConfig(
        name=unique_name_phi,
        depth=depthPhi,
        width=widthPhi,
        activation=activationphi,
        learning_rate=lRatePhi,
        weight_stddev=stddev_phi,
        batch_sizes=[tr_batchsize, SecondBSize],
        epochs=nEpochsPhi,
        network_class=phi_model_load_function,
        weight_decay=True
    )
    
    beta_config = NetworkConfig(
        name=unique_name_LB1,  # Will be updated for each line bundle
        depth=depthBeta,
        width=widthBeta,
        activation=activationbeta,
        learning_rate=lRateBeta,
        weight_stddev=stddev_beta,
        batch_sizes=[tr_batchsize],
        epochs=nEpochsBeta,
        network_class=beta_model_load_function,
        additional_params={'alpha': alphabeta}
    )
    
    sigma_config = NetworkConfig(
        name=unique_name_vH,  # Will be updated for each form
        depth=depthSigma,
        width=widthSigma,
        activation=activationsigma,
        learning_rate=lRateSigma,
        weight_stddev=stddev_sigma,
        batch_sizes=[tr_batchsize],
        epochs=nEpochsSigma,
        network_class=sigma_model_load_function,
        additional_params={
            'alpha': alphasigma1,
            'final_layer_scale': final_layer_scale_sigma,
            'norm_momentum': norm_momentum_sigma
        }
    )
    
    sigma2_config = NetworkConfig(
        name=unique_name_vQ1,  # Will be updated for each form
        depth=depthSigma2,
        width=widthSigma2,
        activation=activationsigma2,
        learning_rate=lRateSigma2,
        weight_stddev=stddev_sigma2,
        batch_sizes=[tr_batchsize],
        epochs=nEpochsSigma2,
        network_class=sigma2_model_load_function,
        additional_params={
            'alpha': alphasigma2,
            'final_layer_scale': final_layer_scale_sigma2,
            'norm_momentum': norm_momentum_sigma2
        }
    )
    
    # For compatibility with existing code, also maintain the dictionary configs
    phimodel_config = phi_config_to_dict(phi_config)
    betamodel_config = phi_config_to_dict(beta_config)
    sigmamodel_config = phi_config_to_dict(sigma_config)
    sigma2model_config = phi_config_to_dict(sigma2_config)
    
    print("phimodel_config: ", phimodel_config)
    print("betamodel_config: ", betamodel_config)
    print("sigmamodel_config: ", sigmamodel_config)
    print("sigma2model_config: ", sigma2model_config)

    print(f"Skipping measures? phi? {skip_measuresPhi}, beta? {skip_measuresBeta}, HF? {skip_measuresHF}")
    print("Number of points: " + str(nPoints), "Number of points to integrate: " + str(n_to_integrate))
    print(f"shapes, phi: {depthPhi}x{widthPhi}, beta: {depthBeta}x{widthBeta}, HF: {depthSigma}x{widthSigma}, HF2: {depthSigma2}x{widthSigma2}")

    print("phimodel_config: ", phimodel_config)
    print("betamodel_config: ", betamodel_config)
    print("sigmamodel_config: ", sigmamodel_config)
    print("sigma2model_config: ", sigma2model_config)
    


if __name__ == '__main__':
    print("Name of invoking script: ", name_of_invoking_script, "modeltype: ", modeltype, "namespace of vH: ", functionforbaseharmonicform_jbar_for_vH.__module__)
    if modeltype == "m13":
        if get_coefficients_here != get_coefficients_m13 or functionforbaseharmonicform_jbar_for_vH.__module__ not in ['yukawas.OneAndTwoFormsForLineBundlesModel13','OneAndTwoFormsForLineBundlesModel13']:
            raise ValueError("invalid configuration for m13: ", get_coefficients_here, functionforbaseharmonicform_jbar_for_vH.__module__)
    elif modeltype == "m1":
        if get_coefficients_here != get_coefficients_m1 or functionforbaseharmonicform_jbar_for_vH.__module__ not in ['yukawas.OneAndTwoFormsForLineBundlesModel1','OneAndTwoFormsForLineBundlesModel1']:
            raise ValueError("invalid configuration for m1: ", get_coefficients_here, functionforbaseharmonicform_jbar_for_vH.__module__)
    else:
        raise ValueError("Invalid model specified")

    
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
    



def purge_dicts_and_mem():
    delete_all_dicts_except('dataEval','manifold_name_and_data', 'phimodel_config', 'betamodel_config', 'sigmamodel_config', 'sigma2model_config')
    gc.collect()
    tf.keras.backend.clear_session()
 
do_extra_stuff_for_integration = True
    

if __name__ ==  '__main__':
    seed_for_gen=int((int(free_coefficient*100000000000)+free_coefficient*1000000))%4294967294 # modulo largest seed
    print("seed for gen", seed_for_gen)

    unique_id_or_coeff = free_coefficient
    coefficientsTQ = get_coefficients_here(free_coefficient)
    manifold_name_and_data = (coefficientsTQ, kmoduliTQ, ambientTQ, monomialsTQ, type_folder, unique_id_or_coeff, manifold_name, data_path)
    
    if start_from != 'end':
        wandb.init(project = type_folder,
            name = f'{modeltype}_fc_{unique_id_or_coeff}_{n_to_integrate}_{job_id}',
            config = {'unique_id_or_coeff': unique_id_or_coeff,
                      'phimodel_config': phimodel_config,
                      'betamodel_config': betamodel_config,
                      'sigmamodel_config': sigmamodel_config,
                      'sigma2model_config': sigma2model_config,
                      'invoking_command': ' '.join(sys.argv),
                      'nPoints': nPoints,
                      'n_to_integrate': n_to_integrate,
                      'tags': [integrate_or_run, modeltype, "nint"+str(n_to_integrate), "npoints"+str(nPoints)]})
    
    
    
    generate_points_and_save_using_defaults(manifold_name_and_data,nPoints,force_generate=force_generate_phi,seed_set=seed_for_gen)
    if train_phi:
        # Use the NetworkConfig-based function directly
        phimodel, training_history, measure_phi = train_and_save_phi_model(
            manifold_name_and_data, 
            phi_config, 
            use_zero_network=use_zero_network_phi
        )
    else:
        # Use the NetworkConfig-based function for loading
        phimodel, training_history, measure_phi = load_phi_model(
            manifold_name_and_data, 
            phi_config, 
            set_weights_to_zero=return_zero_phi,
            skip_measures=skip_measuresPhi,
            set_weights_to_random=return_random_phi
        )


    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB1,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB1:
        # Update the config name for this specific model
        beta_config.name = unique_name_LB1
        betamodel_LB1, training_historyBeta_LB1, measure_LB1 = train_and_save_beta_model(
            manifold_name_and_data,
            linebundleforHYM_LB1,
            beta_config,
            phimodel,
            use_zero_network=use_zero_network_beta
        )
    else: 
        beta_config.name = unique_name_LB1
        betamodel_LB1, training_historyBeta_LB1, measure_LB1 = load_beta_model(
            manifold_name_and_data,
            linebundleforHYM_LB1,
            beta_config,
            phimodel,
            set_weights_to_zero=return_zero_HYM,
            skip_measures=skip_measuresBeta,
            set_weights_to_random=return_random_HYM
        )
    purge_dicts_and_mem()

    
    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB2,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB2:
        beta_config.name = unique_name_LB2
        betamodel_LB2, training_historyBeta_LB2, measure_LB2 = train_and_save_beta_model(
            manifold_name_and_data,
            linebundleforHYM_LB2,
            beta_config,
            phimodel,
            use_zero_network=use_zero_network_beta
        )
    else:
        beta_config.name = unique_name_LB2
        betamodel_LB2, training_historyBeta_LB2, measure_LB2 = load_beta_model(
            manifold_name_and_data,
            linebundleforHYM_LB2,
            beta_config,
            phimodel,
            set_weights_to_zero=return_zero_HYM,
            skip_measures=skip_measuresBeta,
            set_weights_to_random=return_random_HYM
        )
    purge_dicts_and_mem()
  

    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB3,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB3:
        beta_config.name = unique_name_LB3
        betamodel_LB3, training_historyBeta_LB3, measure_LB3 = train_and_save_beta_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            beta_config,
            phimodel,
            use_zero_network=use_zero_network_beta
        )
    else:
        beta_config.name = unique_name_LB3
        betamodel_LB3, training_historyBeta_LB3, measure_LB3 = load_beta_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            beta_config,
            phimodel,
            set_weights_to_zero=return_zero_HYM,
            skip_measures=skip_measuresBeta,
            set_weights_to_random=return_random_HYM
        )
    purge_dicts_and_mem()

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB1,functionforbaseharmonicform_jbar_for_vH,phimodel,betamodel_LB1,nPoints,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vH:
        sigma_config.name = unique_name_vH
        HFmodel_vH, trainingHistoryHF_vH, measure_HF1 = train_and_save_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB1,
            betamodel_LB1,
            phimodel,
            functionforbaseharmonicform_jbar_for_vH,
            sigma_config
        )
    else:
        sigma_config.name = unique_name_vH
        HFmodel_vH, trainingHistoryHF_vH, measure_HF1 = load_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB1,
            betamodel_LB1,
            phimodel,
            functionforbaseharmonicform_jbar_for_vH,
            sigma_config,
            set_weights_to_zero=return_zero_HF,
            skip_measures=skip_measuresHF,
            set_weights_to_random=return_random_HF
        )
    purge_dicts_and_mem()





    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB2,functionforbaseharmonicform_jbar_for_vQ3,phimodel,betamodel_LB2,nPoints,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vQ3:
        sigma_config.name = unique_name_vQ3
        HFmodel_vQ3, trainingHistoryHF_vQ3, measure_HF2 = train_and_save_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB2,
            betamodel_LB2,
            phimodel,
            functionforbaseharmonicform_jbar_for_vQ3,
            sigma_config
        )
    else:
        sigma_config.name = unique_name_vQ3
        HFmodel_vQ3, trainingHistoryHF_vQ3, measure_HF2 = load_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB2,
            betamodel_LB2,
            phimodel,
            functionforbaseharmonicform_jbar_for_vQ3,
            sigma_config,
            set_weights_to_zero=return_zero_HF,
            skip_measures=skip_measuresHF,
            set_weights_to_random=return_random_HF
        )
    purge_dicts_and_mem()


    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB2,functionforbaseharmonicform_jbar_for_vU3,phimodel,betamodel_LB2,nPoints,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vU3:
        sigma_config.name = unique_name_vU3
        HFmodel_vU3, trainingHistoryHF_vU3, measure_HF3 = train_and_save_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB2,
            betamodel_LB2,
            phimodel,
            functionforbaseharmonicform_jbar_for_vU3,
            sigma_config
        )
    else:
        sigma_config.name = unique_name_vU3
        HFmodel_vU3, trainingHistoryHF_vU3, measure_HF3 = load_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB2,
            betamodel_LB2,
            phimodel,
            functionforbaseharmonicform_jbar_for_vU3,
            sigma_config,
            set_weights_to_zero=return_zero_HF,
            skip_measures=skip_measuresHF,
            set_weights_to_random=return_random_HF
        )
    purge_dicts_and_mem()






    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vQ1,phimodel,betamodel_LB3,nPoints,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ1:
        sigma2_config.name = unique_name_vQ1
        HFmodel_vQ1, trainingHistoryHF_vQ1, measure_HF4 = train_and_save_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            betamodel_LB3,
            phimodel,
            functionforbaseharmonicform_jbar_for_vQ1,
            sigma2_config
        )
    else:
        sigma2_config.name = unique_name_vQ1
        HFmodel_vQ1, trainingHistoryHF_vQ1, measure_HF4 = load_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            betamodel_LB3,
            phimodel,
            functionforbaseharmonicform_jbar_for_vQ1,
            sigma2_config,
            set_weights_to_zero=return_zero_HF_2,
            skip_measures=skip_measuresHF,
            set_weights_to_random=return_random_HF_2
        )
    purge_dicts_and_mem()
    

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vQ2,phimodel,betamodel_LB3,nPoints,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ2:   
        sigma2_config.name = unique_name_vQ2
        HFmodel_vQ2, trainingHistoryHF_vQ2, measure_HF5 = train_and_save_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            betamodel_LB3,
            phimodel,
            functionforbaseharmonicform_jbar_for_vQ2,
            sigma2_config
        )
    else:
        sigma2_config.name = unique_name_vQ2
        HFmodel_vQ2, trainingHistoryHF_vQ2, measure_HF5 = load_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            betamodel_LB3,
            phimodel,
            functionforbaseharmonicform_jbar_for_vQ2,
            sigma2_config,
            set_weights_to_zero=return_zero_HF_2,
            skip_measures=skip_measuresHF,
            set_weights_to_random=return_random_HF_2
        )
    purge_dicts_and_mem()   

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vU1,phimodel,betamodel_LB3,nPoints,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU1:
        sigma2_config.name = unique_name_vU1
        HFmodel_vU1, trainingHistoryHF_vU1, measure_HF6 = train_and_save_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            betamodel_LB3,
            phimodel,
            functionforbaseharmonicform_jbar_for_vU1,
            sigma2_config
        )
    else:
        sigma2_config.name = unique_name_vU1
        HFmodel_vU1, trainingHistoryHF_vU1, measure_HF6 = load_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            betamodel_LB3,
            phimodel,
            functionforbaseharmonicform_jbar_for_vU1,
            sigma2_config,
            set_weights_to_zero=return_zero_HF_2,
            skip_measures=skip_measuresHF,
            set_weights_to_random=return_random_HF_2
        )  
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vU2,phimodel,betamodel_LB3,nPoints,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU2:
        sigma2_config.name = unique_name_vU2
        HFmodel_vU2, trainingHistoryHF_vU2, measure_HF7 = train_and_save_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            betamodel_LB3,
            phimodel,
            functionforbaseharmonicform_jbar_for_vU2,
            sigma2_config
        )
    else:
        sigma2_config.name = unique_name_vU2
        HFmodel_vU2, trainingHistoryHF_vU2, measure_HF7 = load_sigma_model(
            manifold_name_and_data,
            linebundleforHYM_LB3,
            betamodel_LB3,
            phimodel,
            functionforbaseharmonicform_jbar_for_vU2,
            sigma2_config,
            set_weights_to_zero=return_zero_HF_2,
            skip_measures=skip_measuresHF,
            set_weights_to_random=return_random_HF_2
        )
    purge_dicts_and_mem()
    

    pg,kmoduli=generate_points_and_save_using_defaults_for_eval(manifold_name_and_data,n_to_integrate,seed_set=seed_for_gen,force_generate=force_generate_eval)
    dirnameEval = os.path.join(data_path,type_folder,f'{manifold_name}_pg_for_eval_with_{unique_id_or_coeff}')
    if not os.path.exists(dirnameEval):
        raise FileNotFoundError(f"Directory {dirnameEval} not found.")
    dataEval=np.load(os.path.join(dirnameEval, 'dataset.npz'))


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
        'points_integration': n_to_integrate,
        'measure_phi': measure_phi,
        'measure_LBs': [measure_LB1, measure_LB2, measure_LB3],
        'measure_HFs': [measure_HF1, measure_HF2, measure_HF3, measure_HF4, measure_HF5, measure_HF6, measure_HF7],
        'config_phi': phimodel_config,
        'config_beta': betamodel_config,
        'config_sigma': sigmamodel_config,
        'config_sigma2': sigma2model_config,
    }

    masses, masserrors = do_integrals(manifold_name_and_data, pg, dataEval, phimodel, betamodel_LB1, betamodel_LB2, betamodel_LB3, HFmodel_vH,
                                       HFmodel_vQ3, HFmodel_vU3, HFmodel_vQ1, HFmodel_vQ2, HFmodel_vU1, HFmodel_vU2, network_params, do_extra_stuff = do_extra_stuff_for_integration,
                                         run_args=sys.argv, dirnameEval=dirnameEval)

print("\nYou ran " + " ".join(sys.argv))
print("--------------------------------")
print("Trained masses:")
for i in range(3):
    print(f"  m{i+1}: {masses[0,i]:.6e} ± {masserrors[i]:.6e}")
print("\nReference masses:")
for i in range(3):
    print(f"  m{i+1}: {masses[1,i]:.6e} ± {masserrors[i]:.6e}")
print("--------------------------------")
