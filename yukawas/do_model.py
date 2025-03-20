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
    resulthostname = subprocess.run(['hostname'], capture_output=True, text=True)
    if 'harvard' in resulthostname.stdout:
        print("We're on the harvard cluster")
        data_path = "data"
        try:
            # from dotenv import load_dotenv
            # load_dotenv()
            # import os
            # if not os.getenv("WANDB_API_KEY"):
            #     raise ValueError("WANDB_API_KEY not found in .env file")
            wandb.login()
        except FileNotFoundError:
            raise FileNotFoundError("No .env file found. Please create one with your WANDB_API_KEY.")
        except ImportError:
            raise ImportError("python-dotenv package not installed. Please install it with 'pip install python-dotenv'")
    else:
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

result_files_path = "results"
# Set addtofilename based on command line arguments

# Check for any other name{whatever} pattern
addtofilename = ""
for arg in sys.argv[1:]:
    if arg.startswith('name'):
        addtofilename = arg[4:]  # Extract whatever comes after 'name'
        break

print("Adding to final filename: ",addtofilename)

print("Saving all files to data path: ",data_path)
print("Saving results to : ",result_files_path)
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
elif modeltype == "m1rotated":
    type_folder = integrate_or_run+"model1rotated"
    from yukawas.RotatedOneAndTwoFormsForLineBundlesModel1 import *
    get_coefficients_here = get_coefficients_m1

    linebundleforHYM_LB1=np.array([0,0,2,-2]) 
    linebundleforHYM_LB2=np.array([-2,1,1,0]) 
    linebundleforHYM_LB3=np.array([2,-1,-3,2]) 
    
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

    functionforbaseharmonicform_jbar_for_vQ1.line_bundle = np.array([2,-1,-3,2]) 
    functionforbaseharmonicform_jbar_for_vQ2.line_bundle = np.array([2,-1,-3,2]) 
    functionforbaseharmonicform_jbar_for_vU1.line_bundle = np.array([2,-1,-3,2]) 
    functionforbaseharmonicform_jbar_for_vU2.line_bundle = np.array([2,-1,-3,2]) 





print_memory_usage(start_time_of_process=start_time_of_process, name = __name__)

if __name__ == '__main__':
    tr_batchsize=64
    #free_coefficient=1.000000004# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
    import sys
    seed_for_gen=int(int((int(free_coefficient*100000000000)+free_coefficient*1000000))+((free_coefficient*10**(18))%123987123157869))%4294967294 # modulo largest seed
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
        elif '100k' == npoints_config:
            nPoints = 100
            n_to_integrate = 100_000
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



    phi_model_load_function = None 
    beta_model_load_function = None
    sigma_model_load_function = None
    sigma2_model_load_function = None
    activationphi = None
    activationbeta = None
    activationsigma = None
    activationsigma2 = None


        
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


    phimodel_config = {'depth': depthPhi, 'width': widthPhi, 'nEpochs': nEpochsPhi, 'lRate': lRatePhi, 'stddev': stddev_phi, 'bSizes': [tr_batchsize,SecondBSize], 'network_function': phi_model_load_function, 'activation': activationphi}
    betamodel_config = {'depth': depthBeta, 'width': widthBeta, 'nEpochs': nEpochsBeta, 'lRate': lRateBeta, 'alpha': alphabeta, 'stddev': stddev_beta, 'bSizes': [tr_batchsize], 'network_function': beta_model_load_function, 'activation':activationbeta}
    sigmamodel_config = {'depth': depthSigma, 'width': widthSigma, 'nEpochs': nEpochsSigma, 'lRate': lRateSigma, 'alpha': alphasigma1, 'stddev': stddev_sigma, 'bSizes': [tr_batchsize], 'final_layer_scale': final_layer_scale_sigma, 'norm_momentum': norm_momentum_sigma, 'network_function': sigma_model_load_function,'activation':activationsigma}
    sigma2model_config = {'depth': depthSigma2, 'width': widthSigma2, 'nEpochs': nEpochsSigma2, 'lRate': lRateSigma2, 'alpha': alphasigma2, 'stddev': stddev_sigma2, 'bSizes': [tr_batchsize], 'final_layer_scale': final_layer_scale_sigma2, 'norm_momentum': norm_momentum_sigma2, 'network_function': sigma2_model_load_function,'activation':activationsigma2}
    
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

    if 'noorbit' in sys.argv[1:]:
        orbit_P1s = False
    elif 'orbitlast' in sys.argv[1:]:
        orbit_P1s = 3
    elif 'orbitsecondlast' in sys.argv[1:]:
        orbit_P1s = 2
    elif 'orbitthirdlast' in sys.argv[1:]:
        orbit_P1s = 1
    elif 'orbitfirst' in sys.argv[1:]:
        orbit_P1s = 0
    elif 'orbit' in sys.argv[1:]:
        orbit_P1s = True
    else:
        orbit_P1s = True
    print(f"Orbiting over P1s")


if __name__ == '__main__':
    print("Name of invoking script: ", name_of_invoking_script, "modeltype: ", modeltype, "namespace of vH: ", functionforbaseharmonicform_jbar_for_vH.__module__)
    if modeltype == "m13":
        if get_coefficients_here != get_coefficients_m13 or functionforbaseharmonicform_jbar_for_vH.__module__ not in ['yukawas.OneAndTwoFormsForLineBundlesModel13','OneAndTwoFormsForLineBundlesModel13']:
            raise ValueError("invalid configuration for m13: ", get_coefficients_here, functionforbaseharmonicform_jbar_for_vH.__module__)
    elif modeltype == "m1":
        if get_coefficients_here != get_coefficients_m1 or  functionforbaseharmonicform_jbar_for_vH.__module__ not in ['yukawas.OneAndTwoFormsForLineBundlesModel1','OneAndTwoFormsForLineBundlesModel1']:
            raise ValueError("invalid configuration for m1: ", get_coefficients_here, functionforbaseharmonicform_jbar_for_vH.__module__)
    elif modeltype == "m1rotated":
        if get_coefficients_here != get_coefficients_m1 or  functionforbaseharmonicform_jbar_for_vH.__module__ not in ['yukawas.RotatedOneAndTwoFormsForLineBundlesModel1','RotatedOneAndTwoFormsForLineBundlesModel1']:
            raise ValueError("invalid configuration for m1rotated: ", get_coefficients_here, functionforbaseharmonicform_jbar_for_vH.__module__)
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

    unique_id_or_coeff = free_coefficient
    coefficientsTQ = get_coefficients_here(free_coefficient)
    manifold_name_and_data = (coefficientsTQ, kmoduliTQ, ambientTQ, monomialsTQ, type_folder, unique_id_or_coeff, manifold_name, data_path)
    
    #if start_from != 'end':
    if n_to_integrate > 500_000 or 'wandb' in sys.argv[1:]:
        wandb.init(project = type_folder,
            name = f'{modeltype}_fc_{unique_id_or_coeff}_{addtofilename}_{job_id}',
            config = {'unique_id_or_coeff': unique_id_or_coeff,
                      'phimodel_config': phimodel_config,
                      'betamodel_config': betamodel_config,
                      'sigmamodel_config': sigmamodel_config,
                      'sigma2model_config': sigma2model_config,
                      'invoking_command': ' '.join(sys.argv),
                      'nPoints': nPoints,
                      'n_to_integrate': n_to_integrate,
                      'doubleprecision': double_precision,
                      'sampling_orbit_type': orbit_P1s,
                      'data_path' : data_path,
                      'manifold_name' : manifold_name,
                      'type_folder' : type_folder,
                      'seed' : seed_for_gen,
                      'tags': [integrate_or_run, modeltype, "nint"+str(n_to_integrate), "npoints"+str(nPoints), addtofilename]})
    
    
    
    generate_points_and_save_using_defaults(manifold_name_and_data,nPoints,force_generate=force_generate_phi,seed_set=seed_for_gen,average_selected_t = orbit_P1s)
    if train_phi:
        #phimodel,training_history=train_and_save_nn(manifold_name_and_data,depthPhi,widthPhi,nEpochsPhi,bSizes=[64,tr_batchsize],lRate=lRatePhi) 
        phimodel,training_history, measure_phi=train_and_save_nn(manifold_name_and_data,phimodel_config,use_zero_network=use_zero_network_phi, unique_name=unique_name_phi)
    else:
        phimodel,training_history, measure_phi=load_nn_phimodel(manifold_name_and_data,phimodel_config,set_weights_to_zero=return_zero_phi,skip_measures=skip_measuresPhi,set_weights_to_random=return_random_phi, unique_name=unique_name_phi)


    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB1,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB1:
        betamodel_LB1,training_historyBeta_LB1, measure_LB1 = train_and_save_nn_HYM(manifold_name_and_data,linebundleforHYM_LB1,betamodel_config,phimodel,load_network=False,use_zero_network=use_zero_network_beta, unique_name=unique_name_LB1)
    else: 
        betamodel_LB1,training_historyBeta_LB1, measure_LB1=load_nn_HYM(manifold_name_and_data,linebundleforHYM_LB1,betamodel_config,phimodel,set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM, unique_name=unique_name_LB1)
    purge_dicts_and_mem()

    
    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB2,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB2:
        betamodel_LB2,training_historyBeta_LB2, measure_LB2 = train_and_save_nn_HYM(manifold_name_and_data,linebundleforHYM_LB2,betamodel_config,phimodel,load_network=False,use_zero_network=use_zero_network_beta, unique_name=unique_name_LB2)
    else:
        betamodel_LB2,training_historyBeta_LB2, measure_LB2=load_nn_HYM(manifold_name_and_data,linebundleforHYM_LB2,betamodel_config,phimodel,set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM, unique_name=unique_name_LB2)
    purge_dicts_and_mem()
  

    generate_points_and_save_using_defaultsHYM(manifold_name_and_data,linebundleforHYM_LB3,nPoints,phimodel,force_generate=force_generate_HYM,seed_set=seed_for_gen)
    if train_LB3:
        betamodel_LB3,training_historyBeta_LB3, measure_LB3 = train_and_save_nn_HYM(manifold_name_and_data,linebundleforHYM_LB3,betamodel_config,phimodel,load_network=False,use_zero_network=use_zero_network_beta, unique_name=unique_name_LB3)
    else:
        betamodel_LB3,training_historyBeta_LB3, measure_LB3=load_nn_HYM(manifold_name_and_data,linebundleforHYM_LB3,betamodel_config,phimodel,set_weights_to_zero=return_zero_HYM,skip_measures=skip_measuresBeta,set_weights_to_random=return_random_HYM, unique_name=unique_name_LB3)
    purge_dicts_and_mem()

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB1,functionforbaseharmonicform_jbar_for_vH,phimodel,betamodel_LB1,nPoints,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vH:
        #HFmodel_vH,trainingHistoryHF_vH, measure = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB1,betamodel_LB1,functionforbaseharmonicform_jbar_for_vH,depthSigma+2,widthSigma//2,nEpochsSigma,[64],lRate=lRateSigma/10,alpha=alphasigma1,stddev=stddev_H33,final_layer_scale=final_layer_scale_H33)
        HFmodel_vH,trainingHistoryHF_vH, measure_HF1 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB1,betamodel_LB1,phimodel,functionforbaseharmonicform_jbar_for_vH,sigmamodel_config, unique_name=unique_name_vH)
        #was /100
    else:
        HFmodel_vH,trainingHistoryHF_vH, measure_HF1=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB1,betamodel_LB1,phimodel,functionforbaseharmonicform_jbar_for_vH,sigmamodel_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF, unique_name=unique_name_vH)
    purge_dicts_and_mem()





    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB2,functionforbaseharmonicform_jbar_for_vQ3,phimodel,betamodel_LB2,nPoints,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vQ3:
        HFmodel_vQ3,trainingHistoryHF_vQ3, measure_HF2 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vQ3,sigmamodel_config, unique_name=unique_name_vQ3)
    else:
        HFmodel_vQ3,trainingHistoryHF_vQ3, measure_HF2=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vQ3,sigmamodel_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF, unique_name=unique_name_vQ3)
    purge_dicts_and_mem()


    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB2,functionforbaseharmonicform_jbar_for_vU3,phimodel,betamodel_LB2,nPoints,force_generate=force_generate_HF,seed_set=seed_for_gen)
    if train_vU3:
        HFmodel_vU3,trainingHistoryHF_vU3, measure_HF3 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vU3,sigmamodel_config, unique_name=unique_name_vU3)
    else:
        HFmodel_vU3,trainingHistoryHF_vU3, measure_HF3=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB2,betamodel_LB2,phimodel,functionforbaseharmonicform_jbar_for_vU3,sigmamodel_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF, unique_name=unique_name_vU3)
    purge_dicts_and_mem()






    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vQ1,phimodel,betamodel_LB3,nPoints,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ1:
        HFmodel_vQ1,trainingHistoryHF_vQ1, measure_HF4 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ1,sigma2model_config, unique_name=unique_name_vQ1)
    else:
        HFmodel_vQ1,trainingHistoryHF_vQ1, measure_HF4=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ1,sigma2model_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF, unique_name=unique_name_vQ1)
    purge_dicts_and_mem()
    

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vQ2,phimodel,betamodel_LB3,nPoints,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vQ2:   
        HFmodel_vQ2,trainingHistoryHF_vQ2, measure_HF5 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ2,sigma2model_config, unique_name=unique_name_vQ2)
    else:
        HFmodel_vQ2,trainingHistoryHF_vQ2, measure_HF5=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vQ2,sigma2model_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF, unique_name=unique_name_vQ2)
    purge_dicts_and_mem()   

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vU1,phimodel,betamodel_LB3,nPoints,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU1:
        HFmodel_vU1,trainingHistoryHF_vU1, measure_HF6 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU1,sigma2model_config, unique_name=unique_name_vU1)
    else:
        HFmodel_vU1,trainingHistoryHF_vU1, measure_HF6=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU1,sigma2model_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF, unique_name=unique_name_vU1)  
    delete_all_dicts_except('dataEval')
    gc.collect()
    tf.keras.backend.clear_session()

    generate_points_and_save_using_defaultsHF(manifold_name_and_data,linebundleforHYM_LB3,functionforbaseharmonicform_jbar_for_vU2,phimodel,betamodel_LB3,nPoints,force_generate=force_generate_HF_2,seed_set=seed_for_gen)
    if train_vU2:
        HFmodel_vU2,trainingHistoryHF_vU2, measure_HF7 = train_and_save_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU2,sigma2model_config, unique_name=unique_name_vU2)
    else:
        HFmodel_vU2,trainingHistoryHF_vU2, measure_HF7=load_nn_HF(manifold_name_and_data,linebundleforHYM_LB3,betamodel_LB3,phimodel,functionforbaseharmonicform_jbar_for_vU2,sigma2model_config,skip_measures=skip_measuresHF,set_weights_to_random=return_random_HF, unique_name=unique_name_vU2)
    purge_dicts_and_mem()
    

    pg,kmoduli=generate_points_and_save_using_defaults_for_eval(manifold_name_and_data,n_to_integrate,seed_set=seed_for_gen,force_generate=force_generate_eval,average_selected_t = orbit_P1s)
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
        'coefficientsTQ': coefficientsTQ,
        'kmoduliTQ': kmoduliTQ,
        'ambientTQ': ambientTQ,
        'monomialsTQ': monomialsTQ,
        'orbit': orbit_P1s,
        'doubleprecision': double_precision
    }

    masses, masserrors = do_integrals(manifold_name_and_data, pg, dataEval, phimodel, betamodel_LB1, betamodel_LB2, betamodel_LB3, HFmodel_vH,
                                       HFmodel_vQ3, HFmodel_vU3, HFmodel_vQ1, HFmodel_vQ2, HFmodel_vU1, HFmodel_vU2, network_params, do_extra_stuff = do_extra_stuff_for_integration,
                                         run_args=sys.argv, dirnameEval=dirnameEval, result_files_path=result_files_path, addtofilename=addtofilename)

print("\nYou ran " + " ".join(sys.argv))
print("--------------------------------")
print("Trained masses:")
for i in range(3):
    print(f"  m{i+1}: {masses[0,i]:.6e} ± {masserrors[i]:.6e}")
print("\nReference masses:")
for i in range(3):
    print(f"  m{i+1}: {masses[1,i]:.6e} ± {masserrors[i]:.6e}")
print("--------------------------------")
