from cymetric.config import real_dtype, complex_dtype
import numpy as np
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
from yukawas.OneAndTwoFormsForLineBundlesModel1 import *
from generate_and_train_all_nns import *


# nPoints=300000
# free_coefficient=1.# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
# nEpochsPhi=100
# nEpochsBeta=60
# nEpochsSigma=50


nPoints=300000
#free_coefficient=1.000000004# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
import sys
#free_coefficient = float(sys.argv[1])
free_coefficientphi=1.000000006
free_coefficient=1.0000000124
#free_coefficient=1.# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
nEpochsPhi=150
nEpochsBeta=150
nEpochsSigma=50

# %%capture outputfor00000001
if __name__ ==  '__main__':
    generate_points_and_save_using_defaults(free_coefficientphi,nPoints)
    #phimodel1,training_history=train_and_save_nn(free_coefficient,4,128,nEpochsPhi,[64,50000],0.001) 
    phimodel1,training_history=load_nn_phimodel(free_coefficientphi,4,128,nEpochsPhi,[64,50000],set_weights_to_zero=False)

    linebundleforHYM_02m20=np.array([0,2,-2,0]) 
    linebundleforHYM_110m2=np.array([1,1,0,-2]) 
    linebundleforHYM_m1m322=np.array([-1,-3,2,2]) 


    generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM_02m20,nPoints,phimodel1,force_generate=True)
    #betamodel_02m20,training_historyBeta_02m20=train_and_save_nn_HYM(free_coefficient,linebundleforHYM_02m20,4,128,nEpochsBeta,[64,50000],0.01)
    betamodel_02m20,training_historyBeta_02m20=load_nn_HYM(free_coefficient,linebundleforHYM_02m20,4,128,nEpochsBeta,[64,50000],set_weights_to_zero=False)


    generate_points_and_save_using_defaultsHYM(free_coefficient,-linebundleforHYM_02m20,nPoints,phimodel1,force_generate=True)
    #mbetamodel_02m20,training_historymBeta_02m20=train_and_save_nn_HYM(free_coefficient,-linebundleforHYM_02m20,4,128,nEpochsBeta,[64,50000],0.01)
    mbetamodel_02m20,training_historymBeta_02m20=load_nn_HYM(free_coefficient,-linebundleforHYM_02m20,4,128,nEpochsBeta,[64,50000],set_weights_to_zero=False)

    pg,kmoduli=generate_points_and_save_using_defaults_for_eval(free_coefficient,300000)
    dataEval=np.load(os.path.join('tetraquadric_pg_for_eval_with_'+str(free_coefficient), 'dataset.npz'))
    n_p = 300000
    phi = phimodel1
    points64=dataEval['X_train'][0:n_p]
    real_pts=tf.cast(dataEval['X_train'][0:n_p],np.float32)
    pointsComplex=tf.cast(point_vec_to_complex(points64),complex_dtype)# we have to cast to complex64 from complex128 sometimes. Just for safety. point_vec_to_complex does the obvious thing
    #mets=phimodel.fubini_study_pb(real_pts,ts=tf.cast(kmoduli,complex_dtype))

    print("\n do analysis: \n\n\n")

    
    H1=betamodel_02m20(real_pts) 
    H2=mbetamodel_02m20(real_pts) 


    prodbundles=tf.reduce_mean(H1*H2)
    prodbundlesstddev=tf.math.reduce_std(H1*H2)
    print("test if good bundle metric" + str(prodbundles) + " and std ddev " + str(prodbundlesstddev))

    prodbundles=tf.reduce_mean(H1)
    prodbundlestdev=tf.math.reduce_std(H1)
    print("mean of first bundle metric" + str(prodbundles) + " and std ddev " + str(prodbundlestdev))

    prodbundles=tf.reduce_mean(H2)
    prodbundlestdev=tf.math.reduce_std(H2)
    print("mean of second bundle metric" + str(prodbundles) + " and std ddev " + str(prodbundlestdev))
