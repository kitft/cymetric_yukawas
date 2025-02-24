from cymetric.config import real_dtype, complex_dtype
from cymetric.models.fubinistudy import FSModel
import tensorflow as tf
import tensorflow.keras as tfk
from laplacian_funcs import *
from BetaModel import *
from HarmonicFormModel import *
from OneAndTwoFormsForLineBundlesModel13 import *
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.pointgen.nphelper import prepare_dataset, prepare_basis_pickle

from cymetric.models.tfmodels import PhiFSModel, MultFSModel, FreeModel, MatrixFSModel, AddFSModel, PhiFSModelToric, MatrixFSModelToric
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.callbacks import SigmaCallback, KaehlerCallback, TransitionCallback, RicciCallback, VolkCallback, AlphaCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, RicciLoss, VolkLoss, TotalLoss

import os
import numpy as np

from NewCustomMetrics import *
from HarmonicFormModel import *
from BetaModel import *
from laplacian_funcs import *
from OneAndTwoFormsForLineBundlesModel13 import *
from custom_networks import *
from custom_networks import batch_process_helper_func

from pympler import tracker
seed_set=0
use_symmetry_reduced_TQ=False
determine_n_funcs=2

ambient = np.array([1,1,1,1])
monomials = np.array([[2, 0, 2, 0, 2, 0, 2, 0], [2, 0, 2, 0, 2, 0, 1, 1], [2, 0, 2, 0, 2, 
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
#kmoduliTQ = np.array([1,1,1,1])
folder_name = "dataM13"


def create_adam_optimizer_with_decay(initial_learning_rate, nEpochs, final_lr_factor=2):
    """
    Creates an Adam optimizer with exponential learning rate decay.

    Parameters:
    - initial_learning_rate: The starting learning rate
    - nEpochs: The number of epochs over which to decay the learning rate
    - final_lr_factor: The factor by which to reduce the learning rate (default: 10)

    Returns:
    - An Adam optimizer with the specified learning rate decay
    """
    decay_rate = (1/final_lr_factor) ** (1/nEpochs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1,
        decay_rate=decay_rate,
        staircase=False
    )
    #print("cutting out adam optimizer decay")

    return tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)#lr_schedule)

def get_coefficients(free_coefficient):
   x = free_coefficient
   coefficients=np.array([1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, \
0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1]) +\
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
   return coefficients

def generate_points_and_save_using_defaults_for_eval(free_coefficient,number_points,force_generate=False,seed_set=0):
   coefficients=get_coefficients(free_coefficient)
   kmoduli = kmoduliTQ 
   ambient = np.array([1,1,1,1])
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
   pg._set_seed(seed_set)
   dirname =folder_name+ '/tetraquadric_pg_for_eval_with_'+str(free_coefficient) 
   print("dirname: " + dirname)
   #test if the directory exists, if not, create it
   if force_generate or (not os.path.exists(dirname)):
      print("Generating: forced? " + str(force_generate))
      kappa = pg.prepare_dataset(number_points, dirname)
      pg.prepare_basis(dirname, kappa=kappa)
   elif os.path.exists(dirname):
      try:
         data = np.load(os.path.join(dirname, 'dataset.npz'))
         length_total = len(data['X_train'])+len(data['X_val'])
         if length_total!=number_points:
            print(f"wrong length {length_total}, want {number_points} - generating anyway")
            kappa = pg.prepare_dataset(number_points, dirname)
            pg.prepare_basis(dirname, kappa=kappa)
      except:
         print("error loading - generating anyway")
         kappa = pg.prepare_dataset(number_points, dirname)
         pg.prepare_basis(dirname, kappa=kappa)
   return pg,kmoduli


def generate_points_and_save_using_defaults(free_coefficient,number_points,force_generate=False,seed_set=0):
   coefficients=get_coefficients(free_coefficient)
   kmoduli = kmoduliTQ
   ambient = np.array([1,1,1,1])
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
   pg._set_seed(seed_set)
   dirname = folder_name+'/tetraquadric_pg_with_'+str(free_coefficient) 
   print("dirname: " + dirname)
   #test if the directory exists, if not, create it
   if force_generate or (not os.path.exists(dirname)):
      print("Generating: forced? " + str(force_generate))
      kappa = pg.prepare_dataset(number_points, dirname)
      pg.prepare_basis(dirname, kappa=kappa)
   elif os.path.exists(dirname):
      try:
         print("loading prexisting dataset")
         data = np.load(os.path.join(dirname, 'dataset.npz'))
         length_total = len(data['X_train'])+len(data['X_val'])
         if length_total!=number_points:
            print(f"wrong length {length_total}, want {number_points} - generating anyway")
            kappa = pg.prepare_dataset(number_points, dirname)
            pg.prepare_basis(dirname, kappa=kappa)
      except:
         print("error loading - generating anyway")
         kappa = pg.prepare_dataset(number_points, dirname)
         pg.prepare_basis(dirname, kappa=kappa)
   

def getcallbacksandmetrics(data):
   #rcb = RicciCallback((data['X_val'], data['y_val']), data['val_pullbacks'])
   scb = SigmaCallback((data['X_val'], data['y_val']))
   volkcb = VolkCallback((data['X_val'], data['y_val']))
   kcb = KaehlerCallback((data['X_val'], data['y_val']))
   tcb = TransitionCallback((data['X_val'], data['y_val']))
   #cb_list = [rcb, scb, kcb, tcb, volkcb]
   #cb_list = [ scb, kcb, tcb, volkcb]
   #cmetrics = [TotalLoss(), SigmaLoss(), KaehlerLoss(), TransitionLoss(), VolkLoss()]#, RicciLoss()]
   cb_list = [ scb ]
   cmetrics = [TotalLoss(), SigmaLoss()]#, RicciLoss()]
   return cb_list, cmetrics

def train_and_save_nn(free_coefficient,nlayer=3,nHidden=128,nEpochs=50,stddev=0.1,bSizes=[192,50000],lRate=0.001,use_zero_network=False):
   dirname = folder_name +'/tetraquadric_pg_with_'+str(free_coefficient)
   name = 'phimodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(bSizes[1]) + 's' + str(nlayer) + 'x' +str(nHidden)
   print('dirname: ' + dirname)
   print('name: ' + name)
   
   data = np.load(os.path.join(dirname, 'dataset.npz'))
   data = convert_to_tensor_dict(data)
   BASIS = prepare_tf_basis(np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True))

   cb_list, cmetrics = getcallbacksandmetrics(data)

   #nlayer = 3
   #nHidden = 128
   act = 'gelu'
   #nEpochs = 100
   #bSizes = [192, 150000]
   alpha = [1., 1., 30., 1., 2.] # 1 AND 3??
   nfold = 3
   n_in = 2*8
   n_out = 1
   #lRate = 0.001
   ambient=tf.cast(tf.math.abs(BASIS['AMBIENT']),tf.int32)

   nfirstlayer= tf.reduce_prod((np.array(ambient)+determine_n_funcs)).numpy().item() if use_symmetry_reduced_TQ else tf.reduce_prod((np.array(ambient)+1)**2).numpy().item() 
   shapeofinternalnetwork=[nHidden]*nlayer
   shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1]

   print("network shape: " + str(shapeofnetwork))
   nn_phi = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,stddev=stddev,use_zero_network=use_zero_network,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_phi_zero =BiholoModelFuncGENERAL(shapeofnetwork,BASIS,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   phimodel = PhiFSModel(nn_phi, BASIS, alpha=alpha)
   phimodelzero = PhiFSModel(nn_phi_zero, BASIS, alpha=alpha)
   #print('nn_phi ' )
   #print('nn_phi ' + str(phimodel(data['X_train'][0:10])))

   #Note, currently running legacy due to ongoing tf issue with M1/M2. 
   #Use the commented line instead if not on an M1/M2 machine
   #opt = tfk.optimizers.Adam(learning_rate=lRate)
   #opt = tfk.optimizers.legacy.Adam(learning_rate=lRate)
   opt = create_adam_optimizer_with_decay(
    initial_learning_rate=lRate,
    nEpochs=nEpochs*(len(data['X_train'])//bSizes[0]),
    final_lr_factor=2  # This will decay to lRate/10
)
   # compile so we can test on validation set before training
   phimodel.compile(custom_metrics=cmetrics)
   phimodelzero.compile(custom_metrics=cmetrics)

   ## compare validation loss before training for zero network and nonzero network
   datacasted=[data['X_val'],data['y_val']]
   #need to re-enable learning, in case there's been a problem:
   phimodel.learn_transition = False
   phimodelzero.learn_transition = False
   phimodel.learn_volk = True
   phimodelzero.learn_volk = True
   #phimodel.learn_ricci_val= True
   #phimodelzero.learn_ricci_val= True
   print("testing")
   valzero=phimodelzero.test_step(datacasted)
   valraw=phimodel.test_step(datacasted)
   print('tested')
   # phimodel.learn_ricci_val=False 
   # phimodelzero.learn_ricci_val=False 
   valzero = {key: float(value.numpy()) for key, value in valzero.items()}
   valraw = {key: float(value.numpy()) for key, value in valraw.items()}

   #print("CHECKING MODEL:")
   #print(data['X_train'][0:1])
   #print(phimodel(data['X_train'][0:1]))
   #print(phimodelzero(data['X_train'][0:1]))
   #print(phimodel.fubini_study_pb(data['X_train'][0:1]))
   #print("DONE")

   #### maybe remove
   phimodel.learn_volk = False
   phimodel.learn_transition=False
   phimodel, training_history = train_model(phimodel, data, optimizer=opt, epochs=nEpochs, batch_sizes=bSizes, 
                                       verbose=1, custom_metrics=cmetrics, callbacks=cb_list)
   print("finished training\n")
   phimodel.model.save_weights(os.path.join(dirname, name) + '.weights.h5')
   np.savez_compressed(os.path.join(dirname, 'trainingHistory-' + name),training_history)
   #now print the initial losses and final losses for each metric
   # first_metrics = {key: value[0] for key, value in training_history.items()}
   # lastometrics = {key: value[-1] for key, value in training_history.items()}
   phimodel.learn_transition = True
   phimodel.learn_volk = True
   #phimodel.learn_ricci_val= True
   valfinal=phimodel.test_step(datacasted)
   valfinal = {key: value.numpy() for key, value in valfinal.items()}
   #phimodel.learn_ricci_val=False 
   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for raw network: ")
   print(valraw)
   print("validation loss for final network: ")
   print(valfinal)
   print("ratio of final to zero: " + str({key + " ratio": value/(valzero[key]+1e-8) for key, value in valfinal.items()}))
   print("ratio of final to raw: " + str({key + " ratio": value/(valraw[key]+1e-8) for key, value in valfinal.items()}))

   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(phimodel,data["X_val"])
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   #IMPLEMENT THE FOLLOWING
   #meanfailuretosolveequation,_,_=measure_laplacian_failure(phimodel,data)
   print("\n\n")
   return phimodel,training_history, None

def load_nn_phimodel(free_coefficient,nlayer=3,nHidden=128,nEpochs=50,bSizes=[192,50000],stddev=0.1,lRate=0.001,set_weights_to_zero=False,set_weights_to_random=False,skip_measures=False):
   dirname = folder_name+ '/tetraquadric_pg_with_'+str(free_coefficient)
   name = 'phimodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(bSizes[1]) + 's' + str(nlayer) + 'x' +str(nHidden)
   print(dirname)
   print(name)
   
   data = np.load(os.path.join(dirname, 'dataset.npz'))
   data = convert_to_tensor_dict(data)
   BASIS = prepare_tf_basis(np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True))

   cb_list, cmetrics = getcallbacksandmetrics(data)

   #nlayer = 3
   #nHidden = 128
   act = 'gelu'
   #nEpochs = 100
   #bSizes = [192, 150000]
   alpha = [1., 1., 30., 1., 2.] # 1 AND 3??
   nfold = 3
   n_in = 2*8
   n_out = 1
   #lRate = 0.001


   ambient=tf.cast(BASIS['AMBIENT'],tf.int32)

   nfirstlayer= tf.reduce_prod((np.array(ambient)+determine_n_funcs)).numpy().item() if use_symmetry_reduced_TQ else tf.reduce_prod((np.array(ambient)+1)**2).numpy().item() 
   shapeofinternalnetwork=[nHidden]*nlayer
   shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1]

   print("network shape: " + str(shapeofnetwork))
   nn_phi = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,stddev=stddev,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_phi_zero =BiholoModelFuncGENERAL(shapeofnetwork,BASIS,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   print("nns made")

   datacasted=[data['X_val'],data['y_val']]

   #    nn_phi = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   #    nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   phimodel = PhiFSModel(nn_phi, BASIS, alpha=alpha)
   phimodelzero = PhiFSModel(nn_phi_zero, BASIS, alpha=alpha)
   #initialise weights
   phimodel(datacasted[0][0:1])
   phimodelzero(datacasted[0][0:1])

   #def print_sample_params(model, num_params=10):
   # print(model.layers)
   # for layer in model.layers:
   #     if len(layer.weights) > 0:  # Check if the layer has weights
   #         weights = layer.weights[0].numpy().flatten()
   #         biases = layer.weights[1].numpy().flatten() if len(layer.weights) > 1 else None

   #         print(f"Layer: {layer.name}")
   #         print("Weights sample:")
   #         print(weights[:num_params])

   #         if biases is not None:
   #             print("Biases sample:")
   #             print(biases[:num_params])

   #         print("\n")
   #     else:
   #         print(layer)
   #         print(layer.weights)
   #         print(dir(layer))
   #         print(layer.kernel)
    
   if set_weights_to_zero:
      print("SETTING WEIGHTS TO ZERO")
      training_history=0
      if skip_measures:
         return phimodelzero, training_history, None
   elif set_weights_to_random:
      print("SETTING WEIGHTS TO RANDOM")
      training_history=0
   else:
      #phimodel.model=tf.keras.layers.TFSMLayer(os.path.join(dirname,name),call_endpoint="serving_default")
      #phimodel.model=tf.keras.models.load_model(os.path.join(dirname, name) + ".keras")
      #print_sample_params(phimodel.model)
      phimodel.model.load_weights(os.path.join(dirname, name) + '.weights.h5')
      #print_sample_params(phimodel.model)
      training_history=np.load(os.path.join(dirname, 'trainingHistory-' + name +'.npz'),allow_pickle=True)['arr_0'].item()

   if skip_measures:
      return phimodel, training_history, None

   print("compiling")
   phimodel.compile(custom_metrics=cmetrics)
   phimodelzero.compile(custom_metrics=cmetrics)

   # compare validation loss before training for zero network and nonzero network
   #need to re-enable learning, in case there's been a problem:
   phimodel.learn_transition = True
   phimodelzero.learn_transition = True
   phimodel.learn_volk = True
   phimodelzero.learn_volk = True
   #phimodel.learn_ricci_val= True
   #phimodelzero.learn_ricci_val= True
   #print("phimodel1: " +str(phimodel(datacasted[0][0:2])))
   #print("phimodelzero: " +str(phimodelzero(datacasted[0][0:2])))

   #print("phimodel1: " +str(phimodel.model(datacasted[0][0:2])))
   #print("phimodelzero: " +str(phimodelzero.model(datacasted[0][0:2])))
   #metricsnames=phimodelzero.metrics_names
   #problem - metricsnames aren't defined unless model has been trained, not just evaluated? SOlution - return_dict
   valzero=phimodelzero.evaluate(datacasted[0],datacasted[1],return_dict=True)
   valtrained=phimodel.evaluate(datacasted[0],datacasted[1],return_dict=True)


   # phimodel.learn_ricci_val=False 
   # phimodelzero.learn_ricci_val=False 
   #valzero = {metricsnames[i]: valzero[i] for i in range(len(valzero))}
   #valtrained= {metricsnames[i]: valtrained[i] for i in range(len(valtrained))}

   #valzero = {key: value.numpy() for key, value in valzero.items()}
   #valtrained = {key: value.numpy() for key, value in valtrained.items()}

   #valtrained = {key: value.numpy() for key, value in valtrained.items()}

   phimodel.learn_transition = True
   phimodel.learn_volk = True

   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for final network: ")
   print(valtrained)
   print("ratio of trained to zero: " + str({key + " ratio": value/(valzero[key]+1e-8) for key, value in valtrained.items()}))
   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(phimodel,data["X_val"])
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   print("\n\n")
   #IMPLEMENT THE FOLLOWING
   #meanfailuretosolveequation,_,_=measure_laplacian_failure(phimodel,data)
   #print("\n\n")
   #print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   tf.keras.backend.clear_session()
   return phimodel,training_history, None

def generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM,number_pointsHYM,phimodel,force_generate=False,seed_set=0):

   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = folder_name+'/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = folder_name+'/tetraquadric_pg_with_'+str(free_coefficient)
   print("dirname for beta: " + dirnameHYM)

   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))
   
   coefficients=get_coefficients(free_coefficient)
   kmoduli = kmoduliTQ 
   ambient = np.array([1,1,1,1])
   
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
   pg._set_seed(seed_set)
   data=np.load(os.path.join(dirnameForMetric, 'dataset.npz'))

   if force_generate or (not os.path.exists(dirnameHYM)):
      print("Generating: forced? " + str(force_generate))
      kappaHYM = prepare_dataset_HYM(pg,data,number_pointsHYM, dirnameHYM,phimodel,linebundleforHYM,BASIS,normalize_to_vol_j=True);
   elif os.path.exists(dirnameHYM):
      try:
         print("loading prexisting dataset")
         data = np.load(os.path.join(dirnameHYM, 'dataset.npz'))
         length_total = len(data['X_train'])+len(data['X_val'])
         if length_total!=number_pointsHYM:
            print(f"wrong length {length_total}, want {number_pointsHYM} - generating anyway")
            kappaHYM = prepare_dataset_HYM(pg,data,number_pointsHYM, dirnameHYM,phimodel,linebundleforHYM,BASIS,normalize_to_vol_j=True);
      except:
         print("problem loading data - generating anyway")
         kappaHYM = prepare_dataset_HYM(pg,data,number_pointsHYM, dirnameHYM,phimodel,linebundleforHYM,BASIS,normalize_to_vol_j=True);
      
   

def getcallbacksandmetricsHYM(databeta):
   databeta_val_dict=dict(list(dict(databeta).items())[len(dict(databeta))//2:])
   tcb = TransitionCallback((databeta['X_val'], databeta['y_val']))
   lplcb = LaplacianCallback(databeta_val_dict)
   # lplcb = LaplacianCallback(data_val)
   cb_list = [lplcb,tcb]
   cmetrics = [TotalLoss(), LaplacianLoss(), TransitionLoss()]
   return cb_list, cmetrics

def convert_to_tensor_dict(data):
   return {
    key: tf.convert_to_tensor(value, dtype=complex_dtype) if value.dtype in [np.complex64, np.complex128] 
         else tf.convert_to_tensor(value, dtype=real_dtype) if value.dtype in [np.float32, np.float64]
         else tf.convert_to_tensor(value, dtype=value.dtype) if value.dtype == np.int32
         else value
    for key, value in data.items()
   }
   
def train_and_save_nn_HYM(free_coefficient,linebundleforHYM,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],stddev=0.1,lRate=0.001,use_zero_network=False,alpha=[1,1],load_network=False):
   
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = folder_name+'/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = folder_name+'/tetraquadric_pg_with_'+str(free_coefficient)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))


   databeta = np.load(os.path.join(dirnameHYM, 'dataset.npz'))
   databeta = convert_to_tensor_dict(databeta)
   databeta_train=tf.data.Dataset.from_tensor_slices(dict(list(dict(databeta).items())[:len(dict(databeta))//2]))
   databeta_val_dict=dict(list(dict(databeta).items())[len(dict(databeta))//2:])
   databeta_val=tf.data.Dataset.from_tensor_slices(databeta_val_dict)
   # batch_sizes=[64,10000]
   databeta_train=databeta_train.shuffle(buffer_size=1024).batch(bSizes[0],drop_remainder=True)
   datacasted=[databeta['X_val'],databeta['y_val']]



   weights = databeta['y_train'][:,0]
   sources = databeta['sources_train']
   integrated_source = tf.reduce_mean(weights*sources)
   integrated_abs_source = tf.reduce_mean(tf.math.abs(weights*sources))
   print("sources integrated ",integrated_source)
   print("abs sources integrated ",integrated_abs_source)

   cb_list, cmetrics = getcallbacksandmetricsHYM(databeta)

   #nlayer = 3
   #nHidden = 128
   act = 'gelu'
   #nEpochs = 30
   #bSizes = [192, 150000]
   #alpha = [1., 1.] # 1 AND 3??
   nfold = 3
   n_in = 2*8
   n_out = 1
   #lRate = 0.001
   name = 'betamodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(nlayer) + 'x' +str(nHidden)
   print("name: " + name)

   ambient=tf.cast(BASIS['AMBIENT'],tf.int32)

   nfirstlayer= tf.reduce_prod((np.array(ambient)+determine_n_funcs)).numpy().item() if use_symmetry_reduced_TQ else tf.reduce_prod((np.array(ambient)+1)**2).numpy().item() 
   shapeofinternalnetwork=[nHidden]*nlayer
   shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1]

   print("network shape: " + str(shapeofnetwork))
   #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=stddev)
   #nn_beta = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,stddev=stddev,use_zero_network=use_zero_network)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_beta_zero = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,use_zero_network=True)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #activ=tf.square
   activ=tfk.activations.gelu
   activ = tf.square
   if nHidden in [64,128,256] or nHidden<20 or nHidden==100:
       residual_Q = True
       load_func_HYM = BiholoModelFuncGENERALforHYMinv3
   elif nHidden in [65,129, 257]:
       residual_Q = False
       load_func_HYM = BiholoModelFuncGENERALforHYMinv4
   elif nHidden in [66, 130, 258, 513]:
       load_func_HYM = BiholoModelFuncGENERALforHYMinv3
      

   nn_beta = load_func_HYM(shapeofnetwork,BASIS,activation=activ,stddev=stddev,use_zero_network=use_zero_network,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_beta_zero = load_func_HYM(shapeofnetwork,BASIS,activation=activ,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #copie from phi above
   #nn_beta = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,stddev=stddev,use_zero_network=True)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_beta_zero =BiholoModelFuncGENERAL(shapeofnetwork,BASIS,use_zero_network=True)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)

   #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.2)
   #nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)#note we don't need a last bias (flat direction)
   #nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network,kernel_initializer=initializer)#note we don't need a last bias (flat direction)
   #nn_beta_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)#note we don't need a last bias (flat direction)
   
   betamodel= BetaModel(nn_beta,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
   if load_network:
      print("loading network")
      #betamodel.model=tf.keras.layers.TFSMLayer(os.path.join(dirnameHYM,name),call_endpoint="serving_default")
      #init vars so the load works
      betamodel(datacasted[0][0:1])
      betamodel.model.load_weights(os.path.join(dirnameHYM, name) + '.weights.h5')
      print("network loaded")

   betamodelzero= BetaModel(nn_beta_zero,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])

   #Note, currently running legacy due to ongoing tf issue with M1/M2. 
   #Use the commented line instead if not on an M1/M2 machine
   #opt = tfk.optimizers.Adam(learning_rate=lRate)
   #opt = tfk.optimizers.legacy.Adam(learning_rate=lRate)
   opt = create_adam_optimizer_with_decay(
    initial_learning_rate=lRate,
    nEpochs=nEpochs*(len(databeta['X_train'])//bSizes[0]),
    final_lr_factor=2  # This will decay to lRate/10
    )
   # compile so we can test on validation set before training
   betamodel.compile(custom_metrics=cmetrics)
   betamodelzero.compile(custom_metrics=cmetrics)
   
   #datacasted=[tf.cast(data['X_val'],real_dtype),tf.cast(data['y_val'],real_dtype)]
   print("testing zero and raw")
   # print("Printing validation dictionary keys and dtypes:")
   # for key, value in databeta_val_dict.items():
   #     print(f"{key}: {value.dtype}")
   valzero=betamodelzero.test_step(databeta_val_dict)
   valraw=betamodel.test_step(databeta_val_dict)
   valzero = {key: value.numpy() for key, value in valzero.items()}
   valraw = {key: value.numpy() for key, value in valraw.items()}
   print(valzero)
   print(valraw)
   print("tested zero and raw")
   
   training_historyBeta={'transition_loss': [10**(-8)],'laplacian_loss': [1000000000000000]}
   i=0
   newLR=lRate
   #while (training_historyBeta['transition_loss'][-1]<10**(-5)) or (training_historyBeta['laplacian_loss'][-1]>1.):
   # continue looping if >10 or is nan
   #while (training_historyBeta['laplacian_loss'][-1]>1.9) or (np.isnan( training_historyBeta['laplacian_loss'][-1])):
   while i==0 or (i<=1 and (np.isnan( training_historyBeta['laplacian_loss'][-1]))):
      print("trying i "+str(i))
      if i >0:

         print('trying again laplacian_loss too big')
         #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.2)
         #nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network,kernel_initializer=initializer)#note we don't need a last bias (flat direction)
         #nn_beta = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,stddev=stddev,use_zero_network=use_zero_network)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
         #nn_beta = BiholoModelFuncGENERALforHYMinv2(shapeofnetwork,BASIS,activation=tfk.activations.gelu,stddev=stddev,use_zero_network=use_zero_network)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
         seed = 3 + i
         tf.random.set_seed(seed)
         nn_beta = load_func_HYM(shapeofnetwork,BASIS,activation=activ,stddev=stddev,use_zero_network=use_zero_network,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
         #nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)#note we don't need a last bias (flat direction)
         if newLR>0.0002:
             newLR=newLR/2
             print("new LR " + str(newLR))
         #opt = tfk.optimizers.legacy.Adam(learning_rate=newLR)
         opt = create_adam_optimizer_with_decay(
                 initial_learning_rate=newLR,
                 #nEpochs=nEpochs,
                 #nEpochs=nEpochs*(len(data['X_train'])//bSizes[0]),
                 nEpochs=nEpochs*(len(databeta['X_train'])//bSizes[0]),
                 final_lr_factor=2  # This will decay to lRate/10
                 )
         #betamodel= BetaModel(nn_beta,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
         betamodel.model=nn_beta
         #cb_list, cmetrics = getcallbacksandmetricsHYM(databeta)
         betamodel.compile(custom_metrics=cmetrics)
         betamodel(datacasted[0][0:1])
      betamodel, training_historyBeta= train_modelbeta(betamodel, databeta_train, optimizer=opt, epochs=nEpochs, batch_sizes=bSizes, 
                                        verbose=1, custom_metrics=cmetrics, callbacks=cb_list)
      i+=1
   print("finished training\n")
   #betamodel.model.export(os.path.join(dirnameHYM, name))
   #betamodel.model.save(os.path.join(dirnameHYM, name) + '.keras')
   betamodel.model.save_weights(os.path.join(dirnameHYM, name) + '.weights.h5')
   np.savez_compressed(os.path.join(dirnameHYM, 'trainingHistory-' + name),training_historyBeta)
   valfinal =betamodel.test_step(databeta_val_dict)
   valfinal = {key: value.numpy() for key, value in valfinal.items()}
   #return training_historyBeta
   #now print the initial losses and final losses for each metric, by taking the first element of each key in the dictionary
   #first_metrics = {key: value[0] for key, value in training_historyBeta.items()}
   #last_metrics = {key: value[-1] for key, value in training_historyBeta.items()}

   #print("initial losses")
   #print(first_metrics)
   #print("final losses")
   #print(last_metrics)


   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for raw network: ")
   print(valraw)
   print("validation loss for final network: ")
   print(valfinal)
   print("ratio of final to zero: " + str({key + " ratio": value/(valzero[key]+1e-8) for key, value in valfinal.items()}))
   print("ratio of final to raw: " + str({key + " ratio": value/(valraw[key]+1e-8) for key, value in valfinal.items()}))


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(betamodel,databeta["X_val"])
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))


   import time
   start=time.time()
   meanfailuretosolveequation,_,_=HYM_measure_val(betamodel,databeta)
   #meanfailuretosolveequation= batch_process_helper_func(
   #     tf.function(lambda x,y,z,w,a: tf.expand_dims(HYM_measure_val_for_batching(betamodel,x,y,z,w,a),axis=0)),
   #     (databeta['X_val'],databeta['y_val'],databeta['val_pullbacks'],databeta['inv_mets_val'],databeta['sources_val']),
   #     batch_indices=(0,1,2,3,4),
   #     batch_size=50
   # )
   meanfailuretosolveequation=tf.reduce_mean(meanfailuretosolveequation).numpy()
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("time to do that: ",time.time()-start)
   print("\n\n")

   tf.keras.backend.clear_session()
   return betamodel,training_historyBeta, meanfailuretosolveequation

def load_nn_HYM(free_coefficient,linebundleforHYM,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],stddev=0.1,lRate=0.001,set_weights_to_zero=False,set_weights_to_random=False,skip_measures=False):
   
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = folder_name+'/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = folder_name+'/tetraquadric_pg_with_'+str(free_coefficient)
   name = 'betamodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(nlayer) + 'x' +str(nHidden)
   print("name: " + name)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))


   databeta = np.load(os.path.join(dirnameHYM, 'dataset.npz'))
   databeta = convert_to_tensor_dict(databeta)
   databeta_train=tf.data.Dataset.from_tensor_slices(dict(list(dict(databeta).items())[:len(dict(databeta))//2]))
   databeta_val_dict=dict(list(dict(databeta).items())[len(dict(databeta))//2:])
   databeta_val=tf.data.Dataset.from_tensor_slices(databeta_val_dict)
   # batch_sizes=[64,10000]
   databeta_train=databeta_train.shuffle(buffer_size=1024).batch(bSizes[0],drop_remainder=True)
   datacasted=[databeta['X_val'],databeta['y_val']]

   cb_list, cmetrics = getcallbacksandmetricsHYM(databeta)

   #nlayer = 3
   #nHidden = 128
   act = 'gelu'
   #nEpochs = 30
   #bSizes = [192, 150000]
   alpha = [1., 1.] # 1 AND 3??
   nfold = 3
   n_in = 2*8
   n_out = 1
   #lRate = 0.001
   ambient=tf.cast(BASIS['AMBIENT'],tf.int32)

   nfirstlayer= tf.reduce_prod((np.array(ambient)+determine_n_funcs)).numpy().item() if use_symmetry_reduced_TQ else tf.reduce_prod((np.array(ambient)+1)**2).numpy().item() 
   shapeofinternalnetwork=[nHidden]*nlayer
   shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[1]

   print("network shape: " + str(shapeofnetwork))
   #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=stddev)
   #nn_beta = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,stddev=stddev,use_zero_network=True)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_beta_zero = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,use_zero_network=True)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   activ=tf.square
   load_func_HYM = BiholoModelFuncGENERALforHYMinv3


   activ=tfk.activations.gelu
   activ = tf.square
   if nHidden in [64,128,256] or nHidden<20 or nHidden==100:
       residual_Q = True
       load_func_HYM = BiholoModelFuncGENERALforHYMinv3
   elif nHidden in [65,129, 257]:
       residual_Q = False
       load_func_HYM = BiholoModelFuncGENERALforHYMinv4
   elif nHidden in [66, 130, 258, 513]:
       load_func_HYM = BiholoModelFuncGENERALforHYMinv3
   nn_beta = load_func_HYM(shapeofnetwork,BASIS,activation=activ,stddev=stddev,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_beta_zero = load_func_HYM(shapeofnetwork,BASIS,activation=activ,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #copie from phi above
   #nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   
   #nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)#note we don't need a last bias (flat direction)
   #nn_beta_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)#note we don't need a last bias (flat direction)
   
   betamodel= BetaModel(nn_beta,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
   betamodelzero= BetaModel(nn_beta_zero,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
   print("example data:")
   print("betamodel: " + str(betamodel(datacasted[0][0:1])))
   print("betamodelzero: " + str(betamodelzero(datacasted[0][0:1])))


   if set_weights_to_zero:
      print("RETURNING ZERO NETWORK")
      training_historyBeta=0
      if skip_measures:
         return betamodelzero, training_historyBeta, None
   elif set_weights_to_random:
      print("RETURNING RANDOM NETWORK")
      training_historyBeta=0
   else:
      #betamodel.model=tf.keras.layers.TFSMLayer(os.path.join(dirnameHYM,name),call_endpoint="serving_default")
      #betamodel.model=tf.keras.models.load_model(os.path.join(dirnameHYM, name) + ".keras")
      #init vars so the load works
      print(betamodel(databeta_val_dict['X_val'][0:1]))
      print(betamodelzero(databeta_val_dict['X_val'][0:1]))
      betamodel.model.load_weights(os.path.join(dirnameHYM, name) + '.weights.h5')
      training_historyBeta=np.load(os.path.join(dirnameHYM, 'trainingHistory-' + name +'.npz'),allow_pickle=True)['arr_0'].item()
      print("second print",betamodel(databeta_val_dict['X_val'][0:1]))

   if skip_measures:
       return betamodel,training_historyBeta, None

   betamodel.compile(custom_metrics=cmetrics)
   betamodelzero.compile(custom_metrics=cmetrics)
   
   #valzero=betamodelzero.evaluate(databeta_val_dict,return_dict=True)
   #valtrained=betamodel.evaluate(databeta_val_dict,return_dict=True)
   ##valzero = {key: value.numpy() for key, value in valzero.items()}
   ##valtrained= {key: value.numpy() for key, value in valtrained.items()}

   valzero=betamodelzero.test_step(databeta_val_dict)
   valtrained=betamodel.test_step(databeta_val_dict)
   valzero = {key: float(value.numpy()) for key, value in valzero.items()}
   valtrained= {key: float(value.numpy()) for key, value in valtrained.items()}

   #metricsnames=betamodel.metrics_names

   #valzero = {metricsnames[i]: valzero[i] for i in range(len(valzero))}
   #valtrained= {metricsnames[i]: valtrained[i] for i in range(len(valtrained))}


   

   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for trained network: ")
   print(valtrained)
   print("ratio of trained to zero: " + str({key + " ratio": value/(valzero[key]+1e-8) for key, value in valtrained.items()}))


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(betamodel,databeta["X_val"])
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   import time
   start=time.time()
   #meanfailuretosolveequation,_,_=HYM_measure_val(betamodel,databeta)
   meanfailuretosolveequation= batch_process_helper_func(
        tf.function(lambda x,y,z,w,a: tf.expand_dims(HYM_measure_val_for_batching(betamodel,x,y,z,w,a),axis=0)),
        (databeta['X_val'],databeta['y_val'],databeta['val_pullbacks'],databeta['inv_mets_val'],databeta['sources_val']),
        batch_indices=(0,1,2,3,4),
        batch_size=50
    )
   meanfailuretosolveequation=tf.reduce_mean(meanfailuretosolveequation)

   
   
   
   
   
   
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("time to do that: ",time.time()-start)
   print("\n\n")
   return betamodel,training_historyBeta, meanfailuretosolveequation


def generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM,functionforbaseharmonicform_jbar,phimodel,betamodel,number_pointsHarmonic,force_generate=False,seed_set=0):
   # get names
   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameForMetric = folder_name+'/tetraquadric_pg_with_'+str(free_coefficient)
   dirnameHYM = folder_name+'/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameHarmonic = folder_name+'/tetraquadricHarmonicH_pg'+str(free_coefficient)+'forLB_'+lbstring+nameOfBaseHF
   print("dirname for harmonic form: " + dirnameHarmonic)

   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))
   
   coefficients=get_coefficients(free_coefficient)
   kmoduli = kmoduliTQ
   ambient = np.array([1,1,1,1])
   
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
   pg._set_seed(seed_set)
   data=np.load(os.path.join(dirnameForMetric, 'dataset.npz'))

   if force_generate or (not os.path.exists(dirnameHarmonic)):
      print("Generating: forced? " + str(force_generate))
      print("N_pointsHF: "+ str(number_pointsHarmonic))
      kappaHarmonic=prepare_dataset_HarmonicForm(pg,data,number_pointsHarmonic,dirnameHarmonic,phimodel,linebundleforHYM,BASIS,functionforbaseharmonicform_jbar,betamodel)
   elif os.path.exists(dirnameHarmonic):
      try:
         print("loading prexisting dataset")
         data = np.load(os.path.join(dirnameHarmonic, 'dataset.npz'))
         print(phimodel.BASIS['KMODULI'])

         #print("CHECKING PHIMODEL:")
         #print(data['X_train'][0:1])
         #print(phimodel(data['X_train'][0:1]))
         #print(phimodel.fubini_study_pb(data['X_train'][0:1]))
         #print(tf.linalg.inv(data['inv_mets_train'][0]))
         #print("DONE")

         length_total = len(data['X_train'])+len(data['X_val'])
         if length_total!=number_pointsHarmonic:
            print(f"wrong length {length_total}, want {number_pointsHarmonic} - generating anyway")
            kappaHarmonic=prepare_dataset_HarmonicForm(pg,data,number_pointsHarmonic,dirnameHarmonic,phimodel,linebundleforHYM,BASIS,functionforbaseharmonicform_jbar,betamodel)
      except:
         print("problem loading data - generating anyway")
         kappaHarmonic=prepare_dataset_HarmonicForm(pg,data,number_pointsHarmonic,dirnameHarmonic,phimodel,linebundleforHYM,BASIS,functionforbaseharmonicform_jbar,betamodel)
   

def getcallbacksandmetricsHF(dataHF):
   dataHF_val_dict=dict(list(dict(dataHF).items())[len(dict(dataHF))//2:])

   tcbHF = TransitionCallback((dataHF['X_val'], dataHF['y_val']))
   lplcbHF = LaplacianCallback(dataHF_val_dict)
   NaNcbHF = NaNCallback()
   # lplcb = LaplacianCallback(dataHF_val)
   cb_listHF = [lplcbHF,tcbHF, NaNcbHF]
   cmetricsHF = [TotalLoss(), LaplacianLoss(), TransitionLoss()]
   return cb_listHF, cmetricsHF

   
def train_and_save_nn_HF(free_coefficient,linebundleforHYM,betamodel,metric_model,functionforbaseharmonicform_jbar,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],lRate=0.001,alpha=[1,500],use_zero_network=False,load_network=False,stddev=0.05,final_layer_scale=1.0,norm_momentum=0.999):
   
   perm= tracker.SummaryTracker()
   #print("perm1")
   #print(perm.print_diff())

   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = folder_name+'/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = folder_name+'/tetraquadric_pg_with_'+str(free_coefficient)
   dirnameHarmonic = folder_name+'/tetraquadricHarmonicH_pg'+str(free_coefficient)+'forLB_'+lbstring+nameOfBaseHF
   name = 'HFmodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+  str(nlayer) + 'x' +str(nHidden)
   print("dirname: " + dirnameHarmonic)
   print("name: " + name)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))

   #print("perm2")
   #print(perm.print_diff())
   dataHF = np.load(os.path.join(dirnameHarmonic, 'dataset.npz'))
   dataHF = convert_to_tensor_dict(dataHF)
   dataHF_train=tf.data.Dataset.from_tensor_slices(dict(list(dict(dataHF).items())[:len(dict(dataHF))//2]))
   dataHF_val_dict=dict(list(dict(dataHF).items())[len(dict(dataHF))//2:])
   dataHF_val_dict = {key: tf.convert_to_tensor(value) for key, value in dataHF_val_dict.items()}
   dataHF_val=tf.data.Dataset.from_tensor_slices(dataHF_val_dict)
   # batch_sizes=[64,10000]
   dataHF_train=dataHF_train.shuffle(buffer_size=1024).batch(bSizes[0],drop_remainder=True)
   datacasted=[dataHF['X_val'],dataHF['y_val']]




   #print("perm3")
   #print(perm.print_diff())
   cb_listHF, cmetricsHF = getcallbacksandmetricsHF(dataHF)

   #nlayer = 3
   #nHidden = 128
   act = 'gelu'
   #nEpochs = 30
   #bSizes = [192, 150000]
   #alpha = [1, 10.] # 1 AND 3?
   nfold = 3
   n_in = 2*8
   n_outcomplex = 1
   n_outreal= n_outcomplex*2 
   #lRate = 0.001

   nsections=np.prod(np.abs(linebundleforHYM)+1)
   ambient=tf.cast(BASIS['AMBIENT'],tf.int32)

   nfirstlayer= tf.reduce_prod((np.array(ambient)+determine_n_funcs)).numpy().item() if use_symmetry_reduced_TQ else tf.reduce_prod((np.array(ambient)+1)**2).numpy().item() 
   shapeofinternalnetwork=[nHidden]*nlayer
   shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[2*nsections]
   print(shapeofnetwork)

   # need a last bias layer due to transition!
   #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.3)
   #nn_HF = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,kernel_initializer=initializer)
   #nn_HF_zero = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,use_zero_network=True)
   #activ=tf.square
   #stddev=stddev
   #nn_HF = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,use_zero_network=use_zero_network,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_HF_zero  = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,use_zero_network=True,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   activ=tf.keras.activations.gelu
   #if True:# np.sum(np.abs(linebundleforHYM -np.array([0,-2,1,3])))<1e-8 or np.sum(np.abs(linebundleforHYM -np.array([0,0,1,-3])))<1e-8:
   #if np.sum(np.abs(linebundleforHYM -np.array([0,-2,1,3])))<1e-8: #or np.sum(np.abs(linebundleforHYM -np.array([0,0,1,-3])))<1e-8:
   #    load_func=BiholoModelFuncGENERALforSigma2_m13_SECOND
   #    activ = tf.keras.activations.gelu
   #elif np.sum(np.abs(linebundleforHYM -np.array([0,0,1,-3])))<1e-8: #or np.sum(np.abs(linebundleforHYM -np.array([0,0,1,-3])))<1e-8:
   #    load_func=BiholoModelFuncGENERALforSigma2_m13_THIRD
   #    activ = tf.keras.activations.gelu
   #else:
   #activ=tf.square
   activ = tf.keras.activations.gelu
   activ = tf.square
   #load_func = BiholoModelFuncGENERALforSigma2_m13
   #load_func = BiholoModelFuncGENERALforSigmaWNorm
   if (nHidden ==64) or (nHidden ==128) or nHidden<20 or nHidden==100:
       #load_func = BiholoBadSectionModel
       load_func =BiholoModelFuncGENERALforSigma2_m13
       #load_func = BiholoModelFuncGENERALforSigmaWNorm_no_log
   elif (nHidden ==65) or (nHidden ==129):
       #load_func = BiholoModelFuncGENERALforSigmaWNorm
       load_func = BiholoModelFuncGENERALforSigma2_m13
   #if (nHidden ==66) or (nHidden ==130) or :
   if nHidden in [66,130,250,430,200]:
        load_func = BiholoModelFuncGENERALforSigmaWNorm_no_log_residual 
   if (nHidden==67) or (nHidden==131):
        load_func = BiholoModelFuncGENERALforSigmaWNorm
   #if np.sum(np.abs(linebundleforHYM -np.array([0,0,1,-3])))<1e-8:
   #    k_phi_here = np.array([0,0,0,1])
   k_phi_here = np.array([0,0,0,0])
   nn_HF = load_func(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=k_phi_here,activation=activ,stddev=stddev,use_zero_network=use_zero_network,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ,norm_momentum=norm_momentum)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_HF_zero  = load_func(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=k_phi_here,activation=activ,stddev=stddev,use_zero_network=True,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ,norm_momentum=norm_momentum)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
  
   print('network arch:',load_func)
   print("activation: ",activ)
   HFmodel = HarmonicFormModel(nn_HF,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
   HFmodelzero = HarmonicFormModel(nn_HF_zero,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
   if load_network:
      print("loading network from weights")
      #HFmodel.model=tf.keras.layers.TFSMLayer(os.path.join(dirnameHarmonic,name),call_endpoint="serving_default")
      #init variables
      HFmodel(dataHF_val_dict['X_val'][0:1])
      HFmodel.model.load_weights(os.path.join(dirnameHarmonic, name) + ".weights.h5")
      #HFmodel.model=tf.keras.models.load_model(os.path.join(dirnameHarmonic, name) + ".keras")
      print("network loaded")

   #print("perm5")
   #print(perm.print_diff())

   #Note, currently running legacy due to ongoing tf issue with M1/M2. 
   #Use the commented line instead if not on an M1/M2 machine
   #opt = tfk.optimizers.Adam(learning_rate=lRate)
   #opt = tfk.optimizers.legacy.Adam(learning_rate=lRate)
   opt = create_adam_optimizer_with_decay(
           initial_learning_rate=lRate,
           nEpochs=nEpochs*(len(dataHF['X_train'])//bSizes[0]),
           final_lr_factor=2#2  # This will decay to lRate/10
           )
   # compile so we can test on validation set before training

   print('testing computation')
   print(HFmodel(datacasted[0][0:1]))
   print(HFmodelzero(datacasted[0][0:1]))

   HFmodel.compile(custom_metrics=cmetricsHF)
   HFmodelzero.compile(custom_metrics=cmetricsHF)

   pts_check = dataHF_val_dict['X_val'][0:1000]
   pullbacks_check = dataHF_val_dict['val_pullbacks'][0:1000]
   #check_vals = closure_check(pts_check,HFmodel.functionforbaseharmonicform_jbar, HFmodel, pullbacks_check)
   #check_valszero = closure_check(pts_check,HFmodelzero.functionforbaseharmonicform_jbar, HFmodelzero, pullbacks_check)
   #print("check1:",tf.reduce_mean(tf.math.abs(check_vals)))
   #print("check2:",tf.reduce_mean(tf.math.abs(check_valszero)))
   #print("perm6")
   #print(perm.print_diff())

   #only use 100 entries in dataHF_val_dict
   #dataHF_val_dict_short={key: value[:100] for key, value in dataHF_val_dict.items()}
   print("testing zero and raw")
   valzero=HFmodelzero.test_step(dataHF_val_dict)
   valraw=HFmodel.test_step(dataHF_val_dict)
   valzero = {key: float(value.numpy()) for key, value in valzero.items()}
   valraw = {key: float(value.numpy()) for key, value in valraw.items()}
   print("tested zero and raw")
   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for raw network: ")
   print(valraw)

   #print("perm7")
   #print(perm.print_diff())




   training_historyHF={'transition_loss': [10**(-8)], 'laplacian_loss':[10.**10]}
   i=0
   newLR = lRate
   # continue looping if >10 or is nan
   #while i==0:#(training_historyHF['laplacian_loss'][-1]>0.9) or (np.isnan( training_historyHF['laplacian_loss'][-1])):
   while i==0 or (i<=3 and (np.isnan( training_historyHF['laplacian_loss'][-1]))):
      if i >0:
         print('trying again, as laplacian was too large or nan or something')
         ##initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.3)
         ##nn_HF = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,kernel_initializer=initializer)
         #del nn_HF,HFmodel
         #del opt,cb_listHF,cmetricsHF
         #opt = 0
         #HFmodel=0
         #nn_HF=0
         #tf.keras.backend.clear_session()
         #nn_HF = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,use_zero_network=use_zero_network,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
         #nn_HF = BiholoModelFuncGENERALforSigma2(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,use_zero_network=use_zero_network,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)

         seed = 3 + i
         tf.random.set_seed(seed)
         nn_HF = load_func(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=k_phi_here,activation=activ,stddev=stddev,use_zero_network=use_zero_network,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ,norm_momentum=norm_momentum)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
         #HFmodel = HarmonicFormModel(nn_HF,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
         if newLR>0.0002:
             newLR=newLR/2
             print("new LR " + str(newLR))
         opt = create_adam_optimizer_with_decay(
                 initial_learning_rate=newLR,
                 nEpochs=nEpochs*(len(dataHF['X_train'])//bSizes[0]),
                 final_lr_factor=2  # This will decay to lRate/10
                 )
         #cb_listHF, cmetricsHF = getcallbacksandmetricsHF(dataHF)
         #HFmodel2.compile(custom_metrics=cmetricsHF)
         ##HFmodel=HFmodel2
         ##opt=opt2
         #HFmodel2, training_historyHF= train_modelHF(HFmodel2, dataHF_train, optimizer=opt2, epochs=nEpochs, batch_sizes=bSizes, 
                 #verbose=1, custom_metrics=cmetricsHF, callbacks=cb_listHF)
         HFmodel.model = nn_HF
         valraw=HFmodel.test_step(dataHF_val_dict)
         valraw = {key: float(value.numpy()) for key, value in valraw.items()}
      HFmodel, training_historyHF= train_modelHF(HFmodel, dataHF_train, optimizer=opt, epochs=nEpochs, batch_sizes=bSizes, 
                                       verbose=1, custom_metrics=cmetricsHF, callbacks=cb_listHF)
      i+=1 


   #print("perm8")
   #print(perm.print_diff())

   tf.print("finished training\n")
   print("finished training\n")
   #test model so it can be exported
   #testdata=dataHF_val_dict['X_val'][0:2]
   #print(testdata)
   #tf.print(testdata)
   #tf.print(HFmodel.model(testdata))
   #tf.print(HFmodel(testdata))
   #print(HFmodel.model(testdata))
   #print(HFmodel(testdata))
   #result = HFmodel.model(testdata)
   #print("Model output shape:", result.shape)


   #HFmodel.model.export(os.path.join(dirnameHarmonic, name))

   HFmodel.model.save_weights(os.path.join(dirnameHarmonic, name) + ".weights.h5")
   #HFmodel.model.save(os.path.join(dirnameHarmonic, name) + ".keras")
   print("saved weights")
   np.savez_compressed(os.path.join(dirnameHarmonic, 'trainingHistory-' + name),training_historyHF)
   tf.print("saved training\n")
   print("saved training\n")
   import gc
   gc.collect()
   tf.keras.backend.clear_session()

   valfinal =HFmodel.test_step(dataHF_val_dict)
   valfinal = {key: float(value.numpy()) for key, value in valfinal.items()}

   #print("perm9")
   #print(perm.print_diff())

   #return training_historyBeta
   #now print the initial losses and final losses for each metric, by taking the first element of each key in the dictionary
   #first_metrics = {key: value[0] for key, value in training_historyHF.items()}
   #last_metrics = {key: value[-1] for key, value in training_historyHF.items()}

   #print("initial losses")
   #print(first_metrics)
   #print("final losses")
   #print(last_metrics)
   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for raw network: ")
   print(valraw)
   print("validation loss for final network: ")
   print(valfinal)
   print("ratio of final to zero: " + str({key + " ratio": value/(valzero[key]+1e-8) for key, value in valfinal.items()}))
   print("ratio of final to raw: " + str({key + " ratio": value/(valraw[key]+1e-8) for key, value in valfinal.items()}))
   import time
   start = time.time()
   print("start time:", time.strftime("%H:%M:%S", time.localtime()))
   check_vals_again = closure_check(pts_check,HFmodel.functionforbaseharmonicform_jbar, HFmodel, pullbacks_check)
   print(time.time()-start)
   print("check1 again:",tf.reduce_mean(tf.math.abs(check_vals_again)))


   #print("perm10")
   #print(perm.print_diff())
   print("-----CHECKS------")
   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure_section(HFmodel,dataHF["X_val"], weights=dataHF["y_val"][:, -2])
   print("average transition discrepancy in standard deviations (note, underestimate as our std.dev. ignores variation in phase): " + str(averagediscrepancyinstdevs))
   transition_loss = compute_transition_loss_for_corrected_HF_model(HFmodel,dataHF["X_val"], weights=dataHF["y_val"][:, -2])
   print("transition loss: " + str(tf.reduce_mean(transition_loss)))
   transition_loss_zero = compute_transition_loss_for_corrected_HF_model(HFmodelzero,dataHF["X_val"], weights=dataHF["y_val"][:, -2] )
   print("transition loss for zero network: " + str(tf.reduce_mean(transition_loss_zero)))
   
   transition_loss_for_uncorrected_HF = compute_transition_loss_for_uncorrected_HF_model(HFmodel,dataHF["X_val"], weights=dataHF["y_val"][:, -2])
   print("transition loss for uncorrected HF: " + str(tf.reduce_mean(transition_loss_for_uncorrected_HF)))
   transition_loss_for_uncorrected_HF_zero = compute_transition_loss_for_uncorrected_HF_model(HFmodelzero,dataHF["X_val"], weights=dataHF["y_val"][:, -2])
   print("transition loss for uncorrected HF zero network: " + str(tf.reduce_mean(transition_loss_for_uncorrected_HF_zero)))




   import time
   start=time.time()
   meanfailuretosolveequation,_,_=HYM_measure_val_with_H(HFmodel,dataHF)
   # meanfailuretosolveequation= batch_process_helper_func(
   #      tf.function(lambda x,y,z,w: tf.expand_dims(HYM_measure_val_with_H_for_batching(HFmodel,x,y,z,w),axis=0)),
   #      (dataHF['X_val'],dataHF['y_val'],dataHF['val_pullbacks'],dataHF['inv_mets_val']),
   #      batch_indices=(0,1,2,3),
   #      batch_size=50
   #  )
   meanfailuretosolveequation=tf.reduce_mean(meanfailuretosolveequation).numpy().item()


   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("time to do that: ",time.time()-start)
   print("\n\n")
   TrainedDivTrained, avgavagTrainedDivTrained, TrainedDivFS, avgavagTrainedDivFS, FS_DivFS, avgavagFS_DivFS = HYM_measure_val_with_H_relative_to_norm(HFmodel,dataHF_val_dict,betamodel,metric_model)
   print("trained coclosure divided by norm of v: " + str(TrainedDivTrained))
   print("avg/avg trained coclosure divided by norm of trained v: " + str(avgavagTrainedDivTrained))
   print("trained coclosure divided by norm of v_FS: " + str(TrainedDivFS))
   print("avg/avg trained coclosure divided by norm of v_FS: " + str(avgavagTrainedDivFS))
   print("FS coclosure divided by norm of v_FS: " + str(FS_DivFS))
   print("avg/avg FS coclosure divided by norm of v_FS: " + str(avgavagFS_DivFS))

   tf.keras.backend.clear_session()
   del dataHF, dataHF_train, dataHF_val_dict,  dataHF_val,valfinal,valraw,valzero
   #print("perm11")
   #print(perm.print_diff())
   return HFmodel,training_historyHF,meanfailuretosolveequation



def load_nn_HF(free_coefficient,linebundleforHYM,betamodel,metric_model,functionforbaseharmonicform_jbar,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],lRate=0.001,alpha=[1,1],set_weights_to_zero=False,set_weights_to_random=False,final_layer_scale=1.0,skip_measures=False,norm_momentum=0.999):
   
   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = folder_name+'/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = folder_name+'/tetraquadric_pg_with_'+str(free_coefficient)
   dirnameHarmonic = folder_name+'/tetraquadricHarmonicH_pg'+str(free_coefficient)+'forLB_'+lbstring+nameOfBaseHF
   name = 'HFmodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+  str(nlayer) + 'x' +str(nHidden)
   print("dirname: " + dirnameHarmonic)
   print("name: " + name)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))

   dataHF = np.load(os.path.join(dirnameHarmonic, 'dataset.npz'))
   dataHF = convert_to_tensor_dict(dataHF)
   dataHF_train=tf.data.Dataset.from_tensor_slices(dict(list(dict(dataHF).items())[:len(dict(dataHF))//2]))
   dataHF_val_dict=dict(list(dict(dataHF).items())[len(dict(dataHF))//2:])
   dataHF_val_dict = {key: tf.convert_to_tensor(value) for key, value in dataHF_val_dict.items()}
   dataHF_val=tf.data.Dataset.from_tensor_slices(dataHF_val_dict)
   # batch_sizes=[64,10000]
   dataHF_train=dataHF_train.shuffle(buffer_size=1024).batch(bSizes[0],drop_remainder=True)




   cb_listHF, cmetricsHF = getcallbacksandmetricsHF(dataHF)

   #nlayer = 3
   #nHidden = 128
   act = 'gelu'
   #nEpochs = 30
   #bSizes = [192, 150000]
   #alpha = [100., 1.] # 1 AND 3??
   nfold = 3
   n_in = 2*8
   n_outcomplex = 1
   n_outreal= n_outcomplex*2 
   #lRate = 0.001

   nsections=np.prod(np.abs(linebundleforHYM)+1)
   ambient=tf.cast(BASIS['AMBIENT'],tf.int32)

   nfirstlayer= tf.reduce_prod((np.array(ambient)+determine_n_funcs)).numpy().item() if use_symmetry_reduced_TQ else tf.reduce_prod((np.array(ambient)+1)**2).numpy().item() 
   shapeofinternalnetwork=[nHidden]*nlayer
   shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[2*nsections]
   print(shapeofnetwork)
   # need a last bias layer due to transition!
   #nn_HF = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,use_zero_network=True,kernel_initializer=initializer)
   #nn_HF_zero = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,use_zero_network=True)
   activ=tf.square
   stddev=0.1 # THIS DOESN'T DO ANYTHING!!
   #nn_HF = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,k_phi=np.array([0,0,0,0]),nsections=nsections,activation=activ,stddev=stddev,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_HF_zero  = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,k_phi=np.array([0,0,0,0]),nsections=nsections,activation=activ,stddev=stddev,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   activ=tf.square
   #activ=tfk.activations.gelu
   #if np.sum(np.abs(linebundleforHYM -np.array([0,-2,1,3])))<1e-8:
   activ=tf.keras.activations.gelu
   #else:
   #    activ=tf.square


   #nn_HF = BiholoModelFuncGENERALforSigma2(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,final_layer_scale=final_layer_scale,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_HF_zero  = BiholoModelFuncGENERALforSigma2(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,k_phi=np.array([0,0,0,0]),nsections=nsections,activation=activ,stddev=stddev,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #activ=tf.keras.activations.gelu
   #if np.sum(np.abs(linebundleforHYM -np.array([0,-2,1,3])))<1e-8:
   #    load_func=BiholoModelFuncGENERALforSigma2_m13_SECOND
   #    activ = tf.keras.activations.gelu
   #else:
   #    activ=tf.square
   #    load_func = BiholoModelFuncGENERALforSigma2_m13
   #if np.sum(np.abs(linebundleforHYM -np.array([0,-2,1,3])))<1e-8: #or np.sum(np.abs(linebundleforHYM -np.array([0,0,1,-3])))<1e-8:
   #    load_func=BiholoModelFuncGENERALforSigma2_m13_SECOND
   #    activ = tf.keras.activations.gelu
   #else:
   #activ=tf.square
   activ = tf.keras.activations.gelu
   activ = tf.square
   #load_func = BiholoModelFuncGENERALforSigma2_m13
   #load_func = BiholoModelFuncGENERALforSigmaWNorm_no_log

   load_func = BiholoModelFuncGENERALforSigma2_m13

   print("network arch:",load_func)
   print("activation: ",activ)
   nn_HF = load_func(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,use_zero_network=False,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ,norm_momentum=norm_momentum)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_HF_zero  = load_func(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,use_zero_network=True,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ,norm_momentum=norm_momentum)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   
   HFmodel = HarmonicFormModel(nn_HF,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
   HFmodelzero = HarmonicFormModel(nn_HF_zero,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
   print("example data:")
   print("HFmodel: " + str(HFmodel(dataHF_val_dict['X_val'][0:1])))
   print("HFmodelzero: " + str(HFmodelzero(dataHF_val_dict['X_val'][0:1])))

   if set_weights_to_zero:
      training_historyHF=0
      if skip_measures:
         print("RETURNING ZERO NETWORK")
         return HFmodelzero, training_historyHF, None
      else:
         print("USING ZERO NETWORK")
   elif set_weights_to_random:
      training_historyHF=0
      if skip_measures:
         print("RETURNING RANDOM NETWORK")
         return HFmodel, training_historyHF, None
      else:
         print("USING RANDOM NETWORK")
   else:
      #print(HFmodel.model.weights[0])
      #HFmodel.model=tf.keras.layers.TFSMLayer(os.path.join(dirnameHarmonic,name),call_endpoint="serving_default")
      #HFmodel.model=tf.keras.models.load_model(os.path.join(dirnameHarmonic, name) + ".keras")
      #init variables
      HFmodel(dataHF_val_dict['X_val'][0:1])
      HFmodel.model.load_weights(os.path.join(dirnameHarmonic, name) + ".weights.h5")
      #print(HFmodel.model.weights[0])
      training_historyHF=np.load(os.path.join(dirnameHarmonic, 'trainingHistory-' + name +'.npz'),allow_pickle=True)['arr_0'].item()

   
   if skip_measures:
       return HFmodel, training_historyHF, None

   HFmodel.compile(custom_metrics=cmetricsHF)
   HFmodelzero.compile(custom_metrics=cmetricsHF)




   valzero=HFmodelzero.test_step(dataHF_val_dict)
   valtrained=HFmodel.test_step(dataHF_val_dict)
   valzero = {key: float(value.numpy()) for key, value in valzero.items()}
   valtrained = {key: float(value.numpy()) for key, value in valtrained.items()}

   #valzero=HFmodelzero.evaluate(dataHF_val_dict,return_dict=True)
   #valtrained=HFmodel.evaluate(dataHF_val_dict,return_dict=True)
   #valzero = {key: value.numpy() for key, value in valzero.items()}
   #valtrained = {key: value.numpy() for key, value in valtrained.items()}

   #metricsnames=HFmodel.metrics_names
   #valzero = {metricsnames[i]: valzero[i] for i in range(len(valzero))}
   #valtrained= {metricsnames[i]: valtrained[i] for i in range(len(valtrained))}
  
   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for trained network: ")
   print(valtrained)
   print("ratio of trained to zero: " + str({key + " ratio": value/(valzero[key]+1e-8) for key, value in valtrained.items()}))


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure_section(HFmodel,dataHF["X_val"],dataHF["y_val"][:, -2])
   print("average transition discrepancy in standard deviations (note, underestimate as our std.dev. ignores variation in phase): " + str(averagediscrepancyinstdevs))
   #meanfailuretosolveequation,_,_=HYM_measure_val_with_H(HFmodel,dataHF)

   #meanfailuretosolveequation= batch_process_helper_func(
   #     tf.function(lambda x,y,z,w: tf.expand_dims(HYM_measure_val_with_H_for_batching(HFmodel,x,y,z,w),axis=0)),
   #     (dataHF['X_val'],dataHF['y_val'],dataHF['val_pullbacks'],dataHF['inv_mets_val']),
   #     batch_indices=(0,1,2,3),
   #     batch_size=50
   # )
   import time
   print("computing mean failure to solve equation", time.strftime("%H:%M:%S", time.localtime()))
   meanfailuretosolveequation,_,_ = HYM_measure_val_with_H(HFmodel,dataHF)
   meanfailuretosolveequation=tf.reduce_mean(meanfailuretosolveequation)
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   TrainedDivTrained, avgavagTrainedDivTrained, TrainedDivFS, avgavagTrainedDivFS, FS_DivFS, avgavagFS_DivFS = HYM_measure_val_with_H_relative_to_norm(HFmodel,dataHF_val_dict,betamodel,metric_model)
   print("trained coclosure divided by norm of v: " + str(TrainedDivTrained))
   print("avg/avg trained coclosure divided by norm of trained v: " + str(avgavagTrainedDivTrained))
   print("trained coclosure divided by norm of v_FS: " + str(TrainedDivFS))
   print("avg/avg trained coclosure divided by norm of v_FS: " + str(avgavagTrainedDivFS))
   print("FS coclosure divided by norm of v_FS: " + str(FS_DivFS))
   print("avg/avg FS coclosure divided by norm of v_FS: " + str(avgavagFS_DivFS))
   print("\n\n")
   return HFmodel,training_historyHF, meanfailuretosolveequation




