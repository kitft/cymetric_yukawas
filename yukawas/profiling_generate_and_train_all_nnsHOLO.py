from cymetric.config import real_dtype, complex_dtype
from cymetric.models.fubinistudy import FSModel
import tensorflow as tf
import tensorflow.keras as tfk
from laplacian_funcs import *
from BetaModel import *
from HarmonicFormModel import *
from OneAndTwoFormsForLineBundles import *
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
from OneAndTwoFormsForLineBundles import *
from custom_networks import *

#from pympler import tracker
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

kmoduli = np.array([1,1,1,1])


def create_adam_optimizer_with_decay(initial_learning_rate, nEpochs, final_lr_factor=10):
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

    return tf.keras.optimizers.Adam(learning_rate=lr_schedule)

def generate_points_and_save_using_defaults_for_eval(free_coefficient,number_points,force_generate=False,seed_set=0):
   coefficients=np.array([1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, \
0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, free_coefficient, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1])
   kmoduli = np.array([1,1,1,1])
   ambient = np.array([1,1,1,1])
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
   pg._set_seed(seed_set)
   dirname = 'data/tetraquadric_pg_for_eval_with_'+str(free_coefficient) 
   print("dirname: " + dirname)
   #test if the directory exists, if not, create it
   if force_generate or (not os.path.exists(dirname)):
      print("Generating: forced? " + str(force_generate))
      kappa = pg.prepare_dataset(number_points, dirname)
      pg.prepare_basis(dirname, kappa=kappa)
   elif os.path.exists(dirname):
      try:
         data = np.load(os.path.join(dirname, 'dataset.npz'))
         if (len(data['X_train'])+len(data['X_val']))!=number_points:
            print("wrong length - generating anyway")
            kappa = pg.prepare_dataset(number_points, dirname)
            pg.prepare_basis(dirname, kappa=kappa)
      except:
         print("error loading - generating anyway")
         kappa = pg.prepare_dataset(number_points, dirname)
         pg.prepare_basis(dirname, kappa=kappa)
   return pg,kmoduli


def generate_points_and_save_using_defaults(free_coefficient,number_points,force_generate=False,seed_set=0):
   coefficients=np.array([1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, \
   0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, free_coefficient, 0, 0, 0, 0, 0, \
   0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, \
   0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1])
   kmoduli = np.array([1,1,1,1])
   ambient = np.array([1,1,1,1])
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
   pg._set_seed(seed_set)
   dirname = 'data/tetraquadric_pg_with_'+str(free_coefficient) 
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
         if (len(data['X_train'])+len(data['X_val']))!=number_points:
            print("wrong length - generating anyway")
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
   dirname = 'data/tetraquadric_pg_with_'+str(free_coefficient)
   name = 'phimodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(bSizes[1]) + 's' + str(nlayer) + 'x' +str(nHidden)
   print('dirname: ' + dirname)
   print('name: ' + name)
   
   data = np.load(os.path.join(dirname, 'dataset.npz'))
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
    final_lr_factor=10  # This will decay to lRate/10
)
   # compile so we can test on validation set before training
   phimodel.compile(custom_metrics=cmetrics)
   phimodelzero.compile(custom_metrics=cmetrics)

   ## compare validation loss before training for zero network and nonzero network
   datacasted=[tf.cast(data['X_val'],real_dtype),tf.cast(data['y_val'],real_dtype)]
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
   valzero = {key: value.numpy() for key, value in valzero.items()}
   valraw = {key: value.numpy() for key, value in valraw.items()}

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

   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(phimodel,tf.cast(data["X_val"],real_dtype))
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   #IMPLEMENT THE FOLLOWING
   #meanfailuretosolveequation,_,_=measure_laplacian_failure(phimodel,data)
   print("\n\n")
   return phimodel,training_history

def load_nn_phimodel(free_coefficient,nlayer=3,nHidden=128,nEpochs=50,bSizes=[192,50000],stddev=0.1,lRate=0.001,set_weights_to_zero=False):
   dirname = 'data/tetraquadric_pg_with_'+str(free_coefficient)
   name = 'phimodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(bSizes[1]) + 's' + str(nlayer) + 'x' +str(nHidden)
   print(dirname)
   print(name)
   
   data = np.load(os.path.join(dirname, 'dataset.npz'))
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

   datacasted=[tf.cast(data['X_val'],real_dtype),tf.cast(data['y_val'],real_dtype)]

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
      training_history=0
   else:
      #phimodel.model=tf.keras.layers.TFSMLayer(os.path.join(dirname,name),call_endpoint="serving_default")
      #phimodel.model=tf.keras.models.load_model(os.path.join(dirname, name) + ".keras")
      #print_sample_params(phimodel.model)
      phimodel.model.load_weights(os.path.join(dirname, name) + '.weights.h5')
      #print_sample_params(phimodel.model)
      training_history=np.load(os.path.join(dirname, 'trainingHistory-' + name +'.npz'),allow_pickle=True)['arr_0'].item()

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
   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(phimodel,tf.cast(data["X_val"],real_dtype))
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   print("\n\n")
   #IMPLEMENT THE FOLLOWING
   #meanfailuretosolveequation,_,_=measure_laplacian_failure(phimodel,data)
   #print("\n\n")
   #print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   tf.keras.backend.clear_session()
   return phimodel,training_history

def generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM,number_pointsHYM,phimodel,force_generate=False,seed_set=0):

   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'data/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = 'data/tetraquadric_pg_with_'+str(free_coefficient)
   print("dirname for beta: " + dirnameHYM)

   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))
   
   coefficients=np.array([1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, \
0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, free_coefficient, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1])
   kmoduli = np.array([1,1,1,1])
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
         if (len(data['X_train'])+len(data['X_val']))!=number_pointsHYM:
            print("wrong length - generating anyway")
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

   
def train_and_save_nn_HYM(free_coefficient,linebundleforHYM,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],stddev=0.1,lRate=0.001,use_zero_network=False,alpha=[1,1],load_network=False):
   
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'data/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = 'data/tetraquadric_pg_with_'+str(free_coefficient)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))


   databeta = np.load(os.path.join(dirnameHYM, 'dataset.npz'))
   databeta_train=tf.data.Dataset.from_tensor_slices(dict(list(dict(databeta).items())[:len(dict(databeta))//2]))
   databeta_val_dict=dict(list(dict(databeta).items())[len(dict(databeta))//2:])
   databeta_val=tf.data.Dataset.from_tensor_slices(databeta_val_dict)
   # batch_sizes=[64,10000]
   databeta_train=databeta_train.shuffle(buffer_size=1024).batch(bSizes[0],drop_remainder=True)
   datacasted=[tf.cast(databeta['X_val'],real_dtype),tf.cast(databeta['y_val'],real_dtype)]

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
   activ=tf.square
   #activ=tfk.activations.gelu
   nn_beta = BiholoModelFuncGENERALforHYMinv3(shapeofnetwork,BASIS,activation=activ,stddev=stddev,use_zero_network=use_zero_network,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_beta_zero = BiholoModelFuncGENERALforHYMinv3(shapeofnetwork,BASIS,activation=activ,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
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
    final_lr_factor=10  # This will decay to lRate/10
    )
   # compile so we can test on validation set before training
   betamodel.compile(custom_metrics=cmetrics)
   betamodelzero.compile(custom_metrics=cmetrics)
   
   #datacasted=[tf.cast(data['X_val'],real_dtype),tf.cast(data['y_val'],real_dtype)]
   print("testing zero and raw")
   valzero=betamodelzero.test_step(databeta_val_dict)
   valraw=betamodel.test_step(databeta_val_dict)
   valzero = {key: value.numpy() for key, value in valzero.items()}
   valraw = {key: value.numpy() for key, value in valraw.items()}
   print("tested zero and raw")
   
   training_historyBeta={'transition_loss': [10**(-8)],'laplacian_loss': [10000000]}
   i=0
   newLR=lRate
   #while (training_historyBeta['transition_loss'][-1]<10**(-5)) or (training_historyBeta['laplacian_loss'][-1]>1.):
   # continue looping if >10 or is nan
   while (training_historyBeta['laplacian_loss'][-1]>10.) or (np.isnan( training_historyBeta['laplacian_loss'][-1])):
      print("trying i "+str(i))
      if i >0:

         print('trying again laplacian_loss too big')
         #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.2)
         #nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network,kernel_initializer=initializer)#note we don't need a last bias (flat direction)
         #nn_beta = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,stddev=stddev,use_zero_network=use_zero_network)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
         #nn_beta = BiholoModelFuncGENERALforHYMinv2(shapeofnetwork,BASIS,activation=tfk.activations.gelu,stddev=stddev,use_zero_network=use_zero_network)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
         nn_beta = BiholoModelFuncGENERALforHYMinv3(shapeofnetwork,BASIS,activation=activ,stddev=stddev,use_zero_network=use_zero_network,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
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
                 final_lr_factor=10  # This will decay to lRate/10
                 )
         #betamodel= BetaModel(nn_beta,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
         betamodel.model=nn_beta
         #cb_list, cmetrics = getcallbacksandmetricsHYM(databeta)
         betamodel.compile(custom_metrics=cmetrics)
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


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(betamodel,tf.cast(databeta["X_val"],real_dtype))
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))


   import time
   start=time.time()
   meanfailuretosolveequation,_,_=HYM_measure_val(betamodel,databeta)
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("time to do that: ",time.time()-start)
   print("\n\n")

   tf.keras.backend.clear_session()
   return betamodel,training_historyBeta

def load_nn_HYM(free_coefficient,linebundleforHYM,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],stddev=0.1,lRate=0.001,set_weights_to_zero=False):
   
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'data/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = 'data/tetraquadric_pg_with_'+str(free_coefficient)
   name = 'betamodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(nlayer) + 'x' +str(nHidden)
   print("name: " + name)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))


   databeta = np.load(os.path.join(dirnameHYM, 'dataset.npz'))
   databeta_train=tf.data.Dataset.from_tensor_slices(dict(list(dict(databeta).items())[:len(dict(databeta))//2]))
   databeta_val_dict=dict(list(dict(databeta).items())[len(dict(databeta))//2:])
   databeta_val=tf.data.Dataset.from_tensor_slices(databeta_val_dict)
   # batch_sizes=[64,10000]
   databeta_train=databeta_train.shuffle(buffer_size=1024).batch(bSizes[0],drop_remainder=True)

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
   nn_beta = BiholoModelFuncGENERALforHYMinv3(shapeofnetwork,BASIS,activation=activ,stddev=stddev,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_beta_zero = BiholoModelFuncGENERALforHYMinv3(shapeofnetwork,BASIS,activation=activ,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #copie from phi above
   #nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   
   #nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)#note we don't need a last bias (flat direction)
   #nn_beta_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)#note we don't need a last bias (flat direction)
   
   betamodel= BetaModel(nn_beta,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
   betamodelzero= BetaModel(nn_beta_zero,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])

   if set_weights_to_zero:
      training_historyBeta=0
   else:
      #betamodel.model=tf.keras.layers.TFSMLayer(os.path.join(dirnameHYM,name),call_endpoint="serving_default")
      #betamodel.model=tf.keras.models.load_model(os.path.join(dirnameHYM, name) + ".keras")
      #init vars so the load works
      betamodel(databeta_val_dict['X_val'][0:1])
      betamodel.model.load_weights(os.path.join(dirnameHYM, name) + '.weights.h5')
      training_historyBeta=np.load(os.path.join(dirnameHYM, 'trainingHistory-' + name +'.npz'),allow_pickle=True)['arr_0'].item()

   betamodel.compile(custom_metrics=cmetrics)
   betamodelzero.compile(custom_metrics=cmetrics)
   
   #valzero=betamodelzero.evaluate(databeta_val_dict,return_dict=True)
   #valtrained=betamodel.evaluate(databeta_val_dict,return_dict=True)
   ##valzero = {key: value.numpy() for key, value in valzero.items()}
   ##valtrained= {key: value.numpy() for key, value in valtrained.items()}

   valzero=betamodelzero.test_step(databeta_val_dict)
   valtrained=betamodel.test_step(databeta_val_dict)
   valzero = {key: value.numpy() for key, value in valzero.items()}
   valtrained= {key: value.numpy() for key, value in valtrained.items()}

   #metricsnames=betamodel.metrics_names

   #valzero = {metricsnames[i]: valzero[i] for i in range(len(valzero))}
   #valtrained= {metricsnames[i]: valtrained[i] for i in range(len(valtrained))}


   

   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for trained network: ")
   print(valtrained)
   print("ratio of trained to zero: " + str({key + " ratio": value/(valzero[key]+1e-8) for key, value in valtrained.items()}))


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(betamodel,tf.cast(databeta["X_val"],real_dtype))
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   import time
   start=time.time()
   meanfailuretosolveequation,_,_=HYM_measure_val(betamodel,databeta)
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("time to do that: ",time.time()-start)
   print("\n\n")
   return betamodel,training_historyBeta


def generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM,functionforbaseharmonicform_jbar,phimodel,betamodel,number_pointsHarmonic,force_generate=False,seed_set=0):
   # get names
   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameForMetric = 'data/tetraquadric_pg_with_'+str(free_coefficient)
   dirnameHYM = 'data/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameHarmonic = 'data/tetraquadricHarmonicH_pg'+str(free_coefficient)+'forLB_'+lbstring+nameOfBaseHF
   print("dirname for harmonic form: " + dirnameHarmonic)

   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))
   
   coefficients=np.array([1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, \
0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, free_coefficient, 0, 0, 0, 0, 0, \
0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, \
0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1])
   kmoduli = np.array([1,1,1,1])
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
         if (len(data['X_train'])+len(data['X_val']))!=number_pointsHarmonic:
            print("wrong length - generating anyway")
            kappaHarmonic=prepare_dataset_HarmonicForm(pg,data,number_pointsHarmonic,dirnameHarmonic,phimodel,linebundleforHYM,BASIS,functionforbaseharmonicform_jbar,betamodel)
      except:
         print("problem loading data - generating anyway")
         kappaHarmonic=prepare_dataset_HarmonicForm(pg,data,number_pointsHarmonic,dirnameHarmonic,phimodel,linebundleforHYM,BASIS,functionforbaseharmonicform_jbar,betamodel)
   

def getcallbacksandmetricsHF(dataHF):
   dataHF_val_dict=dict(list(dict(dataHF).items())[len(dict(dataHF))//2:])

   tcbHF = TransitionCallback((dataHF['X_val'], dataHF['y_val']))
   lplcbHF = LaplacianCallback(dataHF_val_dict)
   # lplcb = LaplacianCallback(dataHF_val)
   cb_listHF = [lplcbHF,tcbHF]
   cmetricsHF = [TotalLoss(), LaplacianLoss(), TransitionLoss()]
   return cb_listHF, cmetricsHF

   
def train_and_save_nn_HF(free_coefficient,linebundleforHYM,betamodel,functionforbaseharmonicform_jbar,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],lRate=0.001,alpha=[1,500],use_zero_network=False,load_network=False,stddev=0.05,final_layer_scale=1.0):
   
   #perm= tracker.SummaryTracker()
   #print("perm1")
   #print(perm.print_diff())

   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'data/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = 'data/tetraquadric_pg_with_'+str(free_coefficient)
   dirnameHarmonic = 'data/tetraquadricHarmonicH_pg'+str(free_coefficient)+'forLB_'+lbstring+nameOfBaseHF
   name = 'HFmodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+  str(nlayer) + 'x' +str(nHidden)
   print("dirname: " + dirnameHarmonic)
   print("name: " + name)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))

   #print("perm2")
   #print(perm.print_diff())
   dataHF = np.load(os.path.join(dirnameHarmonic, 'dataset.npz'))
   dataHF_train=tf.data.Dataset.from_tensor_slices(dict(list(dict(dataHF).items())[:len(dict(dataHF))//2]))
   dataHF_val_dict=dict(list(dict(dataHF).items())[len(dict(dataHF))//2:])
   dataHF_val_dict = {key: tf.convert_to_tensor(value) for key, value in dataHF_val_dict.items()}
   dataHF_val=tf.data.Dataset.from_tensor_slices(dataHF_val_dict)
   # batch_sizes=[64,10000]
   dataHF_train=dataHF_train.shuffle(buffer_size=1024).batch(bSizes[0],drop_remainder=True)




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
   shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[nsections]

   # need a last bias layer due to transition!
   #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.3)
   #nn_HF = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,kernel_initializer=initializer)
   #nn_HF_zero = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,use_zero_network=True)
   activ=tf.square
   stddev=stddev
   nn_HF = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,use_zero_network=use_zero_network,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_HF_zero  = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,use_zero_network=True,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   
   #print("perm4")
   #print(perm.print_diff())

   #while True:
   #     nn_HF = BiholoModelFuncGENERALforSigma(shapeofnetwork, BASIS, linebundleindices=linebundleforHYM, nsections=nsections, k_phi=np.array([0, 0, 0, 0]), activation=activ, stddev=stddev, use_zero_network=use_zero_network)
   #     HFmodel = HarmonicFormModel(nn_HF, BASIS, betamodel, linebundleforHYM, functionforbaseharmonicform_jbar, alpha=alpha, norm=[1. for _ in range(2)])

   #     opt = create_adam_optimizer_with_decay(
   #             initial_learning_rate=lRate,
   #             nEpochs=nEpochs * (len(dataHF['X_train']) // bSizes[0]),
   #             final_lr_factor=10
   #             )

   #     HFmodel.compile(custom_metrics=cmetricsHF)
   #     training_historyHF = {'transition_loss': [10**(-8)], 'laplacian_loss': [10.**10]}
   #     HFmodel, training_historyHF = train_modelHF(HFmodel, dataHF_train, optimizer=opt, epochs=nEpochs, batch_sizes=bSizes,
   #             verbose=1, custom_metrics=cmetricsHF, callbacks=cb_listHF)

   #     if training_historyHF['laplacian_loss'][-1] <= 1000:
   #         break
   #     else:
   #         print('Laplacian loss too high, reinitializing the model and retrying.')

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
           final_lr_factor=10  # This will decay to lRate/10
           )
   # compile so we can test on validation set before training
   HFmodel.compile(custom_metrics=cmetricsHF)
   HFmodelzero.compile(custom_metrics=cmetricsHF)

   #print("perm6")
   #print(perm.print_diff())

   #only use 100 entries in dataHF_val_dict
   #dataHF_val_dict_short={key: value[:100] for key, value in dataHF_val_dict.items()}
   print("testing zero and raw")
   valzero=HFmodelzero.test_step(dataHF_val_dict)
   valraw=HFmodel.test_step(dataHF_val_dict)
   valzero = {key: value.numpy() for key, value in valzero.items()}
   valraw = {key: value.numpy() for key, value in valraw.items()}
   print("tested zero and raw")

   #print("perm7")
   #print(perm.print_diff())


   training_historyHF={'transition_loss': [10**(-8)], 'laplacian_loss':[10.**10]}
   i=0
   newLR = lRate
   # continue looping if >10 or is nan
   while i ==0:
      #while (training_historyHF['laplacian_loss'][-1]>10.**1.5) or (np.isnan( training_historyHF['laplacian_loss'][-1])):
      #while i==0:#((training_historyHF['laplacian_loss'][-1]/training_historyHF['laplacian_loss'][0])<1.1):#wass 0.2
      print("loop HF iteration: " + str(i))
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
         nn_HF = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,nsections=nsections,k_phi=np.array([0,0,0,0]),activation=activ,stddev=stddev,use_zero_network=use_zero_network,final_layer_scale=final_layer_scale,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
         HFmodel = HarmonicFormModel(nn_HF,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
         if newLR>0.0002:
             newLR=newLR/2
             print("new LR " + str(newLR))
         opt = create_adam_optimizer_with_decay(
                 initial_learning_rate=newLR,
                 nEpochs=nEpochs*(len(dataHF['X_train'])//bSizes[0]),
                 final_lr_factor=10  # This will decay to lRate/10
                 )
         #cb_listHF, cmetricsHF = getcallbacksandmetricsHF(dataHF)
         #HFmodel2.compile(custom_metrics=cmetricsHF)
         ##HFmodel=HFmodel2
         ##opt=opt2
         #HFmodel2, training_historyHF= train_modelHF(HFmodel2, dataHF_train, optimizer=opt2, epochs=nEpochs, batch_sizes=bSizes, 
                 #verbose=1, custom_metrics=cmetricsHF, callbacks=cb_listHF)
         valraw=HFmodel.test_step(dataHF_val_dict)
         valraw = {key: value.numpy() for key, value in valraw.items()}
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
   valfinal = {key: value.numpy() for key, value in valfinal.items()}

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

   #print("perm10")
   #print(perm.print_diff())

   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure_section(HFmodel,tf.cast(dataHF["X_val"],real_dtype))
   print("average transition discrepancy in standard deviations (note, underestimate as our std.dev. ignores variation in phase): " + str(averagediscrepancyinstdevs))



   import time
   start=time.time()
   meanfailuretosolveequation,_,_=HYM_measure_val_with_H(HFmodel,dataHF)
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("time to do that: ",time.time()-start)
   print("\n\n")

   tf.keras.backend.clear_session()
   del dataHF, dataHF_train, dataHF_val_dict,  dataHF_val,valfinal,valraw,valzero
   #print("perm11")
   #print(perm.print_diff())
   return HFmodel,training_historyHF



def load_nn_HF(free_coefficient,linebundleforHYM,betamodel,functionforbaseharmonicform_jbar,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],lRate=0.001,alpha=[1,1],set_weights_to_zero=False):
   
   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'data/tetraquadricHYM_pg_with_'+str(free_coefficient)+'forLB_'+lbstring
   dirnameForMetric = 'data/tetraquadric_pg_with_'+str(free_coefficient)
   dirnameHarmonic = 'data/tetraquadricHarmonicH_pg'+str(free_coefficient)+'forLB_'+lbstring+nameOfBaseHF
   name = 'HFmodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+  str(nlayer) + 'x' +str(nHidden)
   print("dirname: " + dirnameHarmonic)
   print("name: " + name)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))

   dataHF = np.load(os.path.join(dirnameHarmonic, 'dataset.npz'))
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
   shapeofnetwork=[nfirstlayer]+shapeofinternalnetwork+[nsections]
   # need a last bias layer due to transition!
   #nn_HF = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,use_zero_network=True,kernel_initializer=initializer)
   #nn_HF_zero = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,use_zero_network=True)
   activ=tf.square
   stddev=0.1 # THIS DOESN'T DO ANYTHING!!
   nn_HF = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,k_phi=np.array([0,0,0,0]),nsections=nsections,activation=activ,stddev=stddev,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_HF_zero  = BiholoModelFuncGENERALforSigma(shapeofnetwork,BASIS,linebundleindices=linebundleforHYM,k_phi=np.array([0,0,0,0]),nsections=nsections,activation=activ,stddev=stddev,use_zero_network=True,use_symmetry_reduced_TQ=use_symmetry_reduced_TQ)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   
   HFmodel = HarmonicFormModel(nn_HF,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
   HFmodelzero = HarmonicFormModel(nn_HF_zero,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])

   if set_weights_to_zero:
      training_historyHF=0
   else:
      #print(HFmodel.model.weights[0])
      #HFmodel.model=tf.keras.layers.TFSMLayer(os.path.join(dirnameHarmonic,name),call_endpoint="serving_default")
      #HFmodel.model=tf.keras.models.load_model(os.path.join(dirnameHarmonic, name) + ".keras")
      #init variables
      HFmodel(dataHF_val_dict['X_val'][0:1])
      HFmodel.model.load_weights(os.path.join(dirnameHarmonic, name) + ".weights.h5")
      #print(HFmodel.model.weights[0])
      training_historyHF=np.load(os.path.join(dirnameHarmonic, 'trainingHistory-' + name +'.npz'),allow_pickle=True)['arr_0'].item()

   HFmodel.compile(custom_metrics=cmetricsHF)
   HFmodelzero.compile(custom_metrics=cmetricsHF)




   valzero=HFmodelzero.test_step(dataHF_val_dict)
   valtrained=HFmodel.test_step(dataHF_val_dict)
   valzero = {key: value.numpy() for key, value in valzero.items()}
   valtrained = {key: value.numpy() for key, value in valtrained.items()}

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


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure_section(HFmodel,tf.cast(dataHF["X_val"],real_dtype))
   print("average transition discrepancy in standard deviations (note, underestimate as our std.dev. ignores variation in phase): " + str(averagediscrepancyinstdevs))
   meanfailuretosolveequation,_,_=HYM_measure_val_with_H(HFmodel,dataHF)
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("\n\n")
   return HFmodel,training_historyHF




