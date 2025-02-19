from cymetric.config import float_dtype, complex_dtype
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

def generate_points_and_save_using_defaults_for_eval(free_coefficient,number_points,force_generate=False):
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   coefficients=np.array([u, 0, t, 0, s, 0, r, 0, q, 0, p, 0, o, 0, n, 0, m, \
           0, l, 0, k, 0, j, 0, i, 0, h, 0, g, 0, f, 0, e, 0, \
           d, 0, c, 0, b, 0, a, 0, b, 0, c, 0, d, 0, e, 0, f, \
           0, g, 0, h, 0, i, 0, j, 0, k, 0, l, 0, m, 0, n, 0, \
           o, 0, p, 0, q, 0, r, 0, s, 0, t, 0, u])

   kmoduli = np.array([1,1,1,1])
   ambient = np.array([1,1,1,1])
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
   dirname = 'tetraquadric_pg_for_eval_with_'+str(free_coefficient_name) 
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


def generate_points_and_save_using_defaults(free_coefficient,number_points,force_generate=False):
   #   #coefficients=np.array([1, 0, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, \
   #0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, free_coefficient_name, 0, 0, 0, 0, 0, \
   #0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, \
   #0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 1])
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   coefficients=np.array([u, 0, t, 0, s, 0, r, 0, q, 0, p, 0, o, 0, n, 0, m, \
0, l, 0, k, 0, j, 0, i, 0, h, 0, g, 0, f, 0, e, 0, \
d, 0, c, 0, b, 0, a, 0, b, 0, c, 0, d, 0, e, 0, f, \
0, g, 0, h, 0, i, 0, j, 0, k, 0, l, 0, m, 0, n, 0, \
o, 0, p, 0, q, 0, r, 0, s, 0, t, 0, u])
   kmoduli = np.array([1,1,1,1])
   ambient = np.array([1,1,1,1])
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
   dirname = 'tetraquadric_pg_with_'+str(free_coefficient_name) 
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
   cb_list = [ scb, kcb, tcb, volkcb]
   cmetrics = [TotalLoss(), SigmaLoss(), KaehlerLoss(), TransitionLoss(), VolkLoss()]#, RicciLoss()]
   return cb_list, cmetrics

#def make_nn(n_in,n_out,nlayer,nHidden,act='gelu',lastbias=False,use_zero_network=False):
#   if use_zero_network:
#      nn_phi = tf.keras.Sequential()
#      nn_phi.add(tfk.Input(shape=(n_in,)))
#      for i in range(nlayer):
#          nn_phi.add(tfk.layers.Dense(nHidden, activation=act,kernel_initializer='zeros'))
#      nn_phi.add(tfk.layers.Dense(n_out, use_bias=lastbias))
#   else:
#      nn_phi = tf.keras.Sequential()
#      nn_phi.add(tfk.Input(shape=(n_in,)))
#      for i in range(nlayer):
#          nn_phi.add(tfk.layers.Dense(nHidden, activation=act))
#      nn_phi.add(tfk.layers.Dense(n_out, use_bias=lastbias))
#   return nn_phi
#

def make_nn(n_in,n_out,nlayer,nHidden,act='gelu',lastbias=False,use_zero_network=False,kernel_initializer='glorot_uniform'):
   if use_zero_network:
      nn_phi = tf.keras.Sequential()
      nn_phi.add(tfk.Input(shape=(n_in,)))
      for i in range(nlayer):
          nn_phi.add(tfk.layers.Dense(nHidden, activation=act,kernel_initializer='zeros'))
      nn_phi.add(tfk.layers.Dense(n_out, use_bias=lastbias))
   else:
      nn_phi = tf.keras.Sequential()
      nn_phi.add(tfk.Input(shape=(n_in,)))
      for i in range(nlayer):
          nn_phi.add(tfk.layers.Dense(nHidden, activation=act,kernel_initializer=kernel_initializer))
      nn_phi.add(tfk.layers.Dense(n_out, use_bias=lastbias))
   return nn_phi
   
def train_and_save_nn(free_coefficient,nlayer=3,nHidden=128,nEpochs=50,bSizes=[192,50000],lRate=0.001,use_zero_network=False):
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)

   dirname = 'tetraquadric_pg_with_'+str(free_coefficient_name)
   name = 'phimodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(bSizes[1]) + 's' + str(nlayer) + 'x' +str(nHidden)
   print('dirname' + dirname)
   print('name' + name)
   
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
   nn_phi = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   phimodel = PhiFSModel(nn_phi, BASIS, alpha=alpha)
   phimodelzero = PhiFSModel(nn_phi_zero, BASIS, alpha=alpha)

   #Note, currently running legacy due to ongoing tf issue with M1/M2. 
   #Use the commented line instead if not on an M1/M2 machine
   #opt = tfk.optimizers.Adam(learning_rate=lRate)
   opt = tfk.optimizers.legacy.Adam(learning_rate=lRate)
   # compile so we can test on validation set before training
   phimodel.compile(custom_metrics=cmetrics)
   phimodelzero.compile(custom_metrics=cmetrics)

   # compare validation loss before training for zero network and nonzero network
   datacasted=[tf.cast(data['X_val'],float_dtype),tf.cast(data['y_val'],float_dtype)]
   #need to re-enable learning, in case there's been a problem:
   phimodel.learn_transition = True
   phimodelzero.learn_transition = True
   phimodel.learn_volk = True
   phimodelzero.learn_volk = True
   #phimodel.learn_ricci_val= True
   #phimodelzero.learn_ricci_val= True
   valzero=phimodelzero.test_step(datacasted)
   valraw=phimodel.test_step(datacasted)
   # phimodel.learn_ricci_val=False 
   # phimodelzero.learn_ricci_val=False 
   valzero = {key: value.numpy() for key, value in valzero.items()}
   valraw = {key: value.numpy() for key, value in valraw.items()}

   phimodel, training_history = train_model(phimodel, data, optimizer=opt, epochs=nEpochs, batch_sizes=bSizes, 
                                       verbose=1, custom_metrics=cmetrics, callbacks=cb_list)
   print("finished training\n")
   phimodel.model.save(os.path.join(dirname, name))
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

   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(phimodel,tf.cast(data["X_val"],float_dtype))
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   #IMPLEMENT THE FOLLOWING
   #meanfailuretosolveequation,_,_=measure_laplacian_failure(phimodel,data)
   print("\n\n")
   return phimodel,training_history

def load_nn_phimodel(free_coefficient,nlayer=3,nHidden=128,nEpochs=50,bSizes=[192,50000],lRate=0.001,set_weights_to_zero=False):
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)

   dirname = 'tetraquadric_pg_with_'+str(free_coefficient_name)
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
   nn_phi = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   phimodel = PhiFSModel(nn_phi, BASIS, alpha=alpha)
   phimodelzero = PhiFSModel(nn_phi_zero, BASIS, alpha=alpha)

   if set_weights_to_zero:
      training_history=0
   else:
      phimodel.model=tf.keras.models.load_model(os.path.join(dirname,name))
      training_history=np.load(os.path.join(dirname, 'trainingHistory-' + name +'.npz'),allow_pickle=True)['arr_0'].item()

   phimodel.compile(custom_metrics=cmetrics)
   phimodelzero.compile(custom_metrics=cmetrics)

   # compare validation loss before training for zero network and nonzero network
   datacasted=[tf.cast(data['X_val'],float_dtype),tf.cast(data['y_val'],float_dtype)]
   #need to re-enable learning, in case there's been a problem:
   phimodel.learn_transition = True
   phimodelzero.learn_transition = True
   phimodel.learn_volk = True
   phimodelzero.learn_volk = True
   #phimodel.learn_ricci_val= True
   #phimodelzero.learn_ricci_val= True
   valzero=phimodelzero.evaluate(datacasted[0],datacasted[1])
   valtrained=phimodel.evaluate(datacasted[0],datacasted[1])
   metricsnames=phimodel.metrics_names
   # phimodel.learn_ricci_val=False 
   # phimodelzero.learn_ricci_val=False 
   valzero = {metricsnames[i]: valzero[i] for i in range(len(valzero))}
   valtrained= {metricsnames[i]: valtrained[i] for i in range(len(valtrained))}

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
   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(phimodel,tf.cast(data["X_val"],float_dtype))
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   print("\n\n")
   #IMPLEMENT THE FOLLOWING
   #meanfailuretosolveequation,_,_=measure_laplacian_failure(phimodel,data)
   #print("\n\n")
   #print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   return phimodel,training_history

def generate_points_and_save_using_defaultsHYM(free_coefficient,linebundleforHYM,number_pointsHYM,phimodel,force_generate=False):
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   coefficients=np.array([u, 0, t, 0, s, 0, r, 0, q, 0, p, 0, o, 0, n, 0, m, \
0, l, 0, k, 0, j, 0, i, 0, h, 0, g, 0, f, 0, e, 0, \
d, 0, c, 0, b, 0, a, 0, b, 0, c, 0, d, 0, e, 0, f, \
0, g, 0, h, 0, i, 0, j, 0, k, 0, l, 0, m, 0, n, 0, \
o, 0, p, 0, q, 0, r, 0, s, 0, t, 0, u])


   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'tetraquadricHYM_pg_with_'+str(free_coefficient_name)+'forLB_'+lbstring
   dirnameForMetric = 'tetraquadric_pg_with_'+str(free_coefficient_name)
   print("dirname for beta: " + dirnameHYM)

   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))
   

   kmoduli = np.array([1,1,1,1])
   ambient = np.array([1,1,1,1])
   
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)

   if force_generate or (not os.path.exists(dirnameHYM)):
      print("Generating: forced? " + str(force_generate))
      kappaHYM = prepare_dataset_HYM(pg,number_pointsHYM, dirnameHYM,phimodel,linebundleforHYM,BASIS,normalize_to_vol_j=True);
   elif os.path.exists(dirnameHYM):
      try:
         print("loading prexisting dataset")
         data = np.load(os.path.join(dirnameHYM, 'dataset.npz'))
         if (len(data['X_train'])+len(data['X_val']))!=number_pointsHYM:
            print("wrong length - generating anyway")
            kappaHYM = prepare_dataset_HYM(pg,number_pointsHYM, dirnameHYM,phimodel,linebundleforHYM,BASIS,normalize_to_vol_j=True);
      except:
         print("problem loading data - generating anyway")
         kappaHYM = prepare_dataset_HYM(pg,number_pointsHYM, dirnameHYM,phimodel,linebundleforHYM,BASIS,normalize_to_vol_j=True);
      
   

def getcallbacksandmetricsHYM(databeta):
   databeta_val_dict=dict(list(dict(databeta).items())[len(dict(databeta))//2:])
   tcb = TransitionCallback((databeta['X_val'], databeta['y_val']))
   lplcb = LaplacianCallback(databeta_val_dict)
   # lplcb = LaplacianCallback(data_val)
   cb_list = [lplcb,tcb]
   cmetrics = [TotalLoss(), LaplacianLoss(), TransitionLoss()]
   return cb_list, cmetrics

   
def train_and_save_nn_HYM(free_coefficient,linebundleforHYM,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],lRate=0.001,use_zero_network=False,alpha=[1,1]):
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'tetraquadricHYM_pg_with_'+str(free_coefficient_name)+'forLB_'+lbstring
   dirnameForMetric = 'tetraquadric_pg_with_'+str(free_coefficient_name)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))


   databeta = np.load(os.path.join(dirnameHYM, 'dataset.npz'))
   databeta_train=tf.data.Dataset.from_tensor_slices(dict(list(dict(databeta).items())[:len(dict(databeta))//2]))
   databeta_val_dict=dict(list(dict(databeta).items())[len(dict(databeta))//2:])
   databeta_val=tf.data.Dataset.from_tensor_slices(databeta_val_dict)
   # batch_sizes=[64,10000]
   databeta_train=databeta_train.shuffle(buffer_size=1024).batch(bSizes[0])

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
   name = 'betamodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(bSizes[1])  + str(nlayer) + 'x' +str(nHidden)
   print("name: " + name)

   initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.2)
   #nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)#note we don't need a last bias (flat direction)
   nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network,kernel_initializer=initializer)#note we don't need a last bias (flat direction)
   nn_beta_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)#note we don't need a last bias (flat direction)
   
   betamodel= BetaModel(nn_beta,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
   betamodelzero= BetaModel(nn_beta_zero,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])

   #Note, currently running legacy due to ongoing tf issue with M1/M2. 
   #Use the commented line instead if not on an M1/M2 machine
   #opt = tfk.optimizers.Adam(learning_rate=lRate)
   opt = tfk.optimizers.legacy.Adam(learning_rate=lRate)
   # compile so we can test on validation set before training
   betamodel.compile(custom_metrics=cmetrics)
   betamodelzero.compile(custom_metrics=cmetrics)
   
   #datacasted=[tf.cast(data['X_val'],float_dtype),tf.cast(data['y_val'],float_dtype)]
   valzero=betamodelzero.test_step(databeta_val_dict)
   valraw=betamodel.test_step(databeta_val_dict)
   valzero = {key: value.numpy() for key, value in valzero.items()}
   valraw = {key: value.numpy() for key, value in valraw.items()}
   
   training_historyBeta={'transition_loss': [10**(-8)],'laplacian_loss': [10]}
   i=0
   newLR=lRate
   while (training_historyBeta['transition_loss'][-1]<10**(-5)) or (training_historyBeta['laplacian_loss'][-1]>1.):
      if i >0:

         print('trying again, as transition_loss was too small or laplacian_loss too big')
         initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.2)
         nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network,kernel_initializer=initializer)#note we don't need a last bias (flat direction)
         #nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)#note we don't need a last bias (flat direction)
         if newLR>0.0002:
             newLR=newLR/2
             print("new LR " + str(newLR))
         opt = tfk.optimizers.legacy.Adam(learning_rate=newLR)
         betamodel= BetaModel(nn_beta,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
      betamodel, training_historyBeta= train_modelbeta(betamodel, databeta_train, optimizer=opt, epochs=nEpochs, batch_sizes=bSizes, 
                                        verbose=1, custom_metrics=cmetrics, callbacks=cb_list)
      i+=1
   print("finished training\n")
   betamodel.model.save(os.path.join(dirnameHYM, name))
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


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(betamodel,tf.cast(databeta["X_val"],float_dtype))
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   meanfailuretosolveequation,_,_=HYM_measure_val(betamodel,databeta)
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("\n\n")
   return betamodel,training_historyBeta

def load_nn_HYM(free_coefficient,linebundleforHYM,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],lRate=0.001,set_weights_to_zero=False):
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'tetraquadricHYM_pg_with_'+str(free_coefficient_name)+'forLB_'+lbstring
   dirnameForMetric = 'tetraquadric_pg_with_'+str(free_coefficient_name)
   name = 'betamodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(bSizes[1])  + str(nlayer) + 'x' +str(nHidden)
   print("name: " + name)

   #data = np.load(os.path.join(dirname, 'dataset.npz'))
   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))


   databeta = np.load(os.path.join(dirnameHYM, 'dataset.npz'))
   databeta_train=tf.data.Dataset.from_tensor_slices(dict(list(dict(databeta).items())[:len(dict(databeta))//2]))
   databeta_val_dict=dict(list(dict(databeta).items())[len(dict(databeta))//2:])
   databeta_val=tf.data.Dataset.from_tensor_slices(databeta_val_dict)
   # batch_sizes=[64,10000]
   databeta_train=databeta_train.shuffle(buffer_size=1024).batch(bSizes[0])

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

   nn_beta = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)#note we don't need a last bias (flat direction)
   nn_beta_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)#note we don't need a last bias (flat direction)
   
   betamodel= BetaModel(nn_beta,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
   betamodelzero= BetaModel(nn_beta_zero,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])

   if set_weights_to_zero:
      training_historyBeta=0
   else:
      betamodel.model=tf.keras.models.load_model(os.path.join(dirnameHYM,name))
      training_historyBeta=np.load(os.path.join(dirnameHYM, 'trainingHistory-' + name +'.npz'),allow_pickle=True)['arr_0'].item()

   betamodel.compile(custom_metrics=cmetrics)
   betamodelzero.compile(custom_metrics=cmetrics)
   
   valzero=betamodelzero.evaluate(databeta_val_dict)
   valtrained=betamodel.evaluate(databeta_val_dict)
   #valzero = {key: value.numpy() for key, value in valzero.items()}
   #valtrained= {key: value.numpy() for key, value in valtrained.items()}


   metricsnames=betamodel.metrics_names

   valzero = {metricsnames[i]: valzero[i] for i in range(len(valzero))}
   valtrained= {metricsnames[i]: valtrained[i] for i in range(len(valtrained))}


   

   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for trained network: ")
   print(valtrained)
   print("ratio of trained to zero: " + str({key + " ratio": value/(valzero[key]+1e-8) for key, value in valtrained.items()}))


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure(betamodel,tf.cast(databeta["X_val"],float_dtype))
   print("average transition discrepancy in standard deviations: " + str(averagediscrepancyinstdevs))
   meanfailuretosolveequation,_,_=HYM_measure_val(betamodel,databeta)
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("\n\n")
   return betamodel,training_historyBeta


def generate_points_and_save_using_defaultsHF(free_coefficient,linebundleforHYM,functionforbaseharmonicform_jbar,phimodel,betamodel,number_pointsHarmonic,force_generate=False):
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   coefficients=np.array([u, 0, t, 0, s, 0, r, 0, q, 0, p, 0, o, 0, n, 0, m, \
0, l, 0, k, 0, j, 0, i, 0, h, 0, g, 0, f, 0, e, 0, \
d, 0, c, 0, b, 0, a, 0, b, 0, c, 0, d, 0, e, 0, f, \
0, g, 0, h, 0, i, 0, j, 0, k, 0, l, 0, m, 0, n, 0, \
o, 0, p, 0, q, 0, r, 0, s, 0, t, 0, u])

   # get names
   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameForMetric = 'tetraquadric_pg_with_'+str(free_coefficient_name)
   dirnameHYM = 'tetraquadricHYM_pg_with_'+str(free_coefficient_name)+'forLB_'+lbstring
   dirnameHarmonic = 'tetraquadricHarmonicH_pg'+str(free_coefficient_name)+'forLB_'+lbstring+nameOfBaseHF
   print("dirname for harmonic form: " + dirnameHarmonic)

   BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))
   
   kmoduli = np.array([1,1,1,1])
   ambient = np.array([1,1,1,1])
   
   pg = PointGenerator(monomials, coefficients, kmoduli, ambient)

   if force_generate or (not os.path.exists(dirnameHarmonic)):
      print("Generating: forced? " + str(force_generate))
      kappaHarmonic=prepare_dataset_HarmonicForm(pg,number_pointsHarmonic,dirnameHarmonic,phimodel,linebundleforHYM,BASIS,functionforbaseharmonicform_jbar,betamodel)
   elif os.path.exists(dirnameHarmonic):
      try:
         print("loading prexisting dataset")
         data = np.load(os.path.join(dirnameHarmonic, 'dataset.npz'))
         if (len(data['X_train'])+len(data['X_val']))!=number_pointsHarmonic:
            print("wrong length - generating anyway")
            kappaHarmonic=prepare_dataset_HarmonicForm(pg,number_pointsHarmonic,dirnameHarmonic,phimodel,linebundleforHYM,BASIS,functionforbaseharmonicform_jbar,betamodel)
      except:
         print("problem loading data - generating anyway")
         kappaHarmonic=prepare_dataset_HarmonicForm(pg,number_pointsHarmonic,dirnameHarmonic,phimodel,linebundleforHYM,BASIS,functionforbaseharmonicform_jbar,betamodel)
   

def getcallbacksandmetricsHF(dataHF):
   dataHF_val_dict=dict(list(dict(dataHF).items())[len(dict(dataHF))//2:])

   tcbHF = TransitionCallback((dataHF['X_val'], dataHF['y_val']))
   lplcbHF = LaplacianCallback(dataHF_val_dict)
   # lplcb = LaplacianCallback(dataHF_val)
   cb_listHF = [lplcbHF,tcbHF]
   cmetricsHF = [TotalLoss(), LaplacianLoss(), TransitionLoss()]
   return cb_listHF, cmetricsHF

   
def train_and_save_nn_HF(free_coefficient,linebundleforHYM,betamodel,functionforbaseharmonicform_jbar,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],lRate=0.001,alpha=[1,500],use_zero_network=False):
   
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)

   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'tetraquadricHYM_pg_with_'+str(free_coefficient_name)+'forLB_'+lbstring
   dirnameForMetric = 'tetraquadric_pg_with_'+str(free_coefficient_name)
   dirnameHarmonic = 'tetraquadricHarmonicH_pg'+str(free_coefficient_name)+'forLB_'+lbstring+nameOfBaseHF
   name = 'HFmodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(bSizes[1])+  str(nlayer) + 'x' +str(nHidden)
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
   dataHF_train=dataHF_train.shuffle(buffer_size=1024).batch(bSizes[0])




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


   # need a last bias layer due to transition!
   initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.3)
   nn_HF = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,kernel_initializer=initializer)
   nn_HF_zero = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,use_zero_network=True)
   
   HFmodel = HarmonicFormModel(nn_HF,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
   HFmodelzero = HarmonicFormModel(nn_HF_zero,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])

   #Note, currently running legacy due to ongoing tf issue with M1/M2. 
   #Use the commented line instead if not on an M1/M2 machine
   #opt = tfk.optimizers.Adam(learning_rate=lRate)
   opt = tfk.optimizers.legacy.Adam(learning_rate=lRate)
   # compile so we can test on validation set before training
   HFmodel.compile(custom_metrics=cmetricsHF)
   HFmodelzero.compile(custom_metrics=cmetricsHF)

   valzero=HFmodelzero.test_step(dataHF_val_dict)
   valraw=HFmodel.test_step(dataHF_val_dict)
   valzero = {key: value.numpy() for key, value in valzero.items()}
   valraw = {key: value.numpy() for key, value in valraw.items()}


   training_historyHF={'transition_loss': [10**(-8)], 'laplacian_loss':[1.]}
   i=0
   while training_historyHF['transition_loss'][-1]<10**(-5) and ((training_historyHF['laplacian_loss'][-1]/training_historyHF['laplacian_loss'][0])>0.2):
      if i >0:
         print('trying again, as transition_loss was too small')
         initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.3)
         nn_HF = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,kernel_initializer=initializer)
         HFmodel = HarmonicFormModel(nn_HF,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
      HFmodel, training_historyHF= train_modelHF(HFmodel, dataHF, optimizer=opt, epochs=nEpochs, batch_sizes=bSizes, 
                                       verbose=1, custom_metrics=cmetricsHF, callbacks=cb_listHF)
      i+=1 

   print("finished training\n")
   HFmodel.model.save(os.path.join(dirnameHarmonic, name))
   np.savez_compressed(os.path.join(dirnameHarmonic, 'trainingHistory-' + name),training_historyHF)

   valfinal =HFmodel.test_step(dataHF_val_dict)
   valfinal = {key: value.numpy() for key, value in valfinal.items()}

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


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure_section(HFmodel,tf.cast(dataHF["X_val"],float_dtype))
   print("average transition discrepancy in standard deviations (note, underestimate as our std.dev. ignores variation in phase): " + str(averagediscrepancyinstdevs))
   meanfailuretosolveequation,_,_=HYM_measure_val_with_H(HFmodel,dataHF)
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("\n\n")
   return HFmodel,training_historyHF


def load_nn_HF(free_coefficient,linebundleforHYM,betamodel,functionforbaseharmonicform_jbar,nlayer=3,nHidden=128,nEpochs=30,bSizes=[192,50000],lRate=0.001,alpha=[1,1],set_weights_to_zero=False):

   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   
   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   dirnameHYM = 'tetraquadricHYM_pg_with_'+str(free_coefficient_name)+'forLB_'+lbstring
   dirnameForMetric = 'tetraquadric_pg_with_'+str(free_coefficient_name)
   dirnameHarmonic = 'tetraquadricHarmonicH_pg'+str(free_coefficient_name)+'forLB_'+lbstring+nameOfBaseHF
   name = 'HFmodel_for_' + str(nEpochs) + '_' + str(bSizes[0]) + '_'+ str(bSizes[1])+  str(nlayer) + 'x' +str(nHidden)
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
   dataHF_train=dataHF_train.shuffle(buffer_size=1024).batch(bSizes[0])




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

   # need a last bias layer due to transition!
   initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.2)
   nn_HF = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,use_zero_network=True,kernel_initializer=initializer)
   nn_HF_zero = make_nn(n_in,n_outreal,nlayer,nHidden,act,lastbias=True,use_zero_network=True)
   
   HFmodel = HarmonicFormModel(nn_HF,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
   HFmodelzero = HarmonicFormModel(nn_HF_zero,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])

   if set_weights_to_zero:
      training_historyHF=0
   else:
      #print(HFmodel.model.weights[0])
      HFmodel.model=tf.keras.models.load_model(os.path.join(dirnameHarmonic,name))
      #print(HFmodel.model.weights[0])
      training_historyHF=np.load(os.path.join(dirnameHarmonic, 'trainingHistory-' + name +'.npz'),allow_pickle=True)['arr_0'].item()

   HFmodel.compile(custom_metrics=cmetricsHF)
   HFmodelzero.compile(custom_metrics=cmetricsHF)

   valzero=HFmodelzero.evaluate(dataHF_val_dict)
   valtrained=HFmodel.evaluate(dataHF_val_dict)
   #valzero = {key: value.numpy() for key, value in valzero.items()}
   #valtrained = {key: value.numpy() for key, value in valtrained.items()}

   metricsnames=HFmodel.metrics_names
   valzero = {metricsnames[i]: valzero[i] for i in range(len(valzero))}
   valtrained= {metricsnames[i]: valtrained[i] for i in range(len(valtrained))}
  
   print("zero network validation loss: ")
   print(valzero)
   print("validation loss for trained network: ")
   print(valtrained)
   print("ratio of trained to zero: " + str({key + " ratio": value/(valzero[key]+1e-8) for key, value in valtrained.items()}))


   averagediscrepancyinstdevs,_=compute_transition_pointwise_measure_section(HFmodel,tf.cast(dataHF["X_val"],float_dtype))
   print("average transition discrepancy in standard deviations (note, underestimate as our std.dev. ignores variation in phase): " + str(averagediscrepancyinstdevs))
   meanfailuretosolveequation,_,_=HYM_measure_val_with_H(HFmodel,dataHF)
   print("mean of difference/mean of absolute value of source, weighted by sqrt(g): " + str(meanfailuretosolveequation))
   print("\n\n")
   return HFmodel,training_historyHF




