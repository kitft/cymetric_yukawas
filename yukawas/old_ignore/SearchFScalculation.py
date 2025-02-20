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
import scipy
import sys

from NewCustomMetrics import *
from HarmonicFormModel import *
from BetaModel import *
from laplacian_funcs import *
from OneAndTwoFormsForLineBundles import *
from custom_networks import *



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



def generate_points_and_save_using_defaults(basedirname,free_coefficient,number_points,force_generate=False,seed=2021):
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
   pg._set_seed(seed)
   dirname = basedirname+'data/tetraquadric_pg_with_'+str(free_coefficient_name) 
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
   return pg
   
def create_untrained_nn(basedirname,free_coefficient):
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   dirname = basedirname+'data/tetraquadric_pg_with_'+str(free_coefficient_name) 
   BASIS = prepare_tf_basis(np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True))

   #nn_phi = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,stddev=stddev,use_zero_network=use_zero_network)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_phi_zero =BiholoModelFuncGENERAL(shapeofnetwork,BASIS,use_zero_network=True)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_phi_zero= lambda x: tf.zeros((tf.shape(x)[0]),dtype=real_dtype)
   #nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)
   phimodel = PhiFSModel(nn_phi_zero, BASIS, alpha=np.array([1,1,30,1,2.]))



   return phimodel
   
def create_untrained_nn_HYM(basedirname,free_coefficient,linebundleforHYM,alpha=[1,1],load_network=False):
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   dirname =basedirname+'data/tetraquadric_pg_with_'+str(free_coefficient_name) 
   BASIS = prepare_tf_basis(np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True))

   #initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=stddev)
   #nn_beta = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,stddev=stddev,use_zero_network=use_zero_network)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_beta_zero = BiholoModelFuncGENERAL(shapeofnetwork,BASIS,use_zero_network=True)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_beta = BiholoModelFuncGENERALnolog(shapeofnetwork,BASIS,activation=tfk.activations.gelu,stddev=stddev,use_zero_network=use_zero_network)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   #nn_beta_zero = BiholoModelFuncGENERALnolog(shapeofnetwork,BASIS,activation=tfk.activations.gelu,use_zero_network=True)#make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=use_zero_network)
   nn_beta_zero= lambda x: tf.zeros((tf.shape(x)[0]),dtype=real_dtype)
   #nn_phi_zero = make_nn(n_in,n_out,nlayer,nHidden,act,use_zero_network=True)

   betamodel= BetaModel(nn_beta_zero,BASIS, linebundleforHYM,alpha=alpha,norm = [1. for _ in range(2)])
   return betamodel
   
def create_untrained_nn_HF(basedirname,free_coefficient,linebundleforHYM,betamodel,functionforbaseharmonicform_jbar,alpha=[1,500],use_zero_network=False):
   
   nameOfBaseHF=functionforbaseharmonicform_jbar.__name__
   lbstring = ''.join(str(e) for e in linebundleforHYM)
   a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u=free_coefficient
   free_coeff_tuple=(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u)
   free_coefficient_name=hash(free_coeff_tuple)
   dirname = basedirname+'data/tetraquadric_pg_with_'+str(free_coefficient_name) 
   BASIS = prepare_tf_basis(np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True))

   #define a lambda funciton that returns a vector of zeros the size of the batch in its argument
   nn_HF_zero= lambda x: tf.zeros((tf.shape(x)[0]),dtype=real_dtype)
   HFmodel = HarmonicFormModel(nn_HF_zero,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
   #HFmodelzero = HarmonicFormModel(nn_HF_zero,BASIS,betamodel, linebundleforHYM,functionforbaseharmonicform_jbar,alpha=alpha,norm = [1. for _ in range(2)])
   return HFmodel

# # Disable
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')

# # Restore
# def enablePrint():
#     sys.stdout = sys.__stdout__


import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class DoNothing:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def calculate_masses(basedirname,free_coefficient,nPoints=300000,blockprint=True,force_generate=True,seed=2021):
    with (HiddenPrints() if blockprint else DoNothing()):
        #nPoints=300000
      #free_coefficient=1.000000004# when the coefficient is 1, ensure that it's 1., not 1 for the sake of the filename
      #nEpochsPhi=100
      nEpochsPhi=1
      nEpochsBeta=1
      nEpochsSigma=1
      depthPhi=2
      widthPhi=2#128 4 in the 1.0s
      depthBeta=2
      widthBeta=2
      depthSigma=2
      widthSigma=2 # up from 256
      alphabeta=[1,10]
      alphasigma1=[1,10] # down form 50
      alphasigma2=[1,80] # down from 200


      linebundleforHYM_02m20=np.array([0,2,-2,0]) 
      linebundleforHYM_110m2=np.array([1,1,0,-2]) 
      linebundleforHYM_m1m322=np.array([-1,-3,2,2]) 

      free_coefficient_name=hash(tuple(free_coefficient))
      print('set of coeffs')
      print(free_coefficient)
      print("hash")
      print(hash(tuple(free_coefficient)))

      pg=generate_points_and_save_using_defaults(basedirname,free_coefficient,number_points=nPoints,force_generate=force_generate,seed=seed)
      phimodel1=create_untrained_nn(basedirname,free_coefficient) 

      betamodel_02m20=create_untrained_nn_HYM(basedirname,free_coefficient,linebundleforHYM_02m20,alpha=alphabeta,load_network=False)
      betamodel_110m2=create_untrained_nn_HYM(basedirname,free_coefficient,linebundleforHYM_110m2,alpha=alphabeta,load_network=False)
      betamodel_m1m322=create_untrained_nn_HYM(basedirname,free_coefficient,linebundleforHYM_m1m322,alpha=alphabeta,load_network=False)

      def vFS_Q1(x):
          return getTypeIIs(x,phimodel1,'vQ1')
      def vFS_Q2(x):
          return getTypeIIs(x,phimodel1,'vQ2')
      def vFS_U1(x):
          return getTypeIIs(x,phimodel1,'vU1')
      def vFS_U2(x):
          return getTypeIIs(x,phimodel1,'vU2')

      HFmodel_vH=create_untrained_nn_HF(basedirname,free_coefficient,linebundleforHYM_02m20,betamodel_02m20,functionforbaseharmonicform_jbar_for_vH,alpha=alphasigma1)
      HFmodel_vQ3=create_untrained_nn_HF(basedirname,free_coefficient,linebundleforHYM_110m2,betamodel_110m2,functionforbaseharmonicform_jbar_for_vQ3,alpha=alphasigma1)
      HFmodel_vU3=create_untrained_nn_HF(basedirname,free_coefficient,linebundleforHYM_110m2,betamodel_110m2,functionforbaseharmonicform_jbar_for_vU3,alpha=alphasigma1)


      HFmodel_vQ1=create_untrained_nn_HF(basedirname,free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_Q1,alpha=alphasigma2)
      HFmodel_vQ2=create_untrained_nn_HF(basedirname,free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_Q2,alpha=alphasigma2)
      HFmodel_vU1=create_untrained_nn_HF(basedirname,free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_U1,alpha=alphasigma2)
      HFmodel_vU2=create_untrained_nn_HF(basedirname,free_coefficient,linebundleforHYM_m1m322,betamodel_m1m322,vFS_U2,alpha=alphasigma2)






      print("fixed by swapping Q12 and U12") 
      print("\n do analysis: \n\n\n")


      #pg,kmoduli=generate_points_and_save_using_defaults_for_eval(free_coefficient,300000)
      #Vol_from_dijk_J=pg.get_volume_from_intersections(kmoduli)
      #Vol_reference_dijk_with_k_is_1=pg.get_volume_from_intersections(np.ones_like(kmoduli)) 
      data=np.load(os.path.join(basedirname+'data/tetraquadric_pg_with_'+str(free_coefficient_name), 'dataset.npz'))
      n_p=nPoints
      volCY_from_Om=tf.reduce_mean(data['y_train'][:n_p,0])/6 #this is the actual volume of the CY computed from Omega.
      real_pts=tf.cast(data['X_train'][0:n_p],real_dtype)# cast to 32 bit
      pointsComplex=tf.cast(point_vec_to_complex(data['X_train'][0:n_p]),complex_dtype)# we have to cast to complex64 from complex128 sometimes. Just for safety. point_vec_to_complex does the obvious thing

      aux_weights=(data['y_train'][:n_p,0]/(data['y_train'][:n_p,1]))*(1/(6))### these are the appropriate 'flat measure' so sum_i aux_weights f is int_X f(x)
      #convertcomptoreal=lambda complexvec: tf.concat([tf.math.real(complexvec),tf.math.imag(complexvec)],-1) # this converts from complex to real

      mats=[]
      mets = phimodel1.fubini_study_pb(real_pts,ts=tf.cast(kmoduli,complex_dtype))
      dets = tf.linalg.det(mets)
      #print('vol')
      #print(tf.reduce_mean(aux_weights*dets))
      #this gives 8, as expected
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
      #print("volCY_from_Om: " + str(volCY_from_Om))
      omega_normalised_to_one=omega/tf.cast(np.sqrt(volCY_from_Om),complex_dtype) # this is the omega that's normalised to 1. Numerical factor missing for the physical Yukawas?
      #print("Check correctly normalised: this should be 1")
      #print(tf.reduce_mean(aux_weights*omega_normalised_to_one*tf.math.conj(omega_normalised_to_one)))
      m = [[0,0,0],[0,0,0],[0,0,0]]
      #mwoH = [[0,0,0],[0,0,0],[0,0,0]]

      tfsqrtandcast=lambda x: tf.cast(tf.math.sqrt(x),complex_dtype)

      # mwoH[0][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ1,vU1)*omega_normalised_to_one)
      # mwoH[0][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ1,vU2)*omega_normalised_to_one)
      # mwoH[0][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ1,vU3)*omega_normalised_to_one)
      # mwoH[1][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ2,vU1)*omega_normalised_to_one)
      # mwoH[1][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ2,vU2)*omega_normalised_to_one)
      # mwoH[1][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ2,vU3)*omega_normalised_to_one)
      # mwoH[2][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ3,vU1)*omega_normalised_to_one)
      # mwoH[2][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ3,vU2)*omega_normalised_to_one)
      # mwoH[2][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,xa,xb,xc->x",pg.lc,vH,vQ3,vU3)*omega_normalised_to_one)
      # print("without H")
      # print(np.round(np.array(mwoH)*10**4,1))

      m[0][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU1)*omega_normalised_to_one)
      m[0][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ1,vU2)*omega_normalised_to_one)
      m[0][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H2),vH,vQ1,vU3)*omega_normalised_to_one)
      m[1][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU1)*omega_normalised_to_one)
      m[1][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H3),vH,vQ2,vU2)*omega_normalised_to_one)
      m[1][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H3*H2),vH,vQ2,vU3)*omega_normalised_to_one)
      m[2][0] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU1)*omega_normalised_to_one)
      m[2][1] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H2*H3),vH,vQ3,vU2)*omega_normalised_to_one)
      m[2][2] = tf.reduce_mean(aux_weights * tf.einsum("abc,x,xa,xb,xc->x",pg.lc,tfsqrtandcast(H1*H2*H2),vH,vQ3,vU3)*omega_normalised_to_one)

      #2 sqrt2 comes out, 4 is |G|
      # 8 Sqrt30 is the group theoretic factor
      m= (1/(8*np.sqrt(30))) *   2*np.sqrt(2)/4* np.array(m)
      print('proper calculation')
      print(np.round(np.array(m)*10**4,1))
      #volCY=tf.reduce_mean(omega)
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
      u,s,v = np.linalg.svd(physical_yukawas)
      return s, physical_yukawas
