from cymetric.config import float_dtype, complex_dtype
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

tf.get_logger().setLevel('ERROR')


from cymetric.models.tfmodels import PhiFSModel, MultFSModel, FreeModel, MatrixFSModel, AddFSModel, PhiFSModelToric, MatrixFSModelToric
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.callbacks import SigmaCallback, KaehlerCallback, TransitionCallback, RicciCallback, VolkCallback, AlphaCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, RicciLoss, VolkLoss, TotalLoss

from NewCustomMetrics import *
from HarmonicFormModel import *
from BetaModel import *
from laplacian_funcs import *
from OneAndTwoFormsForLineBundles import *
#from generate_and_train_all_nnsHOLO import *
from SearchFScalculation import *


# import logging
# logging.basicConfig(level='warning')
import logging

print("running")
class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)
print("running2")

with DisableLogger():
    #npts=100000
    #def calculate_masses_for_descent(free_coefficient):
    #    # your implementation here
    #    #mass= np.abs(np.array([free_coefficient[0], free_coefficient[1]]))
    #    #return mass + np.array([0.001,0.001])
    #    return calculate_masses("/mnt/extraspace/kitft/FubiniStudyGDdata/",free_coefficient,npts,blockprint=True,force_generate=True,seed=0)[0]

    #def calculate_loss(masses):
    #    smalllossfor0closeto1=10*(np.abs(np.log(masses[0])))
    #    smalllossfor1smallerthan0=np.log(masses[1]/masses[0])
    #    print("1loss/ratioloss: " + str(smalllossfor0closeto1) + "," + str(smalllossfor1smallerthan0))
    #    return np.array([smalllossfor0closeto1,smalllossfor1smallerthan0])

    #def gd_step(v, vectors, learning_rate):#, scale):
    #    current_mass=calculate_masses_for_descent(v)
    #    current_loss = np.sum(calculate_loss(current_mass))
    #    print('current loss: '+str(current_loss) + " current mass " + str(current_mass))
    #    losses = np.zeros(len(vectors))
    #    deltas = np.zeros(len(vectors))

    #    # for i in range(len(vectors)):
    #    #     u = vectors[i]
    #    #     masses_plus = calculate_masses_for_descent(v + u)
    #    #     loss_plus = calculate_loss(masses_plus)
    #    #     losses[i] = loss_plus
    #    for i, u in enumerate(vectors):
    #        masses_plus = calculate_masses_for_descent(v + u)
    #        loss_plus = np.sum(calculate_loss(masses_plus))
    #        losses[i] = loss_plus
    #        deltas[i] = loss_plus - current_loss
    #    gradient=np.zeros_like(v)
    #    for i in range(len(vectors)):
    #        gradient += deltas[i] * vectors[i]
    #    adjusted_lr = learning_rate# * np.linalg.norm(gradient)  # set the step size proportional to the gradient magnitude
    #    new_v = v - adjusted_lr * gradient*1/np.sqrt(np.abs(gradient)**2+0.1)
    #    print("\n")
    #    print(v)
    #    print("shift by: ")
    #    print(- adjusted_lr * gradient)
    #    old_mass=current_mass
    #    return new_v, old_mass


    #    gradient = np.zeros(len(v))

    #    for i in range(len(vectors)):
    #        gradient[i] = (losses[i] - losses[-1]) / np.linalg.norm(vectors[i] - v)
    #    #scale += gradient**2
    #    #adaptive scaling
    #    adjusted_lr = learning_rate# might want to change this
    #    print("\n")
    #    print(v)
    #    print("shift by: ")
    #    print(- adjusted_lr * gradient)
    #    new_v = v - adjusted_lr * gradient
    #    old_loss = current_loss
    #    return new_v, losses[-1],old_loss

    #def gradient_descent(v_init, lr_init=1, decay=0.9, n_iter=10,n_vecs=1):
    #    v = v_init.copy()
    #    lr = lr_init
    #    #scale = np.ones(len(v)) * 1e-8  # small number to prevent division by zero
    #    losses = []
    #    masses = []
    #    store_vecs=[]
    #    store_vecs.append(v)
    #    #losses.append(calculate_loss(calculate_masses_for_descent(v)))
    #    for i in range(n_iter):
    #        vectors = np.random.randn(n_vecs, v.shape[0])+1j*  np.random.randn(n_vecs, v.shape[0])# initialization of the n_vecs unit vectors
    #        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    #        v,old_mass = gd_step(v, vectors, lr)#, scale)
    #        lr *= decay
    #        masses.append(old_mass)
    #        old_loss=calculate_loss(old_mass)
    #        losses.append(old_loss)
    #        store_vecs.append(v)
    #    return masses, losses,store_vecs

    #
    #    print('start gradient descent')
    ##v_init = np.random.normal(size=(21,),scale=10)+1j* np.random.normal(size=(21,),scale=10)
    #v_init = np.array([ 15.00828144-22.7926254j, -34.49138809 -0.41259758j, 7.90203097 +9.09126749j, 0.92153488 -6.55882684j, 7.68666651+33.46654982j, 13.80212759 +4.12013117j, 8.6728578 +4.4924097j, 4.30219549-16.59782002j, 9.35019659+16.20962592j, -6.70609042 -5.71985406j, -10.27364939 +2.49523399j, -7.91152919+13.62744773j, -0.58001754 +5.55570609j, 4.39427238 -7.90145148j, -6.05974857+11.81643379j, 1.64424751 +6.2093194j, -9.23274155 -6.67949531j, -20.76608188 +1.83322725j, -4.91590203 -4.60850461j, -7.07087404 -6.74012598j, -22.50688078 -7.92583167j])
    ##masses, losses,store_vecs = gradient_descent(v_init,lr_init=1,decay=1,n_iter=100,n_vecs=40)
    #masses, losses,store_vecs = gradient_descent(v_init,lr_init=3,decay=1,n_iter=100,n_vecs=40)
    def calculate_masses_for_descent(free_coefficient):
        # your implementation here
        return calculate_masses("/mnt/extraspace/kitft/FubiniStudyGDdata/", free_coefficient, npts, blockprint=True, force_generate=True, seed=0)[0]

    def get_all_unit_complex_vecs(v):
        N = v.shape[0]
        # Create an identity matrix of shape (N, N)
        identity_matrix = np.eye(N)
        
        # Generate the real part unit step vectors by adding the identity matrix
        real_step_vectors = v*0+ identity_matrix
        
        # Generate the imaginary part unit step vectors by adding the identity matrix multiplied by 1j
        imag_step_vectors = v*0 + identity_matrix * 1j
        
        return np.concatenate((real_step_vectors,imag_step_vectors),axis=0)


    def calculate_loss(masses):
        smalllossfor0closeto1 = largeupweight* (np.abs(np.log(masses[0])))
        smalllossfor1smallerthan0 = largehierarchy* np.log(masses[1] / masses[0])
        print("1loss/ratioloss: " + str(smalllossfor0closeto1) + "," + str(smalllossfor1smallerthan0))
        return np.array([smalllossfor0closeto1, smalllossfor1smallerthan0])
    
    def adam_step(v, vectors, m, v_adam, t, alpha=1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        current_mass = calculate_masses_for_descent(v)
        current_loss = np.sum(calculate_loss(current_mass))
        print('current loss: ' + str(current_loss) + " current mass " + str(current_mass))
        losses = np.zeros(len(vectors))
        deltas = np.zeros(len(vectors))
    
        for i, u in enumerate(vectors):
            masses_plus = calculate_masses_for_descent(v + u)
            loss_plus = np.sum(calculate_loss(masses_plus))
            losses[i] = loss_plus
            deltas[i] = loss_plus - current_loss
    
        gradient = np.zeros_like(v)
        for i in range(len(vectors)):
            gradient += deltas[i] * vectors[i]
    
        m = beta1 * m + (1 - beta1) * gradient
        v_adam = beta2 * v_adam + (1 - beta2) * (gradient ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v_adam / (1 - beta2 ** t)
        new_v = v - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    
        print("\n")
        print(v)
        print("shift by: ")
        print(- alpha * m_hat / (np.sqrt(v_hat) + epsilon))
        old_mass = current_mass
        return new_v, old_mass, m, v_adam
    
    def gradient_descent(v_init,adamLR=1, n_iter=10, n_vecs=1,norm_vecs=1):
        v = v_init.copy()
        m = np.zeros_like(v)
        v_adam = np.zeros_like(v)
        losses = []
        masses = []
        store_vecs = []
        store_vecs.append(v)
    
        for t in range(1, n_iter + 1):
            #vectors = np.random.randn(n_vecs, v.shape[0]) + 1j * np.random.randn(n_vecs, v.shape[0])
            vectors=get_all_unit_complex_vecs(v)
            vectors=vectors*norm_vecs
            #vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            v, old_mass, m, v_adam = adam_step(v, vectors, m, v_adam, t,alpha=adamLR)
            masses.append(old_mass)
            old_loss = calculate_loss(old_mass)
            losses.append(old_loss)
            store_vecs.append(v)
    
        return masses, losses, store_vecs
    
    print('start gradient descent')
    ##v_init = np.random.normal(size=(21,),scale=10)+1j* np.random.normal(size=(21,),scale=10)
    #v_init = np.array([ 15.00828144-22.7926254j, -34.49138809 -0.41259758j, 7.90203097 +9.09126749j, 0.92153488 -6.55882684j, 7.68666651+33.46654982j, 13.80212759 +4.12013117j, 8.6728578 +4.4924097j, 4.30219549-16.59782002j, 9.35019659+16.20962592j, -6.70609042 -5.71985406j, -10.27364939 +2.49523399j, -7.91152919+13.62744773j, -0.58001754 +5.55570609j, 4.39427238 -7.90145148j, -6.05974857+11.81643379j, 1.64424751 +6.2093194j, -9.23274155 -6.67949531j, -20.76608188 +1.83322725j, -4.91590203 -4.60850461j, -7.07087404 -6.74012598j, -22.50688078 -7.92583167j])

    npts = 100000
    #ADAMalpha=1
    import sys
    ADAMalpha= float(sys.argv[1])
    print(ADAMalpha)
    print("loss ratios")
    
    largeupweight=10
    largehierarchy=10#40
    print("large up ", largeupweight)
    print("large hierarchy ", largehierarchy)

    #loss of -50
    #v_init = np.array([ 11.78787674-16.34409609j ,-45.60731166 -2.46244483j,  -9.43302845 +5.58654545j,  -2.23424302 -8.33410839j,   7.11585836+31.93077287j,  17.29254427 +7.07834338j,   2.92330254 +2.01063257j, -11.34588713-14.77055182j,  19.88920919 +4.37410779j,  16.93759686 +5.86178799j,  -7.8828487  +7.82099713j, -16.40349557+14.42016994j,   7.11309623 +8.59500469j,  -7.80110696 -9.07606873j, -16.21806818+12.71431307j,  10.09339568 +2.95407145j,  -8.83648317 -7.10403135j, -24.61684332 +6.80127508j, -12.41249715 +2.54437671j,  -9.76505578 +1.0402238j, -13.92808567-10.07302649j])    
    v_init = np.array([ 11.82806007-16.87909394j, -48.40203141 -3.35266795j,  -8.15758    +5.67042092j,  -2.11507701 -8.49346115j,  10.77047873+32.18429129j,  17.80993581 +7.03926506j,  -0.12584165 +1.94674417j, -10.37474987-15.56643452j,  21.97571714 +5.5696716j,   14.86964219 +6.05493062j, -12.01091177 +6.77290169j, -15.48572189+14.14145195j,   7.33829674 +7.49588984j,  -7.43069639-10.48879631j, -15.16788862+12.26074463j,   7.83502287 +2.55395022j,  -7.37636452 -6.73701307j, -24.0347079  +6.95712915j, -13.22139922 +1.75119869j, -11.88410181 +0.4562674j, -14.20717211-11.70630235j])
    masses, losses,store_vecs = gradient_descent(v_init,adamLR=ADAMalpha,n_iter=100,n_vecs=40,norm_vecs=0.1)    
    #masses, losses,store_vecs = gradient_descent(v_init,n_iter=100,n_vecs=40)


#save masses, losses and store_vecs all in one file
#save masses, losses and store_vecs all in one file
import pickle
with open('NEW_masses_losses_store_vecs.pkl', 'wb') as f:
    pickle.dump([masses,losses,store_vecs], f)

#write to a CSV file all relevant details: free coefficient, number of epochs for each type, size of the networks, number of points for each, the physical Yukawas, the singular values of the physical Yukawas,
import csv
filename = 'NEW_search_results.csv'
with open(filename, mode='a') as file:
    writer = csv.writer(file)
    writer.writerow([masses[-10:-1],store_vecs[-10:-1]])
    file.close()
