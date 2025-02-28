from cymetric.config import real_dtype, complex_dtype
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
from yukawas.OneAndTwoFormsForLineBundlesModel1 import *
#from generate_and_train_all_nnsHOLO import *
from SearchFScalculation import *

import numpy as np
import logging
import pickle
import csv

print("running")
class DisableLogger():
    def __enter__(self):
        logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)
print("running2")

with DisableLogger():
    import sys
    seed = int(sys.argv[1])
    # Function to calculate masses
    def calculate_masses_for_descent(free_coefficient):
        return calculate_masses("/mnt/extraspace/kitft/FubiniStudyRandomdata/", free_coefficient, npts, blockprint=True, force_generate=True, seed=seed)[0]

    # Function to calculate loss
    def calculate_loss(masses):
        smalllossfor0closeto1 = largeupweight * (np.abs(np.log(masses[0])))
        smalllossfor1smallerthan0 = largehierarchy * np.log(masses[1] / masses[0])
        return np.array([smalllossfor0closeto1, smalllossfor1smallerthan0])

    # Random sampler function
    def random_sampler(n_iter=10):
        rng = np.random.default_rng(seed = seed)  # Create a random generator instance
        masses = []
        losses = []
        store_vecs = []
        best_loss = None
        best_mass = None
        best_v = None

        for t in range(1, n_iter + 1):
            v = rng.normal(size=(21,), scale=10) + 1j * rng.normal(size=(21,), scale=10)
            #print(v)
            current_mass = calculate_masses_for_descent(v)
            current_loss = np.sum(calculate_loss(current_mass))
            masses.append(current_mass)
            losses.append(current_loss)
            store_vecs.append(v)

            print(f'Iteration {t}: current loss: {current_loss}, current mass: {current_mass}')
            if best_loss is None or current_loss < best_loss:
                best_loss = current_loss
                best_mass = current_mass
                best_v = v
                print(f'New best loss: {best_loss}, corresponding mass: {best_mass}')
                print(f'Corresponding v vector: {best_v}')

        return masses, losses, store_vecs, best_loss, best_mass, best_v

    npts = 100000

    print("loss ratios")

    largeupweight = 10
    largehierarchy = 0.1
    print("large up ", largeupweight)
    print("large hierarchy ", largehierarchy)

    # Run the random sampler
    n_iterations = 1000  # Adjust the number of iterations as needed
    masses, losses, store_vecs, best_loss, best_mass, best_v = random_sampler(n_iter=n_iterations)

# Save masses, losses, and store_vecs into a pickle file
with open('random_NEW_masses_losses_store_vecs.pkl', 'wb') as f:
    pickle.dump([masses, losses, store_vecs], f)

# Print the lowest loss and mass found
print(f'Lowest loss found: {best_loss}')
print(f'Corresponding mass: {best_mass}')
print(f'Corresponding v vector: {best_v}')

# Write the results to a CSV file
filename = 'random_NEW_search_results.csv'
with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Best Loss', 'Best Mass', 'Best v'])
    writer.writerow([best_loss, best_mass.tolist(), best_v.tolist()])

