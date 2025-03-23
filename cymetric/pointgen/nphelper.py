""" 
A collection of various numpy helper functions.
"""
import os as os
import numpy as np
import pickle
import itertools as it
from sympy import LeviCivita
from cymetric.config import real_dtype, complex_dtype
import cProfile
import gc
import pstats
import os

import math


def get_levicivita_tensor(dim):
    r"""Computes Levi-Civita tensor in dim dimensions.

    Conventions are zero for same indices, 1 for even permutations
    and -1 for odd permutations.

    Args:
        dim (int): dimension

    Returns:
        ndarray([dim, ..., dim], np.float64): Levi-Civita tensor.
    """
    lc = np.zeros(tuple([dim for _ in range(dim)]))
    for t in it.permutations(range(dim), r=dim):
        lc[t] = LeviCivita(*t)
    return lc


def conf_to_monomials(conf):
    r"""Creates monomials basis from configuration matrix.

    Example:
        Take CICY with ambient space P1xP3

        >>> conf = np.array([[1,2],[1,2]])
        >>> monomials = conf_to_monomials(conf)

    Args:
        conf (ndarray([nProj,nHyper], int)): Configuration matrix.

    Returns:
        list(nHyper,ndarray([nMonomials, nVars], int)): Monomial basis for
            each hypersurface.
    """
    ambient = np.sum(conf, axis=-1)-1
    monomials = []
    for h in np.transpose(conf):
        tmp_m = []
        for i, p in enumerate(ambient):
            tmp_m += [np.array(list(generate_monomials(p+1, h[i])))]
        mbasis = []
        for c in it.product(*tmp_m, repeat=1):
            mbasis += [np.concatenate(c)]
        monomials += [np.array(mbasis)]
    return monomials


def generate_monomials(n, deg):
    r"""Yields a generator of monomials with degree deg in n variables.

    Args:
        n (int): number of variables
        deg (int): degree of monomials

    Yields:
        generator: monomial term
    """
    if n == 1:
        yield (deg,)
    else:
        for i in range(deg + 1):
            for j in generate_monomials(n - 1, deg - i):
                yield (i,) + j

def _prepare_dataset_batched(point_gen, batch_n_p, ltails, rtails, selected_t_val = None):
    r"""Prepares a batch of data from point_gen without splitting or normalizing.

    Returns:
        tuple: (points, weights, omega)
    """
    new_np = int(round(batch_n_p/(1-ltails-rtails)))
    import time
    start_time = time.time()
    pwo = point_gen.generate_point_weights(new_np, omega=True, selected_t_val=selected_t_val)
    print("generate_point_weights took", time.time() - start_time, "seconds")
    start_time = time.time()
    if len(pwo) < new_np:
        print(f"Generating more points as only recieved {len(pwo)} after asking for {new_np}")
        new_np = int((new_np-len(pwo))/(len(pwo))*new_np + 100)
        pwo2 = point_gen.generate_point_weights(new_np, omega=True, selected_t_val=selected_t_val)
        pwo = np.concatenate([pwo, pwo2], axis=0)
    new_np = len(pwo)
    sorted_weights = np.sort(pwo['weight'])
    lower_bound = sorted_weights[round(ltails*new_np)]
    upper_bound = sorted_weights[round((1-rtails)*new_np)-1]
    mask = np.logical_and(pwo['weight'] >= lower_bound, pwo['weight'] <= upper_bound)
    weights = np.expand_dims(pwo['weight'][mask], -1)
    omega = np.expand_dims(pwo['omega'][mask], -1)
    omega = np.real(omega * np.conj(omega))
    points = pwo['point'][mask]
    pullbacks = point_gen.pullbacks(points)
    return points, weights, omega, pullbacks

def _prepare_dataset_batched_for_mp(point_gen, batch_n_p, ltails, rtails, seed : int = None):
    r"""Prepares a batch of data from point_gen without splitting or normalizing.

    Returns:
        tuple: (points, weights, omega)
    """
    new_np = int(round(batch_n_p/(1-ltails-rtails)))
    import time
    start_time = time.time()
    if seed is not None:
        point_gen._set_seed(seed)
    pwo = point_gen.generate_point_weights(new_np, omega=True, selected_t_val=None, use_quadratic_method = True)
    print("generate_point_weights took", time.time() - start_time, "seconds")
    start_time = time.time()
    if len(pwo) < new_np:
        print(f"Generating more points as only recieved {len(pwo)} after asking for {new_np}")
        new_np = int((new_np-len(pwo))/(len(pwo))*new_np + 100)
        pwo2 = point_gen.generate_point_weights(new_np, omega=True, selected_t_val=None, use_quadratic_method = True)
        pwo = np.concatenate([pwo, pwo2], axis=0)
    new_np = len(pwo)
    sorted_weights = np.sort(pwo['weight'])
    lower_bound = sorted_weights[round(ltails*new_np)]
    upper_bound = sorted_weights[round((1-rtails)*new_np)-1]
    mask = np.logical_and(pwo['weight'] >= lower_bound, pwo['weight'] <= upper_bound)
    weights = np.expand_dims(pwo['weight'][mask], -1)
    omega = np.expand_dims(pwo['omega'][mask], -1)
    omegasquared = np.real(omega * np.conj(omega))
    points = pwo['point'][mask]
    pullbacks = pwo['pullbacks'][mask]
    gc.collect()
    return points, weights, omegasquared, pullbacks


def prepare_dataset(point_gen, n_p, dirname, n_batches=None, val_split=0.1, ltails=0, rtails=0, normalize_to_vol_j=True,average_selected_t = False, shuffle_points = True):
    r"""Prepares training and validation data from point_gen in batches.

    Note:
        The dataset will be saved in `dirname/dataset.npz`.
        Data is generated in batches to reduce memory usage.

    Args:
        point_gen (PointGenerator): Any point generator.
        n_p (int): Total number of points.
        dirname (str): Directory name to save data.
        n_batches (int, optional): Number of batches to split the data generation.
            Defaults to n_p//300000 (at least 1).
        val_split (float, optional): Train-val split. Defaults to 0.1.
        ltails (float, optional): Discarded % on the left tail of weight distribution.
        rtails (float, optional): Discarded % on the right tail of weight distribution.
        normalize_to_vol_j (bool, optional): Normalize such that
            ∫_X det(g) = ∑_p det(g) * w|_p  = d^{ijk} t_i t_j t_k.
            Defaults to True.

    Returns:
        np.float: kappa = vol_k / vol_cy computed from the combined data.
    """
    use_quadratic_method = point_gen.use_quadratic_method
    use_jax = point_gen.use_jax
    do_multiprocessing = point_gen.do_multiprocessing
    if use_quadratic_method ==True and do_multiprocessing ==False:
        raise ValueError("use_quadratic_method is True, but do_multiprocessing is False. This is not allowed.")
    # Set USE_PROFILER=1 in environment to enable
    use_profiler = os.environ.get('USE_PROFILER', '0') == '1'
    number_ambients = len(point_gen.ambient)
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
    except:
        print(f"Failed to create directory {dirname}: continuing?")
        pass

    if not do_multiprocessing:
        if n_batches is None and n_p > 300000:
            n_batches = n_p // 300000 if n_p // 300000 > 0 else 1
        elif n_batches is None:
            n_batches = 1
        if average_selected_t==True:
            # Ensure n_batches is divisible by number_ambients
            if n_batches % number_ambients != 0:
                n_batches = n_batches * number_ambients // math.gcd(n_batches, number_ambients)
            fixed_selected_t_val = None
        elif isinstance(average_selected_t, int) and 0 <= average_selected_t < number_ambients:
            fixed_selected_t_val = average_selected_t
        elif average_selected_t==False:
            fixed_selected_t_val = np.argmax(point_gen.ambient)
        else:
            raise ValueError("average_selected_t must be a boolean or an integer between 0 and number_ambients-1, inclusive.")

        if n_batches > 1:
            print(f'Generating {n_p} points using {n_batches} batches')
        base = n_p // n_batches
        rem = n_p % n_batches
        all_points, all_weights, all_omega, all_pullbacks = [], [], [], []
    
        for i in range(n_batches):
            selected_t_val = (i % number_ambients) if (average_selected_t==True) else fixed_selected_t_val
            if n_batches > 1:
                print(f'Generating {base + (1 if i < rem else 0)} points on {i}th batch with solving for P1 ={selected_t_val} of ({(number_ambients)})')
            else:
                print(f'Generating {n_p} points with solving for P1 ={selected_t_val} of ({(number_ambients)}). No batching.')
            batch_n = base + (1 if i < rem else 0)

            if i == 0 and use_profiler:
                # Profile the function
                profiler = cProfile.Profile()
                profiler.enable()
                pts, w, om, pb = _prepare_dataset_batched(
                    point_gen, batch_n, ltails, rtails, selected_t_val=selected_t_val)
                profiler.disable()

                # Print sorted results
                stats = pstats.Stats(profiler).sort_stats('cumtime')
                print("--------------------------------PROFILING--------------------------------")
                stats.print_stats(20)  
                print("--------------------------------PROFILED--------------------------------")
            else:
                pts, w, om, pb = _prepare_dataset_batched(
                    point_gen, batch_n, ltails, rtails, selected_t_val=selected_t_val)
            all_points.append(pts)
            all_weights.append(w)
            all_omega.append(om)
            all_pullbacks.append(pb)
        # Concatenate all batches
        if shuffle_points:
            print("Shuffling points")
            # Process batches in groups of 4
            for i in range(0, len(all_points), 4):
                end_idx = min(i + 4, len(all_points))
                # Concatenate this group of batches
                group_points = np.concatenate(all_points[i:end_idx], axis=0)
                group_weights = np.concatenate(all_weights[i:end_idx], axis=0)
                group_omega = np.concatenate(all_omega[i:end_idx], axis=0)
                group_pullbacks = np.concatenate(all_pullbacks[i:end_idx], axis=0)

                # Shuffle the concatenated data
                indices = np.random.permutation(len(group_points))
                group_points = group_points[indices]
                group_weights = group_weights[indices]
                group_omega = group_omega[indices]
                group_pullbacks = group_pullbacks[indices]

                # Replace the original batches with equal-sized chunks of the shuffled data
                start_idx = 0
                for j in range(i, end_idx):
                    batch_size = len(all_points[j])
                    all_points[j] = group_points[start_idx:start_idx + batch_size]
                    all_weights[j] = group_weights[start_idx:start_idx + batch_size]
                    all_omega[j] = group_omega[start_idx:start_idx + batch_size]
                    all_pullbacks[j] = group_pullbacks[start_idx:start_idx + batch_size]
                    start_idx += batch_size
        else:
            if average_selected_t==True:
                raise ValueError("Shuffling points must be done when average_selected_t is True")

    elif do_multiprocessing:
        print("Using multiprocessing with fork")
        # Set JAX to use only 1 thread per process to avoid oversubscription
        # Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
        # https://github.com/google/jax/issues/743.
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        
        all_points, all_weights, all_omega, all_pullbacks = [], [], [], []
        

        # import pickle
        # import cloudpickle
        # pickle.dumps = cloudpickle.dumps
        # pickle.loads = cloudpickle.loads
        # pickle.Pickler = cloudpickle.Pickler
        # from multiprocessing import reduction
        # reduction.ForkingPickler = cloudpickle.Pickler
        # import multiprocessing
        # # Replace the default serializer
        # #ForkingPickler.dumps = cloudpickle.dumps
        # # print(f"Generating {n_p} points using {n_batches} batches, and with {n_cpus} CPUs")
        # try:
        #     multiprocessing.set_start_method('fork', force=False)
        # except Exception as e:
        #     print("Failed to set start method to fork, error: ", e)
        # from contextlib import nullcontext
        #with nullcontext():
        #Import cloudpickle
        #Import multiprocessing as mp
        #From multiprocessing.reduction import ForkingPickler
        #Import numpy as np
        # Use joblib for efficient parallel processing with numpy arrays
        from joblib import Parallel, delayed
        import multiprocessing
        n_cpus = 1
        try:
            # Check SLURM environment variables
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_JOB_CPUS_PER_NODE')
            if slurm_cpus:
                available_cpus = int(slurm_cpus)-1
                print(f"we are on a SLURM job, with {available_cpus} CPUs available, using that many (nb we have subtracted 1)")
                n_cpus = available_cpus
            else:
                n_cpus = multiprocessing.cpu_count() - 1
                n_cpus = max(1, n_cpus)  # Ensure at least 1 CPU is used
                print(f"we are not on a SLURM job, using all available CPUs (nb we have subtracted 1): {n_cpus}")
        except Exception as e:
            print("Exception in core counting, falling back to 1 CPU", e)
            pass
        numpy_seed = np.random.get_state()[1][0]# use the same seed as numpy for the jax seed

        n_batches = (n_p//10000) if n_p//10000 > 0 else 1
        n_batches = n_cpus*(int(np.ceil(n_batches/n_cpus)))
        batch_n = n_p//n_batches

        random_seeds = np.random.RandomState(numpy_seed).randint(0, 2**32, size=n_batches)
        print(f"attempting to parallelise {n_batches} batches over at most {n_cpus} processes, each doing {batch_n} points (so /8 random sections). Each batch has a random seed.") 
        # Execute the batch processing in parallel
        results = Parallel(n_jobs=n_cpus, prefer="processes")(
            delayed(_prepare_dataset_batched_for_mp)(point_gen, batch_n, ltails, rtails, seed)
            for seed in random_seeds
        )
        
        # Collect results
        for pts, w, om, pb in results:
            all_points.append(pts)
            all_weights.append(w)
            all_omega.append(om)
            all_pullbacks.append(pb)
        # Not using multiprocessing pool
        # results = []
        # for _ in range(n_batches):
        #     result = _prepare_dataset_batched_for_mp(point_gen, batch_n, ltails, rtails)
        #     results.append(result)
            
        # # Unpack results
        # for pts, w, om, pb in results:
        #     all_points.append(pts)
        #     all_weights.append(w)
        #     all_omega.append(om)
        #     all_pullbacks.append(pb)

        os.environ["XLA_FLAGS"] = ""
    
    all_points = np.concatenate(all_points, axis=0)
    all_weights = np.concatenate(all_weights, axis=0)
    all_omega = np.concatenate(all_omega, axis=0)
    all_pullbacks = np.concatenate(all_pullbacks, axis=0)

    # Normalize weights if requested (after all batches are combined)
    if normalize_to_vol_j:
        print("normalizing to vol_j")
        fs_ref = point_gen.fubini_study_metrics(all_points, vol_js=np.ones_like(point_gen.kmoduli))
        fs_ref_pb = np.einsum('xai,xij,xbj->xab', all_pullbacks, fs_ref, np.conj(all_pullbacks))
        aux_weights = all_omega.flatten() / all_weights.flatten()
        norm_fac = point_gen.vol_j_norm / np.mean(np.real(np.linalg.det(fs_ref_pb)) / aux_weights)
        all_weights = norm_fac * all_weights
        print("normalized to vol_j")

    # Split into train and validation sets after all processing
    n_total = len(all_points)
    t_i = int((1-val_split) * n_total)
    
    # Create train/val datasets
    X_train = np.concatenate((all_points[:t_i].real, all_points[:t_i].imag), axis=-1)
    y_train = np.concatenate((all_weights[:t_i], all_omega[:t_i]), axis=1)
    X_val = np.concatenate((all_points[t_i:].real, all_points[t_i:].imag), axis=-1)
    y_val = np.concatenate((all_weights[t_i:], all_omega[t_i:]), axis=1)
    
    # Generate pullbacks for train and validation sets
    train_pullbacks = all_pullbacks[:t_i]
    val_pullbacks = all_pullbacks[t_i:]
    
    # Save the dataset
    np.savez_compressed(os.path.join(dirname, 'dataset'),
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        train_pullbacks=train_pullbacks,
                        val_pullbacks=val_pullbacks)
    print('computing kappa')
    return point_gen.compute_kappa(all_points, all_weights, all_omega)


def prepare_basis(point_gen, dirname, kappa=1.):
    r"""Prepares monomial basis for NNs from point_gen as .npz dict.

    Args:
        point_gen (point_gen): point generator
        dirname (str): dir name to save
        kappa (float): kappa value (ratio of Kahler and CY volume)

    Returns:
        int: 0
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    np.savez_compressed(os.path.join(dirname, 'basis'),
                        DQDZB0=point_gen.BASIS['DQDZB0'],
                        DQDZF0=point_gen.BASIS['DQDZF0'],
                        AMBIENT=point_gen.ambient,
                        KMODULI=point_gen.kmoduli,
                        NFOLD=point_gen.nfold,
                        NHYPER=point_gen.nhyper,
                        INTNUMS=point_gen.intersection_tensor,
                        KAPPA=kappa
                        )
    return 0


def prepare_basis_pickle(point_gen, dirname, kappa=1.):
    r"""Prepares pickled monomial basis for NNs from PointGenerator.

    Args:
        point_gen (PointGenerator): Any point generator.
        dirname (str): dir name to save
        kappa (float): kappa value (ratio of Kahler and CY volume)

    Returns:
        int: 0
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    new_dict = point_gen.BASIS
    new_dict['NFOLD'] = point_gen.nfold
    new_dict['AMBIENT'] = point_gen.ambient
    new_dict['KMODULI'] = point_gen.kmoduli
    new_dict['NHYPER'] = point_gen.nhyper
    new_dict['INTNUMS'] = point_gen.intersection_tensor
    new_dict['KAPPA'] = kappa
    
    with open(os.path.join(dirname, 'basis.pickle'), 'wb') as handle:
        pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 0


def get_all_patch_degrees(glsm_charges, patch_masks):
    r"""Computes the degrees of every coordinate for each patch to rescale such
    that the largest coordinates will be 1+0.j.

    Args:
        glsm_charges (ndarray([nscaling, ncoords], int)): GLSM charges.
        patch_masks (ndarray([npatches, ncoords], bool)): Patch masks with
            True at each coordinates, which is not allowed to vanish.

    Returns:
        ndarray([npatches, ncoords, ncoords], int): degrees
    """
    npatches, ncoords = np.shape(patch_masks)
    all_patch_degrees = np.zeros((npatches, ncoords, ncoords), dtype=int)
    for i in range(npatches):
        all_patch_degrees[i] = np.eye(ncoords, dtype=int)
        patch_coords = np.where(patch_masks[i])[0]
        for j in range(ncoords):
            factors = np.linalg.solve(
                glsm_charges[:, patch_coords], glsm_charges[:, j].T)
            if not np.allclose(factors, np.round(factors)):
                print('WARNING GLSM: NO INTEGER COEFFICIENTS.')
            for l, k in enumerate(patch_coords):
                all_patch_degrees[i, j, k] -= np.round(factors[l]).astype(int)
    return all_patch_degrees


def compute_all_w_of_x(patch_degrees, patch_masks, dim_cy = 3):
    r"""Computes monomials to reexpress the good coordinates in one patch in 
    terms of the good coordinates in another patch with respect to the 
    homogeneous ambient space coordinates.

    Args:
        patch_degrees (ndarray([npatches, ncoords, ncoords], int)): See also
            :py:func:`get_all_patch_degrees()`.
        patch_masks (ndarray([npatches, ncoords], bool)): Patch masks with
            True at each coordinates, which is not allowed to vanish.
        dim_cy (int, optional): Dimension of the Calabi-Yau. Defaults to 3.

    Returns:
        tuple: w_of_x, del_w_of_x, del_w_of_z
    """
    npatches, ncoords = np.shape(patch_masks)    
    w_of_x = np.zeros(
        (ncoords, npatches, npatches, dim_cy, dim_cy), dtype=int)
    del_w_of_x = np.zeros(
        (ncoords, npatches, npatches, dim_cy, dim_cy, dim_cy), dtype=int)
    del_w_of_z = np.zeros(
        (ncoords, npatches, npatches, dim_cy, dim_cy, ncoords), dtype=int)
    # TODO: Add a warning for when the array becomes too large.
    # NOTE: There will be many zeros.
    for i in range(ncoords):
        allowed_patches = np.where(~patch_masks[:,i])[0]
        for j in allowed_patches:
            for k in allowed_patches:
                # good coordinates in patch 1
                g1mask = np.ones(ncoords, dtype=bool)
                g1mask[patch_masks[j]] = False
                g1mask[i] = False
                # good coordinates in patch 2
                g2mask = np.ones(ncoords, dtype=bool)
                g2mask[patch_masks[k]] = False
                g2mask[i] = False
                # rewrite each good coordinate in patch 2 in terms of patch2
                for l, v in enumerate(patch_degrees[k][g2mask]):
                    coeff, _, _, _ = np.linalg.lstsq(
                        patch_degrees[j][g1mask].T, v, rcond=None)
                    if not np.allclose(coeff, np.round(coeff)):
                        print('WARNING W(X): NO INTEGER COEFFICIENTS.')
                    w_of_x[i, j, k, l] = np.round(coeff).astype(int)
                    # compute the derivative wrt to the g1 coordinates
                    del_w_of_x[i, j, k, l] = w_of_x[i, j, k, l] - np.eye(dim_cy, dtype=int)
                    # re-express everything in terms of degrees of the homogeneous
                    # ambient space coordinates
                    for m in range(dim_cy):
                        del_w_of_z[i, j, k, l, m] = np.einsum('j,ji', del_w_of_x[i, j, k, l, m], patch_degrees[j][g1mask])
    # w_of_x contains the derivative coefficients
    # del_w_of_x express g2 in terms of g1 coordinates
    # del_w_of_z express g2 in terms of homogeneous coordinates.
    return w_of_x, del_w_of_x, del_w_of_z
