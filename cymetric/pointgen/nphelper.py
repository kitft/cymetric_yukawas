""" 
A collection of various numpy helper functions.
"""
import os as os
import numpy as np
import pickle
import itertools as it
from sympy import LeviCivita
from cymetric.config import real_dtype, complex_dtype



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


def _prepare_dataset_batched(point_gen, batch_n_p, val_split, ltails, rtails, normalize_to_vol_j):
    r"""Prepares a batch of training and validation data from point_gen.

    Returns:
        tuple: (points, weights, omega, X_train, y_train, X_val, y_val, val_pullbacks)
    """
    new_np = int(round(batch_n_p/(1-ltails-rtails)))
    pwo = point_gen.generate_point_weights(new_np, omega=True)
    if len(pwo) < new_np:
        new_np = int((new_np-len(pwo))/len(pwo)*new_np + 100)
        pwo2 = point_gen.generate_point_weights(new_np, omega=True)
        pwo = np.concatenate([pwo, pwo2], axis=0)
    new_np = len(pwo)
    sorted_weights = np.sort(pwo['weight'])
    lower_bound = sorted_weights[round(ltails*new_np)]
    upper_bound = sorted_weights[round((1-rtails)*new_np)-1]
    mask = np.logical_and(pwo['weight'] >= lower_bound, pwo['weight'] <= upper_bound)
    weights = np.expand_dims(pwo['weight'][mask], -1)
    omega = np.expand_dims(pwo['omega'][mask], -1)
    omega = np.real(omega * np.conj(omega))
    
    n_mask = len(weights)
    t_i = int((1-val_split)*n_mask)
    points = pwo['point'][mask]
    
    if normalize_to_vol_j:
        pbs = point_gen.pullbacks(points)
        fs_ref = point_gen.fubini_study_metrics(points, vol_js=np.ones_like(point_gen.kmoduli))
        fs_ref_pb = np.einsum('xai,xij,xbj->xab', pbs, fs_ref, np.conj(pbs))
        aux_weights = omega.flatten() / weights.flatten()
        norm_fac = point_gen.vol_j_norm / np.mean(np.real(np.linalg.det(fs_ref_pb)) / aux_weights)
        weights = norm_fac * weights

    X_train = np.concatenate((points[:t_i].real, points[:t_i].imag), axis=-1)
    y_train = np.concatenate((weights[:t_i], omega[:t_i]), axis=1)
    X_val = np.concatenate((points[t_i:].real, points[t_i:].imag), axis=-1)
    y_val = np.concatenate((weights[t_i:], omega[t_i:]), axis=1)
    val_pullbacks = point_gen.pullbacks(points[t_i:])
    return points, weights, omega, X_train, y_train, X_val, y_val, val_pullbacks


def prepare_dataset(point_gen, n_p, dirname, n_batches=None, val_split=0.1, ltails=0, rtails=0, normalize_to_vol_j=True):
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
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if n_batches is None and n_p > 300000:
        n_batches = n_p // 50000 if n_p // 50000 > 0 else 1
    elif n_batches is None:
        n_batches = 1
    if n_batches > 1:
        print(f'Generating {n_p} points using {n_batches} batches')
    base = n_p // n_batches
    rem = n_p % n_batches
    all_points, all_weights, all_omega = [], [], []
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list, val_pb_list = [], [], []
    for i in range(n_batches):
        print(f'Generating {base + (1 if i < rem else 0)} points using {i}th batch')
        batch_n = base + (1 if i < rem else 0)
        pts, w, om, X_tr, y_tr, X_v, y_v, val_pb = _prepare_dataset_batched(
            point_gen, batch_n, val_split, ltails, rtails, normalize_to_vol_j)
        all_points.append(pts)
        all_weights.append(w)
        all_omega.append(om)
        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        X_val_list.append(X_v)
        y_val_list.append(y_v)
        val_pb_list.append(val_pb)
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    val_pullbacks = np.concatenate(val_pb_list, axis=0)
    all_points = np.concatenate(all_points, axis=0)
    all_weights = np.concatenate(all_weights, axis=0)
    all_omega = np.concatenate(all_omega, axis=0)
    np.savez_compressed(os.path.join(dirname, 'dataset'),
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        val_pullbacks=val_pullbacks)
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
        os.makedirs(dirname)

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
