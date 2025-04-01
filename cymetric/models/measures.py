""" 
Various error measures for neural nets 
representing (ricci flat) kaehler metrics.
"""
import tensorflow as tf
import sys
from cymetric.config import real_dtype, complex_dtype



def sigma_measure(model, points, y_true, pullbacks = None, batch = False):
    r"""We compute the Monge Ampere equation

    .. math::

        \sigma = 1 / (\text{Vol}_\text{cy} n_p) \sum_i |1 - (\det(g) \text{Vol}_\text{cy})/(|\Omega|^2 \text{Vol}_\text{K})|

    Args:
        model (tfk.model): Any (sub-)class of FSModel.
        points (tensor[(n_p,2*ncoord), real_dtype]): NN input
        y_true (tensor[(n_p,2), real_dtype]): (weights,  Omega \wedge \bar(\Omega)|_p)

    Returns:
        tf.float: sigma measure
    """
    if batch:
        # Process in batches using tf.while_loop
        batch_size = 1000
        num_points = tf.shape(points)[0]
        
        # Initialize accumulator variables
        i = tf.constant(0)
        sigma_sum = tf.constant(0.0, dtype=real_dtype)
        volume_cy_sum = tf.constant(0.0, dtype=real_dtype)
        
        # Define condition and body for while loop
        def condition(i, sigma_sum, volume_cy_sum):
            return i < num_points
        
        def body(i, sigma_sum, volume_cy_sum):
            end_idx = tf.minimum(i + batch_size, num_points)
            batch_points = points[i:end_idx]
            batch_y_true = y_true[i:end_idx]
            
            g = model(batch_points, pb = pullbacks)
            weights = batch_y_true[:, -2]
            omega = batch_y_true[:, -1]
            
            det = tf.math.real(tf.linalg.det(g))
            det_over_omega = det / omega
            batch_volume_cy = tf.math.reduce_mean(weights, axis=-1)
            vol_k = tf.math.reduce_mean(det_over_omega * weights, axis=-1)
            ratio = batch_volume_cy / vol_k
            sigma_integrand = tf.abs(tf.ones(tf.shape(det_over_omega), dtype=real_dtype) - det_over_omega * ratio) * weights
            
            # Accumulate weighted sum
            batch_sigma_sum = tf.reduce_sum(sigma_integrand)
            batch_vol_sum = tf.reduce_sum(weights)
            
            return i + batch_size, sigma_sum + batch_sigma_sum, volume_cy_sum + batch_vol_sum
        
        # Run the loop
        _, final_sigma_sum, final_volume_cy_sum = tf.while_loop(
            condition, body, [i, sigma_sum, volume_cy_sum])
        
        # Compute final result
        sigma = final_sigma_sum / final_volume_cy_sum
        return sigma
    else:
        g = model(points, pb = pullbacks)
        weights = y_true[:, -2]
        omega = y_true[:, -1]
        # use gamma series
        det = tf.math.real(tf.linalg.det(g))  # * factorial / (2**nfold)
        det_over_omega = det / omega
        volume_cy = tf.math.reduce_mean(weights, axis=-1)
        vol_k = tf.math.reduce_mean(det_over_omega * weights, axis=-1)
        ratio = volume_cy / vol_k
        sigma_integrand = tf.abs(tf.ones(tf.shape(det_over_omega), dtype=real_dtype) - det_over_omega * ratio) * weights
        sigma = tf.math.reduce_mean(sigma_integrand) / volume_cy
        return sigma


def ricci_measure(model, points, y_true, pullbacks=None, verbose=0):
    r"""Computes the Ricci measure for a kaehler metric.

    .. math::

        ||R|| \equiv \frac{\text{Vol}_K^{\frac{1}{\text{nfold}}}}{\text{Vol}_{\text{CY}}}
            \int_X d\text{Vol}_K |R|

    Args:
        model (tfk.model): Any (sub-)class of FSModel.
        points (tensor[(n_p,2*ncoord), real_dtype]): NN input
        y_true (tensor[(n_p,2), real_dtype]): (weights, 
            Omega \wedge \bar(\Omega)|_p)
        pullbacks (tensor[(n_p,nfold,ncoord), complex_dtype]): Pullback tensor
            Defaults to None. Then gets computed.
        verbose (int, optional): if > 0 prints some intermediate
            infos. Defaults to 0.

    Returns:
        real_dtype: Ricci measure
    """
    nfold = tf.cast(model.nfold, dtype=real_dtype)
    ncoords = model.ncoords
    weights = y_true[:, -2]
    omega = y_true[:, -1]
    if pullbacks is None:
        pullbacks = model.pullbacks(points)
    # factorial = tf.exp(tf.math.lgamma(nfold+1))
    x_vars = points
    # take derivatives
    with tf.GradientTape(persistent=False) as tape1:
        tape1.watch(x_vars)
        with tf.GradientTape(persistent=False) as tape2:
            tape2.watch(x_vars)
            prediction = model(x_vars)
            det = tf.math.real(tf.linalg.det(prediction)) * 1.  # factorial / (2**nfold)
            log = tf.math.log(det)
        di_dg = tape2.gradient(log, x_vars)
    didj_dg = tf.cast(tape1.batch_jacobian(di_dg, x_vars), dtype=complex_dtype)
    # add derivatives together to complex tensor
    ricci_ij = didj_dg[:, 0:ncoords, 0:ncoords]
    ricci_ij += 1j * didj_dg[:, 0:ncoords, ncoords:]
    ricci_ij -= 1j * didj_dg[:, ncoords:, 0:ncoords]
    ricci_ij += didj_dg[:, ncoords:, ncoords:]
    ricci_ij *= 0.25
    pred_inv = tf.linalg.inv(prediction)
    ricci_scalar = tf.einsum('xba,xai,xij,xbj->x', pred_inv, pullbacks,
                             ricci_ij, tf.math.conj(pullbacks))
    ricci_scalar = tf.math.abs(tf.math.real(ricci_scalar))
    if verbose > 0:
        tf.print(' - Avg ricci scalar is',
                 tf.math.reduce_mean(ricci_scalar), output_stream=sys.stdout)
        if verbose > 1:
            tf.print(' - Max ricci scalar is',
                     tf.reduce_max(ricci_scalar), output_stream=sys.stdout)
            tf.print(' - Min ricci scalar is',
                     tf.reduce_min(ricci_scalar), output_stream=sys.stdout)

    # compute ricci measure
    det_over_omega = det / omega
    volume_cy = tf.math.reduce_mean(weights, axis=-1)
    vol_k = tf.math.reduce_mean(det_over_omega * weights, axis=-1)
    ricci_measure = (vol_k ** (1 / nfold) / volume_cy) * tf.math.reduce_mean(det_over_omega * ricci_scalar * weights, axis=-1)
    return ricci_measure


def ricci_scalar_fn(model, points, pullbacks=None, verbose=0, rdet=True):
    r"""Computes the Ricci scalar for a kaehler metric.

    .. math::
        R = g^{ij} \partial_i \bar{\partial}_j \log \det g

    Args:
        model (tfk.model): Any (sub-)class of FSModel.
        points (tensor[(n_p,2*ncoord), real_dtype]): NN input
        pullbacks (tensor[(n_p,nfold,ncoord), complex_dtype]): Pullback tensor. Defaults to None. Then gets computed.
        verbose (int, optional): if > 0 prints some intermediate infos. Defaults to 0.
        rdet (bool, optional): if True also returns det. Defaults to True.
            This is a bit hacky, because the output signature changes
            but avoids recomputing the determinant after batching.

    Returns:
        real_dtype(tensor[(n_p,), real_dtype]): Ricci scalar
    """
    ncoords = model.ncoords
    x_vars = points
    if pullbacks is None:
        pullbacks = model.pullbacks(points)
    # take derivatives
    with tf.GradientTape(persistent=False) as tape1:
        tape1.watch(x_vars)
        with tf.GradientTape(persistent=False) as tape2:
            tape2.watch(x_vars)
            prediction = model(x_vars)
            det = tf.math.real(tf.linalg.det(prediction)) * 1.  # factorial / (2**nfold)
            log = tf.math.log(det)
        di_dg = tape2.gradient(log, x_vars)
    didj_dg = tf.cast(tape1.batch_jacobian(di_dg, x_vars), dtype=complex_dtype)
    # add derivatives together to complex tensor
    ricci_ij = didj_dg[:, 0:ncoords, 0:ncoords]
    ricci_ij += 1j * didj_dg[:, 0:ncoords, ncoords:]
    ricci_ij -= 1j * didj_dg[:, ncoords:, 0:ncoords]
    ricci_ij += didj_dg[:, ncoords:, ncoords:]
    ricci_ij *= 0.25
    pred_inv = tf.linalg.inv(prediction)
    ricci_scalar = tf.einsum('xba,xai,xij,xbj->x', pred_inv, pullbacks,
                             ricci_ij, tf.math.conj(pullbacks))
    ricci_scalar = tf.math.real(ricci_scalar)
    if verbose > 0:
        tf.print(' - Avg ricci scalar is',
                 tf.math.reduce_mean(ricci_scalar), output_stream=sys.stdout)
        if verbose > 1:
            tf.print(' - Max ricci scalar is',
                     tf.reduce_max(ricci_scalar), output_stream=sys.stdout)
            tf.print(' - Min ricci scalar is',
                     tf.reduce_min(ricci_scalar), output_stream=sys.stdout)
    if rdet:
        return ricci_scalar, det
    else:
        return ricci_scalar


def sigma_measure_loss(model, points, omegas):
    r"""

    Args:
        model (tfk.model): Any (sub-)class of FSModel.
        points (tensor[(n_p,2*ncoord), real_dtype]): NN input
        omegas (tensor[(n_p), real_dtype]): \|Omega\|^2 for the points provided

    Returns:
        tf.float: sigma measure
    """
    return tf.math.reduce_mean(model.sigma_loss(omegas, model(points)))


def kaehler_measure_loss(model, points):
    r"""Computes the Kahler loss measure.

    Args:
        model (tfk.model): Any (sub-)class of FSModel.
        points (tensor[(n_p,2*ncoord), real_dtype]): NN input

    Returns:
        real_dtype: Kahler loss measure
    """
    return tf.math.reduce_mean(model.compute_kaehler_loss(points))


def transition_measure_loss(model, points):
    r"""Computes the Transition loss measure.

    Args:
        model (tfk.model): Any (sub-)class of FSModel.
        points (tensor[(n_p,2*ncoord), real_dtype]): NN input

    Returns:
        real_dtype: Transition loss measure
    """
    return tf.math.reduce_mean(
        model.compute_transition_loss(tf.cast(points, dtype=real_dtype)))
