""" 
A collection of various helper functions.
"""
import tensorflow as tf
from cymetric.config import real_dtype, complex_dtype



def prepare_tf_basis(basis, dtype=complex_dtype):
    r"""Casts each entry in Basis to dtype.

    Args:
        basis (dict): dictionary containing geometric information
        dtype (_type_, optional): type to cast to. Defaults to complex_dtype.

    Returns:
        dict: with tensors rather than ndarrays
    """
    new_basis = {}
    for key in basis:
        new_basis[key] = tf.cast(basis[key], dtype=dtype)
    return new_basis


def train_model(fsmodel, data, optimizer=None, epochs=50, batch_sizes=[64, 10000],
                verbose=1, custom_metrics=[], callbacks=[], sw=False):
    r"""Training loop for fixing the Kähler class. It consists of two 
    optimisation steps. 
        1. With a small batch size and volk loss disabled.
        2. With only MA and volk loss enabled and a large batchsize such that 
            the MC integral is a reasonable approximation and we don't lose 
            the MA progress from the first step.

    Args:
        fsmodel (cymetric.models.tfmodels): Any of the custom metric models.
        data (dict): numpy dictionary with keys 'X_train' and 'y_train'.
        optimizer (tfk.optimiser, optional): Any tf optimizer. Defaults to None.
            If None Adam is used with default hyperparameters.
        epochs (int, optional): # of training epochs. Every training sample will
            be iterated over twice per Epoch. Defaults to 50.
        batch_sizes (list, optional): batch sizes. Defaults to [64, 10000].
        verbose (int, optional): If > 0 prints epochs. Defaults to 1.
        custom_metrics (list, optional): List of tf metrics. Defaults to [].
        callbacks (list, optional): List of tf callbacks. Defaults to [].
        sw (bool, optional): If True, use integration weights as sample weights.
            Defaults to False.

    Returns:
        model, training_history
    """
    training_history = {}
    hist1 = {}
    hist2 = {}
    # Store original learning flags
    learn_kaehler = fsmodel.learn_kaehler
    learn_transition = fsmodel.learn_transition
    learn_ricci = fsmodel.learn_ricci
    learn_ricci_val = fsmodel.learn_ricci_val
    
    # Set up sample weights if needed
    if sw:
        sample_weights = data['y_train'][:, -2]
    else:
        sample_weights = None
        
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
        
    # Compile once at start of training to avoid resetting optimizer
    fsmodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
    
    # Create datasets outside the loop to avoid recreating them each epoch
    dataset1 = tf.data.Dataset.from_tensor_slices(
        (tf.cast(data['X_train'], real_dtype), tf.cast(data['y_train'], real_dtype)))
    ).cache()
    
    dataset2 = dataset1.cache()  # Reuse the same dataset with different batch size
    
    for epoch in range(epochs):
        if verbose > 0:
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
        # Step 1: Small batch size, volk loss disabled
        batch_size = batch_sizes[0]
        fsmodel.learn_kaehler = learn_kaehler
        fsmodel.learn_transition = learn_transition
        fsmodel.learn_ricci = learn_ricci
        fsmodel.learn_ricci_val = learn_ricci_val
        fsmodel.learn_volk = tf.cast(False, dtype=tf.bool)
        
        steps_per_epoch = len(data['X_train']) // batch_size
        batched_dataset = dataset1.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        history = fsmodel.fit(
            batched_dataset.repeat(),
            epochs=1, 
            verbose=verbose,
            callbacks=None, 
            sample_weight=sample_weights,
            steps_per_epoch=steps_per_epoch
        )
        
        # Update history
        for k in history.history.keys():
            if k not in hist1:
                hist1[k] = history.history[k]
            else:
                hist1[k] += history.history[k]
        
        # Step 2: Large batch size, only MA and volk loss enabled
        batch_size = min(batch_sizes[1], len(data['X_train']))
        fsmodel.learn_kaehler = tf.cast(False, dtype=tf.bool)
        fsmodel.learn_transition = tf.cast(False, dtype=tf.bool)
        fsmodel.learn_ricci = tf.cast(False, dtype=tf.bool)
        fsmodel.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        fsmodel.learn_volk = tf.cast(True, dtype=tf.bool)
        
        steps_per_epoch = len(data['X_train']) // batch_size
        batched_dataset = dataset2.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        history = fsmodel.fit(
            batched_dataset.repeat(),
            epochs=1, 
            verbose=verbose,
            callbacks=callbacks, 
            sample_weight=sample_weights,
            steps_per_epoch=steps_per_epoch
        )
        
        # Update history
        for k in history.history.keys():
            if k not in hist2:
                hist2[k] = history.history[k]
            else:
                hist2[k] += history.history[k]
    
    # Combine histories
    for k in set(list(hist1.keys()) + list(hist2.keys())):
        training_history[k] = hist2[k] if (k not in hist1 or (k in hist2 and max(hist2[k]) != 0)) else hist1[k]
    
    training_history['epochs'] = list(range(epochs))
    return fsmodel, training_history
    
