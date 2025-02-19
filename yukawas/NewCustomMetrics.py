from cymetric.config import float_dtype, complex_dtype
import tensorflow.keras as tfk
import tensorflow as tf


class LaplacianLoss(tfk.metrics.Metric):
    def __init__(self, name='laplacian_loss', **kwargs):
        super(LaplacianLoss, self).__init__(name=name, **kwargs)
        self.laplacian_loss = self.add_weight(name='lpl', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: tupe (data['X_val'], data['y_val'])
            sample_weight: sample weights for the validation set (Default: None)

        Returns:

        """
        loss = values['laplacian_loss']
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            loss = tf.multiply(loss, sample_weight)
        new_value = (tf.reduce_mean(loss, axis=-1) - self.laplacian_loss)/(self.count+1)
        self.laplacian_loss.assign_add(new_value)
        #tf.print("hiasg")
        #tf.print(tf.shape(self.laplacian_loss))
        self.count.assign_add(1)

    def result(self):
        return self.laplacian_loss

    def reset_state(self):
        self.laplacian_loss.assign(0)
        self.count.assign(0)



def laplacian_measure_loss(model, validation_data):
    r"""Computes the Transition loss measure.

    Args:
        model (tfk.model): Any (sub-)class of FSModel.
        points (tensor[(n_p,2*ncoord), float_dtype]): NN input

    Returns:
        float_dtype: Transition loss measure
    """
    #X_val, aux = validation_data
    X_val = validation_data["X_val"]
    pullbacks = validation_data["val_pullbacks"]
    invmetrics = validation_data["inv_mets_val"]
    sources = validation_data["sources_val"]

    #X_val = tf.cast(X_val, float_dtype)
    #y_val = tf.cast(y_val, float_dtype)
    #aux=tf.cast(aux, float_dtype)
    #sort out this float32 problem!
    return tf.math.reduce_mean(
        model.compute_laplacian_loss(X_val,pullbacks,invmetrics,sources))

laplacian_measure_tf = tf.function(func=laplacian_measure_loss)
class LaplacianCallback(tfk.callbacks.Callback):
    """Callback that tracks the transition loss weighted over the CY."""
    def __init__(self, validation_data, initial=False):
        r"""A callback which computes the transition measure for
        the validation data after every epoch end.

        Args:
            validation_data (tuple(X_val, y_val)): validation data
            initial (bool, optional): If True does one iteration before training.
                Defaults to False.
        """
        super(LaplacianCallback, self).__init__()
        self.data=validation_data
        self.initial=initial
        
    def on_epoch_end(self, epoch, logs=None):
        r"""Computes transition measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): history.history. Defaults to None.
        """
        laplacian = laplacian_measure_tf(self.model, self.data)

        cb_res = laplacian.numpy().tolist()
        logs['laplacian_val'] = cb_res
        if cb_res <= 1e-3:
            print(' - Laplacian measure val: {:.4e}'.format(cb_res))
        else:
            print(' - Laplacian measure val: {:.4f}'.format(cb_res))
    
    def on_train_begin(self, logs=None):
        r"""Compute transition measure before training as baseline.

        Args:
            logs (dict, optional): History. Defaults to None.
        """
        if self.initial:
            self.on_epoch_end(-1, logs=logs)

class NaNCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if tf.math.is_nan(logs.get('loss')) and epoch > 1:  # epoch >= 1 means after epoch 2 (since epoch counting starts at 0)
            print(f"Breaking due to NaN loss after epoch {epoch}")
            self.model.stop_training = True

