from cymetric.models.fubinistudy import FSModel
from laplacian_funcs import *
import tensorflow as tf
import os
import numpy as np
#from pympler import tracker
from cymetric.config import real_dtype, complex_dtype

def point_vec_to_complex(p):
    #if len(p) == 0: 
    #    return tf.constant([[]])
    plen = len(p[0])//2
    return tf.complex(p[:, :plen],p[:, plen:])

class AlphaPrimeModel(FSModel):
    r"""FreeModel from which all other models inherit.

    The training and validation steps are implemented in this class. All
    other computational routines are inherited from:
    cymetric.models.fubinistudy.FSModel
    
    Example:
        Assume that `BASIS` and `data` have been generated with a point 
        generator.

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from cymetric.models.tfmodels import FreeModel
        >>> from cymetric.models.tfhelper import prepare_tf_basis
        >>> tfk = tf.keras
        >>> data = np.load('dataset.npz')
        >>> BASIS = prepare_tf_basis(np.load('basis.pickle', allow_pickle=True))
    
        set up the nn and FreeModel

        >>> nfold = 3
        >>> ncoords = data['X_train'].shape[1]
        >>> nn = tfk.Sequential(
        ...     [   
        ...         tfk.layers.Input(shape=(ncoords)),
        ...         tfk.layers.Dense(64, activation="gelu"),
        ...         tfk.layers.Dense(nfold**2),
        ...     ]
        ... )
        >>> model = FreeModel(nn, BASIS)

        next we can compile and train

        >>> from cymetric.models.metrics import TotalLoss
        >>> metrics = [TotalLoss()]
        >>> opt = tfk.optimizers.Adam()
        >>> model.compile(custom_metrics = metrics, optimizer = opt)
        >>> model.fit(data['X_train'], data['y_train'], epochs=1)

        For other custom metrics and callbacks to be tracked, check
        :py:mod:`cymetric.models.metrics` and
        :py:mod:`cymetric.models.callbacks`.
    """
    def __init__(self, tfmodel, BASIS,phimodel, alphaprime,euler_char,alpha=None, **kwargs):
        r"""FreeModel is a tensorflow model predicting CY metrics. 
        
        The output is
            
            .. math:: g_{\text{out}} = g_{\text{NN}}
        
        a hermitian (nfold, nfold) tensor with each float directly predicted
        from the neural network.

        NOTE:
            * The model by default does not train against the ricci loss.
                
                To enable ricci training, set `self.learn_ricci = True`,
                **before** the tracing process. For validation data 
                `self.learn_ricci_val = True`,
                can be modified separately.

            * The models loss contributions are

                1. sigma_loss
                2. kaehler loss
                3. transition loss
                4. ricci loss (disabled)
                5. volk loss

            * The different losses are weighted with alpha.

            * The (FB-) norms for each loss are specified with the keyword-arg

                >>> model = FreeModel(nn, BASIS, norm = [1. for _ in range(5)])

            * Set kappa to the kappa value of your training data.

                >>> kappa = np.mean(data['y_train'][:,-2])

        Args:
            tfmodel (tfk.model): the underlying neural network.
            BASIS (dict): a dictionary containing all monomials and other
                relevant information from cymetric.pointgen.pointgen.
            alpha ([5//NLOSS], float): Weighting of each loss contribution.
                Defaults to None, which corresponds to equal weights.
        """
        super(AlphaPrimeModel, self).__init__(BASIS=BASIS, **kwargs)
        self.model = tfmodel
        self.NLOSS = 2
        # variable or constant or just tensor?
        if alpha is not None:
            self.alpha = [tf.Variable(a, dtype=tf.float32) for a in alpha]
        else:
            self.alpha = [tf.Variable(1., dtype=tf.float32) for _ in range(self.NLOSS)]
        self.learn_transition = tf.cast(True, dtype=tf.bool)
        self.learn_laplacian = tf.cast(True, dtype=tf.bool)

        #self.learn_kaehler = tf.cast(False, dtype=tf.bool)
        #self.learn_ricci = tf.cast(False, dtype=tf.bool)
        #self.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        #self.learn_volk = tf.cast(False, dtype=tf.bool)

        self.custom_metrics = None
        #self.kappa = tf.cast(BASIS['KAPPA'], dtype=tf.float32)
        self.gclipping = float(5.0)
        # add to compile?
        #self.sigma_loss = sigma_loss(self.kappa, tf.cast(self.nfold, dtype=tf.float32))
        self.phimodel =phimodel
        self.alphaprime=alphaprime
        self.euler_char=euler_char    


    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point. In the case of the Phi model, we demand that \phi(\lambda^q_i z_i)=\phi(z_i)

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize], tf.float32): Transition loss at each point.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        patch_indices = tf.where(~inv_one_mask)[:, 1]
        patch_indices = tf.reshape(patch_indices, (-1, self.nProjective))
        current_patch_mask = self._indices_to_mask(patch_indices)
        fixed = self._find_max_dQ_coords(points)
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        if self.nhyper == 1:
            other_patches = tf.gather(self.fixed_patches, fixed)
        else:
            combined = tf.concat((fixed, patch_indices), axis=-1)
            other_patches = self._generate_patches_vec(combined)
        
        other_patches = tf.reshape(other_patches, (-1, self.nProjective))
        other_patch_mask = self._indices_to_mask(other_patches)
        # NOTE: This will include same to same patch transitions
        exp_points = tf.repeat(cpoints, self.nTransitions, axis=-2)#expanded points
        patch_points = self._get_patch_coordinates(exp_points, tf.cast(other_patch_mask, dtype=tf.bool)) # other patches
        real_patch_points = tf.concat((tf.math.real(patch_points), tf.math.imag(patch_points)), axis=-1)
        gj = self.model(real_patch_points, training=True)
        gi = tf.repeat(self.model(points), self.nTransitions, axis=0)
        all_t_loss = tf.math.abs(gi-gj)
        all_t_loss = tf.reshape(all_t_loss, (-1, self.nTransitions))
        all_t_loss = tf.math.reduce_sum(all_t_loss**self.n[1], axis=-1)
        return all_t_loss/(self.nTransitions)


    def compute_laplacian_loss(self,x,pullbacks,invmetrics,sources):
        r"""Computes transition loss at each point. In the case of the Phi model, we demand that \phi(\lambda^q_i z_i)=\phi(z_i)

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize], tf.float32): Transition loss at each point.
        """
        #cast to real
        lpl_losses=tf.math.abs(tf.math.real(laplacian(self.model,x,pullbacks,invmetrics))-(sources))
        all_lpl_loss = lpl_losses**self.n[0]
        return all_lpl_loss


    def call(self, input_tensor, j_elim=None):
        r"""Prediction of the model.

        .. math::

            g_{\text{out}; ij} = g_{\text{FS}; ij} + \
                partial_i \bar{\partial}_j \phi_{\text{NN}}

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional):
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64):
                Prediction at each point.
        """
        # nn prediction
        with tf.GradientTape(persistent=False) as tape1:
            tape1.watch(input_tensor)
            with tf.GradientTape(persistent=False) as tape2:
                tape2.watch(input_tensor)
                # Need to disable training here, because batch norm
                # and dropout mix the batches, such that batch_jacobian
                # is no longer reliable.
                phi = self.model(input_tensor, training=False)
            d_phi = tape2.gradient(phi, input_tensor)
        dd_phi = tape1.batch_jacobian(d_phi, input_tensor)
        dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = \
            0.25*dd_phi[:, :self.ncoords, :self.ncoords], \
            0.25*dd_phi[:, :self.ncoords, self.ncoords:], \
            0.25*dd_phi[:, self.ncoords:, :self.ncoords], \
            0.25*dd_phi[:, self.ncoords:, self.ncoords:]
        dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi = tf.einsum('xai,xij,xbj->xab', pbs, dd_phi, tf.math.conj(pbs))
        zeta_of_3=1.2020569031595942853997381
        dd_shift_to_KP = ((2*np.pi*self.alphaprime)**3)/4* zeta_of_3*dd_phi

        # fs metric
        
        original_adjusted_met = self.phimodel(input_tensor, j_elim=j_elim)
        # return g_fs + \del\bar\del\phi
        return tf.math.add(original_adjusted_met, dd_shift_to_KP)

    def call_only_shift(self, input_tensor, j_elim=None):
        r"""Prediction of the model.

        .. math::

            g_{\text{out}; ij} = g_{\text{FS}; ij} + \
                partial_i \bar{\partial}_j \phi_{\text{NN}}

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional):
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64):
                Prediction at each point.
        """
        # nn prediction
        with tf.GradientTape(persistent=False) as tape1:
            tape1.watch(input_tensor)
            with tf.GradientTape(persistent=False) as tape2:
                tape2.watch(input_tensor)
                # Need to disable training here, because batch norm
                # and dropout mix the batches, such that batch_jacobian
                # is no longer reliable.
                phi = self.model(input_tensor, training=False)
            d_phi = tape2.gradient(phi, input_tensor)
        dd_phi = tape1.batch_jacobian(d_phi, input_tensor)
        dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = \
            0.25*dd_phi[:, :self.ncoords, :self.ncoords], \
            0.25*dd_phi[:, :self.ncoords, self.ncoords:], \
            0.25*dd_phi[:, self.ncoords:, :self.ncoords], \
            0.25*dd_phi[:, self.ncoords:, self.ncoords:]
        dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi = tf.einsum('xai,xij,xbj->xab', pbs, dd_phi, tf.math.conj(pbs))
        zeta_of_3=1.2020569031595942853997381
        dd_shift_to_KP = ((2*np.pi*self.alphaprime)**3)/4* zeta_of_3*dd_phi

        # fs metric
        
        #original_adjusted_met = self.phimodel(input_tensor, j_elim=j_elim)
        # return g_fs + \del\bar\del\phi
        return dd_shift_to_KP#tf.math.add(original_adjusted_met, dd_shift_to_KP)
        

    def compile(self, custom_metrics=None, **kwargs):
        r"""Compiles the model.
        kwargs takes any argument of regular `tf.model.compile()`
        Example:
            >>> model = FreeModel(nn, BASIS)
            >>> from cymetric.models.metrics import TotalLoss
            >>> metrics = [TotalLoss()]
            >>> opt = tfk.optimizers.Adam()
            >>> model.compile(metrics=metrics, optimizer=opt)
        Args:
            custom_metrics (list, optional): List of custom metrics.
                See also :py:mod:`cymetric.models.metrics`. If None, no metrics
                are tracked during training. Defaults to None.
        """
        if custom_metrics is not None:
            kwargs['metrics'] = custom_metrics
        super(AlphaPrimeModel, self).compile(**kwargs)

    @property
    def metrics(self):
        r"""Returns the model's metrics, including custom metrics.
        Returns:
            list: metrics
        """
        return self._metrics

    def train_step(self, data):
        r"""Train step of a single batch in model.fit().

        NOTE:
            1. The first epoch will take additional time, due to tracing.
            
            2. Warnings are plentiful. Disable on your own risk with 

                >>> tf.get_logger().setLevel('ERROR')
            
            3. The conditionals need to be set before tracing. 
            
            4. We employ under the hood gradient clipping.

        Args:
            data (tuple): test_data (x,y, sample_weight)

        Returns:
            dict: metrics
        """
        # if len(data) == 3:
        #     x, sample_weight = data
        # else:
        #     sample_weight = None
        #     x, aux = dataX_train, y_train, train_pullback,inv_mets_train,sources_train
        x = data["X_train"]
        y = None
        y_pred=None
        # print("hi")
        # print(x.shape)
        # print(len(x))
        # The 'y_train/val' arrays contain the integration weights and $\\Omega \\wedge \\bar\\Omega$ for each point. In principle, they can be used for any relevant pointwise information that could be needed during the training process."

        sample_weight = None
        pbs = data["train_pullbacks"]
        invmets = data["inv_mets_train"]
        sources = data["sources_train"]
        #x,sample_weight, pbs, invmets, sources = data#.values()
        # print("help")
        # print(type(data))
        # print(type(data.values()))
        # print(data)
        # print("hi")
        # print(list(x))
        with tf.GradientTape(persistent=False) as tape:
            trainable_vars = self.model.trainable_variables
            #tape.watch(trainable_vars)
            #automatically watch trainable vars
            # add other loss contributions.
            if self.learn_transition:
                t_loss = self.compute_transition_loss(x)
            else:
                t_loss = tf.zeros_like(x[:, 0])
            if self.learn_laplacian:
                lpl_loss = self.compute_laplacian_loss(x,pbs,invmets,sources)
                #print("lpl beta")
            else:
                lpl_loss = tf.zeros_like(x[:, 0])

            #omega = tf.expand_dims(y[:, -1], -1)
            #sigma_loss_cont = self.sigma_loss(omega, y_pred)**self.n[0]
            total_loss = self.alpha[0]*lpl_loss +\
                self.alpha[1]*t_loss 
            # weight the loss.
            if sample_weight is not None:
                total_loss *= sample_weight
            total_loss_mean=tf.reduce_mean(total_loss)
        # Compute gradients
        gradients = tape.gradient(total_loss_mean, trainable_vars)
        # remove nans and gradient clipping from transition loss.
        gradients = [tf.where(tf.math.is_nan(g), 1e-8, g) for g in gradients]
        gradients, _ = tf.clip_by_global_norm(gradients, self.gclipping)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return metrics. NOTE: This interacts badly with any regular MSE
        # compiled loss. Make it so that only custom metrics are updated?
        loss_dict = {m.name: m.result() for m in self.metrics}
        loss_dict['loss'] = tf.reduce_mean(total_loss)
        loss_dict['laplacian_loss'] = tf.reduce_mean(lpl_loss)
        loss_dict['transition_loss'] = tf.reduce_mean(t_loss)
        return loss_dict

    def test_step(self, data):
        r"""Same as train_step without the outer gradient tape.
        Does *not* update the NN weights.

        NOTE:
            1. Computes the exaxt same losses as train_step
            
            2. Ricci loss val can be separately enabled with
                
                >>> model.learn_ricci_val = True
            
            3. Requires additional tracing.

        Args:
            data (tuple): test_data (x,y, sample_weight)

        Returns:
            dict: metrics
        """
        # unpack data
        # if len(data) == 3:
        #     x, aux, sample_weight = data
        # else:
        #     sample_weight = None
        #     x, aux = data
        #x,sample_weight, pbs, invmets, sources = data.values()
        y = None
        y_pred=None
        x = data["X_val"]
        sample_weight = None
        pbs = data["val_pullbacks"]
        invmets = data["inv_mets_val"]
        sources = data["sources_val"]
        #print("validation happening")
        #y_pred = self(x)
        # add loss contributions
        if self.learn_transition:
            t_loss = self.compute_transition_loss(x)
        else:
            t_loss = tf.zeros_like(x[:, 0])
        if self.learn_laplacian:
            lpl_loss = self.compute_laplacian_loss(x,pbs,invmets,sources)
        else:
            lpl_loss = tf.zeros_like(x[:, 0])

        #omega = tf.expand_dims(y[:, -1], -1)
        #sigma_loss_cont = self.sigma_loss(omega, y_pred)**self.n[0]
        total_loss = self.alpha[0]*lpl_loss +\
            self.alpha[1]*t_loss 
        # weight the loss.
        if sample_weight is not None:
            total_loss *= sample_weight
        loss_dict = {m.name: m.result() for m in self.metrics}
        loss_dict['loss'] = tf.reduce_mean(total_loss)
        loss_dict['laplacian_loss'] = tf.reduce_mean(lpl_loss)
        loss_dict['transition_loss'] = tf.reduce_mean(t_loss)
        return loss_dict


    def save(self, filepath, **kwargs):
        r"""Saves the underlying neural network to filepath.

        NOTE: 
            Currently does not save the whole custom model.

        Args:
            filepath (str): filepath
        """
        # TODO: save graph? What about Optimizer?
        # https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        self.model.save(filepath=filepath, **kwargs)


def prepare_dataset_Alpha(point_gen, data, dirname, metricModel,euler_char,BASIS,val_split=0.1, ltails=0, rtails=0, normalize_to_vol_j=True,batch_size=100):
    r"""Prepares training and validation data from point_gen.

    Note:
        The dataset will be saved in `dirname/dataset.npz`.

    Args:
        point_gen (PointGenerator): Any point generator.
        n_p (int): # of points.
        dirname (str): dir name to save data.
        val_split (float, optional): train-val split. Defaults to 0.1.
        ltails (float, optional): Discarded % on the left tail of weight 
            distribution.
        rtails (float, optional): Discarded % on the left tail of weight 
            distribution.
        normalize_to_vol_j (bool, optional): Normalize such that

            .. math::
            
                \int_X \det(g) = \sum_p \det(g) * w|_p  = d^{ijk} t_i t_j t_k

            Defaults to True.

    Returns:
        np.float: kappa = vol_k / vol_cy
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    
    X_train=tf.cast(data['X_train'],tf.float32)
    y_train=tf.cast(data['y_train'],tf.float32)
    X_val=tf.cast(data['X_val'],tf.float32)
    y_val=tf.cast(data['y_val'],tf.float32)
    ncoords=int(len(X_train[0])/2)

    #y_train=data['y_train']
    #y_val=data['y_val']
    ys=tf.concat((y_train,y_val),axis=0)
    weights=tf.cast(tf.expand_dims(ys[:,0],axis=-1),tf.float32)
    omega=tf.cast(tf.expand_dims(ys[:,1],axis=-1),tf.float32)
    
    realpoints=tf.concat((X_train,X_val),axis=0)
    points=tf.complex(realpoints[:,0:ncoords],realpoints[:,ncoords:])
    new_np = len(realpoints)
    t_i = int((1-val_split)*new_np)


    mets = metricModel(realpoints)
    absdets = tf.abs(tf.linalg.det(mets))
    inv_mets=tf.linalg.inv(mets)
    inv_mets_train=inv_mets[:t_i]
    inv_mets_val=inv_mets[t_i:]

    flat_weights=weights[:,0]*omega[:,0]**(-1)*1/6
    cy_weights=flat_weights*absdets
    #print(tf.shape(weights))
    #print(tf.shape(flat_weights))
    ##print(tf.shape(absdets))
    #print(tf.shape(cy_weights))



    #still need to generate pullbacks apparently
    pullbacks = point_gen.pullbacks(points)
    train_pullbacks=tf.cast(pullbacks[:t_i],tf.complex64) 
    val_pullbacks=tf.cast(pullbacks[t_i:],tf.complex64) 

    # points = pwo['point'][mask]
    det = tf.math.real(absdets)  # * factorial / (2**nfold)
    #print("hi")
    det_over_omega = det / omega[:,0]
    #print("hi")
    volume_cy = tf.math.reduce_mean(weights[:,0], axis=-1)# according to raw CY omega calculation and sampling...
    #print("hi")
    vol_k_no6 = tf.math.reduce_mean(det_over_omega * weights[:,0], axis=-1)#missing factor of 6
    #print("hi")
    kappaover6 = tf.cast(vol_k_no6,tf.float32) / tf.cast(volume_cy,tf.float32)
    #rint(ratio)
    #print("hi")
    tf.cast(kappaover6,tf.float32)
    #print("hi")
    det = tf.cast(det,tf.float32)
    print('kappa over 6 ')
    print(kappaover6) 
 
    source_computing_class= Q_compiled_function(metricModel,realpoints[0:batch_size],batch_size)    
    Q_values,euler_all_with_sqrtg = compute_batched_func(source_computing_class.compute_Q,realpoints, batch_size,cy_weights)
    #sources = euler_char/volume - Qs
    sources = euler_char/(vol_k_no6/6)-Q_values
    sources_train=sources[:t_i]
    sources_val=sources[t_i:]

    print("Euler_characteristic with " + str(len(euler_all_with_sqrtg)) + "points: " + str(tf.reduce_mean(euler_all_with_sqrtg)))
    print("integral of sources: " + str(tf.reduce_mean(euler_char-euler_all_with_sqrtg)))

    
    np.savez_compressed(os.path.join(dirname, 'dataset'),
                        X_train=X_train,
                        y_train=y_train,
                        train_pullbacks=train_pullbacks,
                        inv_mets_train=inv_mets_train,
                        sources_train=sources_train,
                        X_val=X_val,
                        y_val=y_val,
                        val_pullbacks=val_pullbacks,
                        inv_mets_val=inv_mets_val,
                        sources_val=sources_val
                        )
    print("print 'kappa/6'")
    return kappaover6#point_gen.compute_kappa(points, weights, omega)


def train_modelalpha(alphaprimemodel, data_train, optimizer=None, epochs=50, batch_sizes=[64, 10000],
                verbose=1, custom_metrics=[], callbacks=[], sw=False):
    r"""Training loop for fixing the Kähler class. It consists of two 
    optimisation steps. 
        1. With a small batch size and volk loss disabled.
        2. With only MA and volk loss enabled and a large batchsize such that 
            the MC integral is a reasonable approximation and we don't lose 
            the MA progress from the first step.

    Args:
        alphaprimemodel (cymetric.models.tfmodels): Any of the custom metric models.
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
    # hist1['opt'] = ['opt1' for _ in range(epochs)]
    hist2 = {}
    # hist2['opt'] = ['opt2' for _ in range(epochs)]
    learn_laplacian = alphaprimemodel.learn_laplacian
    learn_transition = alphaprimemodel.learn_transition
    if sw:
        sample_weights = data_train['y_train'][:, -2]
    else:
        sample_weights = None
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    permint= tracker.SummaryTracker()
    for epoch in range(epochs):
        #print("internal")
        #print(permint.print_diff())
        batch_size = batch_sizes[0]
        alphaprimemodel.learn_transition = learn_transition
        alphaprimemodel.learn_laplacian = learn_laplacian
        alphaprimemodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
        if verbose > 0:
            print("\nEpoch {:2d}/{:d}".format(epoch + 1, epochs))
        history = alphaprimemodel.fit(
            data_train,
            epochs=1, batch_size=batch_size, verbose=verbose,
            callbacks=callbacks, sample_weight=sample_weights
        )
        #print(history)
        for k in history.history.keys():
            if k not in hist1.keys():
                hist1[k] = history.history[k]
            else:
                hist1[k] += history.history[k]

        #add learning rate schedule
        # if loss is not decreasing, reduce learning rate

        if epoch>5 and hist1['laplacian_loss'][-1]>hist1['laplacian_loss'][-2]:
            alphaprimemodel.optimizer.lr = alphaprimemodel.optimizer.lr*0.9
            print("cutting LR, multiplying by 0.9 - new LR: " + str(alphaprimemodel.optimizer.lr))

        #after 30 epochs, decrease learning rate by factor of 10
        if epoch==10:
            alphaprimemodel.optimizer.lr = alphaprimemodel.optimizer.lr*0.1
            print("cutting LR, multiplying by 0.1 - new LR: " + str(alphaprimemodel.optimizer.lr))

        # after 60 epochs, decrease lr by a factor of 10
        if epoch==20:
            alphaprimemodel.optimizer.lr = alphaprimemodel.optimizer.lr*0.1
            print("cutting LR, multiplying by 0.1 - new LR: " + str(alphaprimemodel.optimizer.lr))

        #after 90 epochs, decrease lr by a factor of 10
        if epoch==30:
            alphaprimemodel.optimizer.lr = alphaprimemodel.optimizer.lr*0.1
            print("cutting LR, multiplying by 0.1 - new LR: " + str(alphaprimemodel.optimizer.lr))

        if tf.math.is_nan(hist1['loss'][-1]):
            break

        #print("internal2")
        #print(permint.print_diff())
        # if history.history['transition_loss'][-1]<10**(-8):
        #     print("t_loss too low")
        #     break
        # batch_size = min(batch_sizes[1], len(data['X_train']))
        # alphaprimemodel.learn_kaehler = tf.cast(False, dtype=tf.bool)
        # alphaprimemodel.learn_transition = tf.cast(False, dtype=tf.bool)
        # alphaprimemodel.learn_ricci = tf.cast(False, dtype=tf.bool)
        # alphaprimemodel.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        # alphaprimemodel.learn_volk = tf.cast(True, dtype=tf.bool)
        # alphaprimemodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
        # history = alphaprimemodel.fit(
        #     data['X_train'], data['y_train'],
        #     epochs=1, batch_size=batch_size, verbose=verbose,
        #     callbacks=callbacks, sample_weight=sample_weights
        # )
        # for k in history.history.keys():
        #     if k not in hist2.keys():
        #         hist2[k] = history.history[k]
        #     else:
        #         hist2[k] += history.history[k]
    # training_history['epochs'] = list(range(epochs)) + list(range(epochs))
    # for k in hist1.keys():
    #     training_history[k] = hist1[k] + hist2[k]
    #for k in set(list(hist1.keys()) + list(hist2.keys())):
    for k in set(list(hist1.keys())):
        #training_history[k] = hist2[k] if k in hist2 and max(hist2[k]) != 0 else hist1[k]
        training_history[k] = hist1[k]
    training_history['epochs'] = list(range(epochs))
    return alphaprimemodel, training_history


class Q_compiled_function(tf.Module):
    def __init__(self,phimodel,ptsrealtoinit,batch_size):
        self.phimodel=phimodel
        self.batch_size=batch_size
        print("compiling")
        self.compute_christoffel_symbols_holo_not_pb = tf.function( 
            self.compute_christoffel_symbols_holo_not_pb_uncomp,
            input_signature=(tf.TensorSpec(shape=[batch_size, self.phimodel.ncoords*2], dtype=tf.float32),)
        )
        self.compute_riemann_m_nb_rb_sbUP = tf.function( 
            self.compute_riemann_m_nb_rb_sbUP_uncomp,
            input_signature=(tf.TensorSpec(shape=[batch_size, self.phimodel.ncoords*2], dtype=tf.float32),)
        )
        self.compute_Q = tf.function( 
            self.compute_Q_uncomp,
            input_signature=(tf.TensorSpec(shape=[batch_size, self.phimodel.ncoords*2], dtype=tf.float32),)
        )
        print("compiled")
        #Now compile the various bits
        self.compute_christoffel_symbols_holo_not_pb(ptsrealtoinit[0:batch_size])
        self.compute_riemann_m_nb_rb_sbUP(ptsrealtoinit[0:batch_size])
        self.compute_Q(ptsrealtoinit[0:batch_size])

    def compute_christoffel_symbols_holo_not_pb_uncomp(self,x):
        x_vars = x
        print('starting christoffel tape')
        with tf.GradientTape(persistent=True) as tapeC:
            tapeC.watch(x_vars)
            g=self.phimodel(x_vars)
            Rg=tf.math.real(g)
            Ig=tf.math.imag(g)
        #with tapeC.stop_recording():
        print('christoffel tape1')
        dXreal_dRg= tf.cast(tapeC.batch_jacobian(Rg, x_vars),dtype=tf.complex64)
        print('christoffel tape2')
        dXreal_dIg = tf.cast(tapeC.batch_jacobian(Ig, x_vars),dtype=tf.complex64)
        del tapeC
        print('del christoffel tape')
        inverseg=tf.linalg.inv(g)#this has indices inverse of a bbar = bbar a

        # add derivatives together to complex tensor
        #derivative goes in the last index
        #df/dz = f_x -i f_y/2.
        # dXcomplex_dg = dXreal_dg[:, :,:,0:self.phimodel.ncoords]
        # dXcomplex_dg -= 1j*dXreal_dg[:, :,:,self.phimodel.ncoords:]
        # dXcomplex_dg *= 1/2
        dXcomplex_dRg = dXreal_dRg[:, :,:,0:self.phimodel.ncoords]
        dXcomplex_dRg -= 1j*dXreal_dRg[:, :,:,self.phimodel.ncoords:]
        dXcomplex_dRg *= 1/2
        dXcomplex_dIg = dXreal_dIg[:, :,:,0:self.phimodel.ncoords]
        dXcomplex_dIg -= 1j*dXreal_dIg[:, :,:,self.phimodel.ncoords:]
        dXcomplex_dIg *= 1/2
        dXcomplex_dg=(dXcomplex_dRg+1j*dXcomplex_dIg)

        #OMIT the pullback, as we don't want to have to take the derivative of the pullback
        gammac_Ib = tf.einsum('xDc,xbDk->xckb', inverseg,#k index is the derivative index, capital indicates conjugate
                                 dXcomplex_dg)#ck is conjugated

        return gammac_Ib

    def compute_riemann_m_nb_rb_sbUP_uncomp(self,x):
        # take derivatives
        x_vars=x
        pullbacks = (self.phimodel.pullbacks(x_vars))
        pullbacksbar=tf.math.conj(pullbacks)
        #compute the pullbacks outside the gradienttape - this is fine due to holo/antiholo nature
        # do the contraction inside the gradienttape to minimise memory footprint
        print('starting tapeR')
        with tf.GradientTape(persistent=True) as tapeR:
            tapeR.watch(x_vars)
            gammaholo=self.compute_christoffel_symbols_holo_not_pb(x_vars)
            gammaantiholoK_MN=tf.math.conj(gammaholo)
            gammaantiholoK_AN = tf.einsum('xAI,xSIR->xSAR',pullbacksbar,gammaantiholoK_MN)
            RgammaantiholoK_AN=tf.math.real(gammaantiholoK_AN)
            IgammaantiholoK_AN=tf.math.imag(gammaantiholoK_AN)
        #with tapeR.stop_recording():
        print('only runs during compilation')
        print('first gradienttape')
        RdXreal_dGammaC= tf.cast(tapeR.batch_jacobian(RgammaantiholoK_AN, x_vars),dtype=tf.complex64)
        print('second gradienttape')
        IdXreal_dGammaC= tf.cast(tapeR.batch_jacobian(IgammaantiholoK_AN, x_vars),dtype=tf.complex64)
        del tapeR
        print('tapes deleted')

        # add derivatives together to complex tensor
        #derivative goes in the last index
        #df/dz = f_x -i f_y/2.
        RdXcomplex_dGammaC = RdXreal_dGammaC[:, :,:,:,0:self.phimodel.ncoords]
        RdXcomplex_dGammaC -= 1j*RdXreal_dGammaC[:, :,:,:,self.phimodel.ncoords:]
        RdXcomplex_dGammaC *= 1/2#has index structure g^c_aB,ki
        IdXcomplex_dGammaC = IdXreal_dGammaC[:, :,:,:,0:self.phimodel.ncoords]
        IdXcomplex_dGammaC -= 1j*IdXreal_dGammaC[:, :,:,:,self.phimodel.ncoords:]
        IdXcomplex_dGammaC *= 1/2#has index structure g^c_aB,k
        dXcomplex_dGammaC=RdXcomplex_dGammaC+(IdXcomplex_dGammaC)*1.j
        riemann_m_nb_rb_sbUP = tf.einsum('xmk,xSNRk->xmNRS', pullbacks,dXcomplex_dGammaC)#ck is conjugated

        return riemann_m_nb_rb_sbUP    

    def compute_Q_uncomp(self,x):
        ginverse=tf.linalg.inv(self.phimodel(x))
        R_m_nb_rb_sbUP = self.compute_riemann_m_nb_rb_sbUP(x)
        R_m_nb_rUP_sbarUP = tf.einsum('xra,xmnrs->xmnas',ginverse,R_m_nb_rb_sbUP)# r is barred, a is initially hol, ends up antiholo after conj
        R_m_nUP_r_sUP = -1*tf.math.conj(tf.einsum('xam,xmnrs->xnars',ginverse,R_m_nb_rb_sbUP))
        term1 = tf.einsum('xabcd,xcdef,xefab->x', R_m_nb_rUP_sbarUP, R_m_nb_rUP_sbarUP, R_m_nb_rUP_sbarUP)
        term2=tf.einsum('xacbd,xcedf,xeafb->x',R_m_nUP_r_sUP,R_m_nUP_r_sUP,R_m_nUP_r_sUP)
        return -8/(3*(2*np.pi)**3)*(term1-term2)#,1/(3*(2*np.pi)**3)*(term1),1/(3*(2*np.pi)**3)*(term2)
        #added minus sign so answer is -128
        #THIS MAY BE WRONG
    # added the factor of 8 back in


# def compute_batched_func(compute_Q,input_vector,batch_size,weights):
#     #returns the Q vector, and the sqrt(g) Q vector
#     print("computing batched with batch size " + str(batch_size) + " and total length " + str(len(input_vector)))
#     resultall2=[]
#     for i in range(0, len(input_vector), batch_size):
#         batch = input_vector[i:i+batch_size]
#         if len(batch)<batch_size:
#             #copy batch as many times as necessary until you get batch_size
#             batch=tf.concat([batch for _ in range((batch_size//len(batch))+1)],axis=0)[0:batch_size]
            
#         result=compute_Q(batch)
#         resultall2.append(result)
#         result_temp=tf.math.real(tf.concat(resultall2,axis=0))
#         #fix incorrect length in final batch
#         length=min(len(input_vector),len(result_temp))
#         euler_all=weights[0:length]*result_temp[0:length]
#         euler=tf.reduce_mean(euler_all)
#         vol=tf.reduce_mean(weights[0:length])
#         print("in " + str(i+batch_size) + " euler: " + str(euler.numpy())+  " vol " + str(vol.numpy()))
#     #concatenate and fix length issue, also cast to real as it should be/is real
#     resultarr2=tf.math.real(tf.concat(resultall2,axis=0)[:len(input_vector)])
#     return resultarr2, euler_all


def compute_batched_func(compute_Q, input_vector, batch_size, weights):
    total_length = tf.shape(input_vector)[0]
    num_batches = (total_length + batch_size - 1) // batch_size

    result_array = tf.TensorArray(tf.float32, size=num_batches)
    euler_sum = tf.constant(0.0)
    weight_sum = tf.constant(0.0)

    for i in tf.range(num_batches):
        start = i * batch_size
        end = tf.minimum((i + 1) * batch_size, total_length)
        batch = input_vector[start:end]

        current_batch_size = tf.shape(batch)[0]
        #fix incorrect length in final batch
        if current_batch_size < batch_size:
            repeat_times = tf.cast(tf.math.ceil(batch_size / current_batch_size), tf.int32)
            batch=tf.tile(batch, [repeat_times,1])[0:batch_size]

        result = tf.math.real(compute_Q(batch))
        result_array = result_array.write(i, result)

        batch_weights = weights[start:end]
        number_of_points=tf.shape(batch_weights)[0]#can be different to batch_size on laast iteration
        batch_result = tf.math.real(result[:number_of_points])
        batch_euler = tf.reduce_sum(batch_weights * batch_result)
        batch_weight_sum = tf.reduce_sum(batch_weights)

        euler_sum += batch_euler
        weight_sum += batch_weight_sum

        tf.print("in", end, "euler:", euler_sum / float(min((i+1)*batch_size,total_length)), "vol", weight_sum/float(min((i+1)*batch_size,total_length)))

    #stack, concatenate and fix length issue, also cast to real as it should be/is real
    resultarr2 = tf.math.real(result_array.stack())
    resultarr2=tf.reshape(resultarr2,[-1])[:total_length]


    euler_all = weights[:total_length] * resultarr2

    return resultarr2, euler_all