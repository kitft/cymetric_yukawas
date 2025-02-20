from cymetric.config import real_dtype, complex_dtype
from cymetric.models.fubinistudy import FSModel
from laplacian_funcs import *
from custom_networks import batch_process_helper_func
import tensorflow as tf
import os
import numpy as np
from pympler import tracker

def point_vec_to_complex(p):
    #if len(p) == 0: 
    #    return tf.constant([[]])
    plen = ((p[0]).shape)[-1]//2
    return tf.complex(p[:, :plen],p[:, plen:])



class BetaModel(FSModel):
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
    def __init__(self, tfmodel, BASIS,linebundleforHYM, alpha=None, **kwargs):
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
        super(BetaModel, self).__init__(BASIS=BASIS, **kwargs)
        self.model = tfmodel
        self.NLOSS = 2
        # variable or constant or just tensor?
        if alpha is not None:
            self.alpha = [tf.Variable(a, dtype=real_dtype) for a in alpha]
        else:
            self.alpha = [tf.Variable(1., dtype=real_dtype) for _ in range(self.NLOSS)]
        self.learn_transition = tf.cast(True, dtype=tf.bool)
        self.learn_laplacian = tf.cast(True, dtype=tf.bool)

        #self.learn_kaehler = tf.cast(False, dtype=tf.bool)
        #self.learn_ricci = tf.cast(False, dtype=tf.bool)
        #self.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        #self.learn_volk = tf.cast(False, dtype=tf.bool)

        self.custom_metrics = None
        #self.kappa = tf.cast(BASIS['KAPPA'], dtype=real_dtype)
        self.gclipping = float(5.0)
        # add to compile?
        #self.sigma_loss = sigma_loss(self.kappa, tf.cast(self.nfold, dtype=real_dtype))
        self.linebundleforHYM =linebundleforHYM


    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point. In the case of the Phi model, we demand that \phi(\lambda^q_i z_i)=\phi(z_i)

        Args:
            points (tf.tensor([bSize, 2*ncoords], real_dtype)): Points.

        Returns:
            tf.tensor([bSize], real_dtype): Transition loss at each point.
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
            points (tf.tensor([bSize, 2*ncoords], real_dtype)): Points.

        Returns:
            tf.tensor([bSize], real_dtype): Transition loss at each point.
        """
        lpl_losses=tf.math.abs(laplacian(self.model,x,pullbacks,invmetrics)-(sources))
        all_lpl_loss = lpl_losses**self.n[0]
        return all_lpl_loss


    def call(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the NN.

        .. math:: g_{\text{out}} = g_{\text{NN}}

        The additional arguments are included for inheritance reasons.

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], real_dtype)): Points.
            training (bool, optional): Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                Not used in this model. Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex):
                Prediction at each point.
        """
        # nn prediction
        #print("called model")
        #print(input_tensor.dtype)
        #print(type(input_tensor))
        cpoints=point_vec_to_complex(input_tensor)
        return tf.cast(self.raw_FS_HYM_c(cpoints),real_dtype)*tf.math.exp(self.model(input_tensor, training=training)[:,0])

    
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
        super(BetaModel, self).compile(**kwargs)

    def raw_FS_HYM_c(self,cpoints):

        linebundleforHYM=self.linebundleforHYM
        K1=tf.reduce_sum(cpoints[:,0:2]*tf.math.conj(cpoints[:,0:2]),1)
        K2=tf.reduce_sum(cpoints[:,2:4]*tf.math.conj(cpoints[:,2:4]),1)
        K3=tf.reduce_sum(cpoints[:,4:6]*tf.math.conj(cpoints[:,4:6]),1)
        K4=tf.reduce_sum(cpoints[:,6:8]*tf.math.conj(cpoints[:,6:8]),1)
        #generalise this
        return (K1**(-linebundleforHYM[0]))*(K2**(-linebundleforHYM[1]))*(K3**(-linebundleforHYM[2]))*(K4**(-linebundleforHYM[3]))
    
    def raw_FS_HYM_r(self,rpoints):
        cpoints=point_vec_to_complex(rpoints)
        linebundleforHYM=self.linebundleforHYM
        K1=tf.reduce_sum(cpoints[:,0:2]*tf.math.conj(cpoints[:,0:2]),1)
        K2=tf.reduce_sum(cpoints[:,2:4]*tf.math.conj(cpoints[:,2:4]),1)
        K3=tf.reduce_sum(cpoints[:,4:6]*tf.math.conj(cpoints[:,4:6]),1)
        K4=tf.reduce_sum(cpoints[:,6:8]*tf.math.conj(cpoints[:,6:8]),1)
        #generalise this
        return (K1**(-linebundleforHYM[0]))*(K2**(-linebundleforHYM[1]))*(K3**(-linebundleforHYM[2]))*(K4**(-linebundleforHYM[3]))


    def raw_FS_HYM_for_LB_c(self,cpoints,linebundleforHYM):
        K1=tf.reduce_sum(cpoints[:,0:2]*tf.math.conj(cpoints[:,0:2]),1)
        K2=tf.reduce_sum(cpoints[:,2:4]*tf.math.conj(cpoints[:,2:4]),1)
        K3=tf.reduce_sum(cpoints[:,4:6]*tf.math.conj(cpoints[:,4:6]),1)
        K4=tf.reduce_sum(cpoints[:,6:8]*tf.math.conj(cpoints[:,6:8]),1)
        #generalise this
        return (K1**(-linebundleforHYM[0]))*(K2**(-linebundleforHYM[1]))*(K3**(-linebundleforHYM[2]))*(K4**(-linebundleforHYM[3]))

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
        for g, var in zip(gradients, trainable_vars):
            if g is None:
                print(f"None gradient for variable: {var.name}")
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
        sample_weight = data["y_val"][:,0]
        sample_weight = sample_weight/tf.reduce_mean(sample_weight)
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

    # @tf.function
    # def to_hermitian(self, x):
    #     r"""Returns a hermitian tensor.
        
    #     Takes a tensor of length (-1,nfold**2) and transforms it
    #     into a (-1,nfold,nfold) hermitian matrix.

    #     Args:
    #         x (tensor[(-1,nfold**2), tf.float]): input tensor

    #     Returns:
    #         tensor[(-1,nfold,nfold), tf.float]: hermitian matrix
    #     """
    #     t1 = tf.reshape(tf.complex(x, tf.zeros_like(x)),
    #                     (-1, self.nfold, self.nfold))
    #     up = tf.linalg.band_part(t1, 0, -1)
    #     low = tf.linalg.band_part(1j * t1, -1, 0)
    #     out = up + tf.transpose(up, perm=[0, 2, 1]) - \
    #         tf.linalg.band_part(t1, 0, 0)
    #     return out + low + tf.transpose(low, perm=[0, 2, 1], conjugate=True)

    # @tf.function
    # def compute_volk_loss(self, input_tensor, wo, pred=None):
    #     r"""Computes volk loss.

    #     NOTE:
    #         This is an integral over the batch. Thus batch dependent.

    #     .. math::

    #         \mathcal{L}_{\text{vol}_k} = |\int_B g_{\text{FS}} -
    #             \int_B g_{\text{out}}|_n

    #     Args:
    #         input_tensor (tf.tensor([bSize, 2*ncoords], real_dtype)): Points.
    #         weights (tf.tensor([bSize], real_dtype)): Integration weights.
    #         pred (tf.tensor([bSize, nfold, nfold], complex_dtype), optional):
    #             Prediction from `self(input_tensor)`.
    #             If None will be calculated. Defaults to None.

    #     Returns:
    #         tf.tensor([bSize], real_dtype): Volk loss.
    #     """
    #     if pred is None:
    #         pred = self(input_tensor)
        
    #     aux_weights = tf.cast(wo[:, 0] / wo[:, 1], dtype=complex_dtype)
    #     aux_weights = tf.repeat(tf.expand_dims(aux_weights, axis=0), repeats=[len(self.BASIS['KMODULI'])], axis=0)
    #     # pred = tf.repeat(tf.expand_dims(pred, axis=0), repeats=[len(self.BASIS['KMODULI'])], axis=0)
    #     # ks = tf.eye(len(self.BASIS['KMODULI']), dtype=complex_dtype)
    #     # ks = tf.repeat(tf.expand_dims(self.fubini_study_pb(input_tensor), axis=0), repeats=[len(self.BASIS['KMODULI'])], axis=0)
    #     # input_tensor = tf.repeat(tf.expand_dims(input_tensor, axis=0), repeats=[len(self.BASIS['KMODULI'])], axis=0)
    #     # print(input_tensor.shape, pred.shape, ks.shape)
    #     # actual_slopes = tf.vectorized_map(self._calculate_slope, [input_tensor, pred, ks])
    #     ks = tf.eye(len(self.BASIS['KMODULI']), dtype=complex_dtype)

    #     def body(input_tensor, pred, ks, actual_slopes):
    #         f_a = self.fubini_study_pb(input_tensor, ts=ks[len(actual_slopes)])
    #         res = tf.expand_dims(self._calculate_slope([pred, f_a]), axis=0)
    #         actual_slopes = tf.concat([actual_slopes, res], axis=0)
    #         return input_tensor, pred, ks, actual_slopes

    #     def condition(input_tensor, pred, ks, actual_slopes):
    #         return len(actual_slopes) < len(self.BASIS['KMODULI'])

    #     f_a = self.fubini_study_pb(input_tensor, ts=ks[0])
    #     actual_slopes = tf.expand_dims(self._calculate_slope([pred, f_a]), axis=0)
    #     if len(self.BASIS['KMODULI']) > 1:
    #         _, _, _, actual_slopes = tf.while_loop(condition, body, [input_tensor, pred, ks, actual_slopes], shape_invariants=[input_tensor.get_shape(), pred.get_shape(), ks.get_shape(), tf.TensorShape([None, actual_slopes.shape[-1]])])
    #     actual_slopes = tf.reduce_mean(aux_weights * actual_slopes, axis=-1)
    #     loss = tf.reduce_mean(tf.math.abs(actual_slopes - self.slopes)**self.n[4])
        
    #     # return tf.repeat(tf.expand_dims(loss, axis=0), repeats=[input_tensor.shape[0]], axis=0)
    #     return tf.repeat(tf.expand_dims(loss, axis=0), repeats=[len(wo)], axis=0)

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


def prepare_dataset_HYM(point_gen, data,n_p, dirname, metricModel,linebundleforHYM,BASIS,val_split=0.1, ltails=0, rtails=0, normalize_to_vol_j=True):
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
        os.makedirs(dirname)
    # new_np = int(round(n_p/(1-ltails-rtails)))
    # pwo = point_gen.generate_point_weights(new_np, omega=True)
    # if len(pwo) < new_np:
    #     new_np = int((new_np-len(pwo))/len(pwo)*new_np + 100)
    #     pwo2 = point_gen.generate_point_weights(new_np, omega=True)
    #     pwo = np.concatenate([pwo, pwo2], axis=0)
    # new_np = len(pwo)
    # sorted_weights = np.sort(pwo['weight'])
    # lower_bound = sorted_weights[round(ltails*new_np)]
    # upper_bound = sorted_weights[round((1-rtails)*new_np)-1]
    # mask = np.logical_and(pwo['weight'] >= lower_bound,
    #                       pwo['weight'] <= upper_bound)
    # weights = np.expand_dims(pwo['weight'][mask], -1)
    # omega = np.expand_dims(pwo['omega'][mask], -1)
    # omega = np.real(omega * np.conj(omega))
    
    # #points = tf.cast(points,complex_dtype)
    

    # if normalize_to_vol_j:
    #     pbs = point_gen.pullbacks(points)
    #     fs_ref = point_gen.fubini_study_metrics(points, vol_js=np.ones_like(point_gen.kmoduli))
    #     fs_ref_pb = tf.einsum('xai,xij,xbj->xab', pbs, fs_ref, np.conj(pbs))
    #     aux_weights = omega.flatten() / weights.flatten()
    #     norm_fac = point_gen.vol_j_norm / np.mean(np.real(np.linalg.det(fs_ref_pb)) / aux_weights)
    #     #print("point_gen.vol_j_norm")
    #     #print(point_gen.vol_j_norm)
    #     weights = norm_fac * weights # I.E. this is vol_j_norm/ integral of g_FS. That is, we normalise our volume to d_rst 1 1 1, when it is calculated with integral of omega wedge omegabar, i.e. just the weights. I.e. sum over just weights is that.
    #     # not sure if the above explanation is correct

    # X_train = np.concatenate((points[:t_i].real, points[:t_i].imag), axis=-1)
    # y_train = np.concatenate((weights[:t_i], omega[:t_i]), axis=1)
    # X_val = np.concatenate((points[t_i:].real, points[t_i:].imag), axis=-1)
    # y_val = np.concatenate((weights[t_i:], omega[t_i:]), axis=1)

    
    
    # realpoints=tf.concat((tf.math.real(points), tf.math.imag(points)), axis=-1)
    # realpoints=tf.cast(realpoints,real_dtype)

    # X_train=tf.cast(X_train,real_dtype)
    # y_train=tf.cast(y_train,real_dtype)
    # X_val=tf.cast(X_val,real_dtype)
    # y_val=tf.cast(y_val,real_dtype)
    # #realpoints=tf.cast(realpoints,real_dtype)
    X_train=tf.cast(data['X_train'],real_dtype)
    y_train=tf.cast(data['y_train'],real_dtype)
    X_val=tf.cast(data['X_val'],real_dtype)
    y_val=tf.cast(data['y_val'],real_dtype)
    ncoords=int(len(X_train[0])/2)

    #y_train=data['y_train']
    #y_val=data['y_val']
    ys=tf.concat((y_train,y_val),axis=0)
    weights=tf.cast(tf.expand_dims(ys[:,0],axis=-1),real_dtype)
    omega=tf.cast(tf.expand_dims(ys[:,1],axis=-1),real_dtype)

    realpoints=tf.concat((X_train,X_val),axis=0)
    points=tf.complex(realpoints[:,0:ncoords],realpoints[:,ncoords:])
    new_np = len(realpoints)
    t_i = int((1-val_split)*new_np)

    #still need to generate pullbacks apparently
    pullbacks = point_gen.pullbacks(points)
    train_pullbacks=tf.cast(pullbacks[:t_i],complex_dtype) 
    val_pullbacks=tf.cast(pullbacks[t_i:],complex_dtype) 

    # points = pwo['point'][mask]

    #batch to make more neat
    mets = batch_process_helper_func(metricModel, (realpoints,), batch_indices=(0,), batch_size=10000)

    absdets = tf.abs(tf.linalg.det(mets))
    inv_mets=tf.linalg.inv(mets)
    inv_mets_train=inv_mets[:t_i]
    inv_mets_val=inv_mets[t_i:]
    #linebundleindicea=tf.convert_to_tensor(np.ones_like(kmoduli))
    #J_abbar = -ig_abbar
    #2jnp.pi*(kJ) is F, but J = -ig so it works out. Neecd to pull back
    #linebundleforHYM=np.array([-4,2,-2,-1])
    F_forsource = -1*(2*np.pi/1j)*(1j/2)*point_gen.fubini_study_metrics(points, vol_js=linebundleforHYM)#F_FSab = - 2 pi i  k J, and Jab = i/2g? No, that isn't true... J = ig???. In the end, we want F = -k_i/kappa_i^2, so it should be -1 for the ^(-k), then (pi/i) to counteract the J, then i to convert from g to J..
    F_forsource_pb= tf.einsum('xai,xij,xbj->xab', pullbacks, F_forsource, tf.math.conj(pullbacks))

    FS_metric_pb = tf.einsum('xai,xij,xbj->xab', pullbacks, point_gen.fubini_study_metrics(points, vol_js=point_gen.kmoduli), np.conj(pullbacks))
    FSmetricdets=tf.abs(tf.linalg.det(FS_metric_pb))
    FSmetricinv = tf.linalg.inv(FS_metric_pb)
    #print(tf.shape(fs_ref))
    
    #sourcesCY= tf.cast((1/2)*tf.einsum('xba,xab->x',inv_mets, F_forsource_pb),complex_dtype)# why the factor of 2???
    sourcesCY= tf.cast(tf.einsum('xba,xab->x',tf.cast(inv_mets,complex_dtype), tf.cast(F_forsource_pb,complex_dtype)),complex_dtype)# why the factor of 2???
    sources_train=tf.cast(sourcesCY[:t_i],complex_dtype)
    sources_val=tf.cast(sourcesCY[t_i:],complex_dtype)
    
    
    print("check slope: isec, FSmetric, CYomega,CYdirect") 
    print(tf.einsum('abc,a,b,c',BASIS["INTNUMS"],point_gen.kmoduli,point_gen.kmoduli,linebundleforHYM))
    #print(weights[:,0])
    #print(sources.shape)
    #print((weights*sources).shape)
    #g = phimodel(points)

    # use gamma series
    det = tf.math.real(absdets)  # * factorial / (2**nfold)
    #print("hi")
    det_over_omega = det / omega[:,0]
    #print("hi")
    volume_cy = tf.math.reduce_mean(weights[:,0], axis=-1)# according to raw CY omega calculation and sampling...
    #print("hi")
    vol_k = tf.math.reduce_mean(det_over_omega * weights[:,0], axis=-1)
    #print("hi")
    kappaover6 = tf.cast(vol_k,real_dtype) / tf.cast(volume_cy,real_dtype)
    #rint(ratio)
    #print("hi")
    tf.cast(kappaover6,real_dtype)
    weightscomp=tf.cast(weights[:,0],complex_dtype)
    #print("hi")
    det = tf.cast(det,real_dtype)
    print('kappa over 6 ')
    print(kappaover6)
    #print(norm_fac)
    #print(norm_fac)
    #slopefromvolCYrhoCY=(2/np.pi)*(1/(ratio))*tf.math.real(tf.reduce_mean(weightscomp*sourcesCY, axis=-1))
    volfromCY=tf.math.real(tf.reduce_mean(weightscomp, axis=-1))*kappaover6
    slopefromvolCYrhoCY=(1/6)*(2/np.pi)*tf.math.real(tf.reduce_mean(weightscomp*sourcesCY, axis=-1))*kappaover6
    # slopeCY2=(2/np.pi)*tf.reduce_mean((omega.flatten() / weights.flatten())* weights.flatten() *sourcesCY, axis=-1)
    print("CY volume and slope: " +str(volfromCY)+" and " + str(slopefromvolCYrhoCY))# this is just the integrate source
    integratedabsolutesource=(1/6)*(2/np.pi)*tf.math.real(tf.reduce_mean(weights[:,0]*tf.math.abs(sourcesCY), axis=-1))*kappaover6
    print("Integrated slope but with absolute val: " + str(integratedabsolutesource))
    # print(slopeCY2)
    #xba as the inverse has bbar first then 
    #print(FSmetricinv.shape)
    #print(fs_forsource.shape)
    #sourceFS=(1/2)*tf.einsum('xba,xab->x',FSmetricinv,F_forsource_pb)# again why the factor of 2?
    sourceFS=tf.cast(-tf.einsum('xba,xab->x',FSmetricinv,F_forsource_pb),real_dtype)# again why the factor of 2?
    #print(FSmetricdets[0:3])
    #slopefromvolFSrhoFS=(2/np.pi)*(1/(6*ratio))*tf.reduce_mean((weights[:,0]/det)* tf.cast(FSmetricdets,real_dtype) *sourceFS, axis=-1)
    #print('reduce')
    #print(tf.reduce_mean(tf.linalg.det(FS_metric_pb)))
    #slopefromvolFSrhoFS=(2/np.pi)*tf.reduce_mean((weights[:,0]/omega[:,0])* tf.cast(FSmetricdets,real_dtype) *sourceFS, axis=-1)/vol_k #vol_k is the actual CY volume.
    #slopefromvolFSrhoFS=(1/((3/2) * np.pi))*(2/np.pi)*(6*norm_fac*kappaover6)*tf.reduce_mean(weights[:,0]*(tf.cast(FSmetricdets,real_dtype)/omega[:,0])*sourceFS , axis=-1)#vol_k is the actual CY volume.
    slopefromvolFSrhoFS=(1/6)*(2/np.pi)*tf.reduce_mean(weights[:,0]*(tf.cast(FSmetricdets,real_dtype)/omega[:,0])*sourceFS , axis=-1)#vol_k is the actual CY volume.
    #volfromFSmetric=tf.reduce_mean((weights[:,0]/omega[:,0])* tf.cast(FSmetricdets,real_dtype) , axis=-1)/vol_k #vol_k is the actual CY volume.
    #volfromFSmetric=(6*norm_fac*kappaover6)*tf.reduce_mean(weights[:,0]*(tf.cast(FSmetricdets,real_dtype)/omega[:,0]) , axis=-1) #vol_k is the actual CY volume.
    volfromFSmetric=tf.reduce_mean(weights[:,0]*(tf.cast(FSmetricdets,real_dtype)/omega[:,0]) , axis=-1) #vol_k is the actual CY volume.
    print('FS vol and slope: ' + str(volfromFSmetric) + " " + str(slopefromvolFSrhoFS))
    #print(tf.reduce_mean(weights[:,0], axis=-1))
    ess = tf.square(tf.reduce_sum(weights[:,0])) / tf.reduce_sum(tf.square(weights[:,0]))
    error = 1/tf.sqrt(ess)
    print("ESS: ", ess)
    print("error: ", error)
    
    # save everything to compressed dict.
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


def train_modelbeta(betamodel, data_train, optimizer=None, epochs=50, batch_sizes=[64, 10000],
                verbose=1, custom_metrics=[], callbacks=[], sw=False):
    r"""Training loop for fixing the Kähler class. It consists of two 
    optimisation steps. 
        1. With a small batch size and volk loss disabled.
        2. With only MA and volk loss enabled and a large batchsize such that 
            the MC integral is a reasonable approximation and we don't lose 
            the MA progress from the first step.

    Args:
        betamodel (cymetric.models.tfmodels): Any of the custom metric models.
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
    learn_laplacian = betamodel.learn_laplacian
    learn_transition = betamodel.learn_transition
    if sw:
        sample_weights = data_train['y_train'][:, -2]
    else:
        sample_weights = None
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    #permint= tracker.SummaryTracker()
    #for epoch in range(epochs):
    for epoch in range(1):
        #print("internal")
        #print(permint.print_diff())
        batch_size = batch_sizes[0]
        betamodel.learn_transition = learn_transition
        betamodel.learn_laplacian = learn_laplacian
        betamodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
        if verbose > 0:
            print("\nEpoch {:2d}/{:d}".format(epoch + 1, epochs))

        history = betamodel.fit(
            data_train,
            epochs=epochs, batch_size=batch_size, verbose=verbose,
            callbacks=callbacks, sample_weight=sample_weights
        )
        #print(history)
        for k in history.history.keys():
            if k not in hist1.keys():
                hist1[k] = history.history[k]
            else:
                hist1[k] += history.history[k]
        #print("internal2")
        #print(permint.print_diff())
        # if history.history['transition_loss'][-1]<10**(-8):
        #     print("t_loss too low")
        #     break
        # batch_size = min(batch_sizes[1], len(data['X_train']))
        # betamodel.learn_kaehler = tf.cast(False, dtype=tf.bool)
        # betamodel.learn_transition = tf.cast(False, dtype=tf.bool)
        # betamodel.learn_ricci = tf.cast(False, dtype=tf.bool)
        # betamodel.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        # betamodel.learn_volk = tf.cast(True, dtype=tf.bool)
        # betamodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
        # history = betamodel.fit(
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
    return betamodel, training_history
