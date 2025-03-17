import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import os
from cymetric.config import real_dtype, complex_dtype
from cymetric.models.tfhelper import prepare_tf_basis, train_model

# Import necessary functions and classes
# The actual imports may need adjustment based on your codebase
from laplacian_funcs import  compute_transition_pointwise_measure
from custom_networks import (
    BiholoModelFuncGENERAL, 
    BiholoModelFuncGENERALforHYMinv3,
    BiholoModelFuncGENERALforHYMinv4,
    BiholoModelFuncGENERALforSigma2_m13,
    safe_log_abs
)

class NetworkConfig:
    """Unified configuration for neural network models."""
    
    def __init__(self, 
                 name='default',
                 depth=3, 
                 width=128, 
                 activation='gelu',
                 learning_rate=0.001,
                 weight_stddev=0.1,
                 batch_sizes=[192, 150000],
                 epochs=100,
                 network_class=BiholoModelFuncGENERAL,
                 optimizer_config=None,
                 #use_symmetry_reduced=False,#deprecated
                 weight_decay=None,
                 final_lr_factor=2,
                 additional_params=None):
        """
        Create a unified network configuration.
        
        Args:
            name: Unique identifier for this configuration
            depth: Number of hidden layers
            width: Width of each hidden layer
            activation: Activation function (string or TF function)
            learning_rate: Initial learning rate
            weight_stddev: Standard deviation for weight initialization
            batch_sizes: List of batch sizes [training, evaluation]
            epochs: Number of training epochs
            network_class: Class to use for network implementation
            optimizer_config: Dict of optimizer configuration parameters
            use_symmetry_reduced: Whether to use symmetry reduced implementation - should not be used, probably.
            weight_decay: Weight decay rate for optimizer (if any)
            final_lr_factor: Factor for final learning rate in scheduled decay
            additional_params: Dict of additional parameters specific to the network_class
        """
        self.name = name
        self.depth = depth
        self.width = width
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_stddev = weight_stddev
        self.batch_sizes = batch_sizes
        self.epochs = epochs
        self.network_class = network_class
        self.optimizer_config = optimizer_config or {}
        #self.use_symmetry_reduced = use_symmetry_reduced
        self.weight_decay = weight_decay
        self.final_lr_factor = final_lr_factor
        self.additional_params = additional_params or {}
        
    def create_model(self, basis, use_zero_network=False):
        """Create and return a model instance based on this configuration."""
        # Calculate layer sizes
        ambient = tf.cast(tf.math.abs(basis['AMBIENT']), tf.int32)
        
        # # Determine the first layer size based on whether to use symmetry reduction
        # if hasattr(self, 'use_symmetry_reduced') and self.use_symmetry_reduced:
        #     n_first_layer = tf.reduce_prod((np.array(ambient) + determine_n_funcs)).numpy().item()
        # else:
        n_first_layer = tf.reduce_prod((np.array(ambient) + 1)**2).numpy().item()
        
        # Build network shape
        shape_of_network = [n_first_layer] + [self.width] * self.depth + [1]
        
        # Convert string activation to TF function if needed
        activation = self.activation
        if isinstance(activation, str):
            if activation.lower() == 'gelu':
                activation = tf.keras.activations.gelu
            elif activation.lower() == 'relu':
                activation = tf.keras.activations.relu
            elif activation.lower() == 'tanh':
                activation = tf.keras.activations.tanh
            elif activation.lower() == 'sigmoid':
                activation = tf.keras.activations.sigmoid
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        
        # Extract additional parameters for the network class
        additional_params = {}
        if self.additional_params:
            additional_params = self.additional_params
        
        # Instantiate the model
        model_params = {
            'layer_sizes': shape_of_network,
            'BASIS': basis,
            'activation': activation,
            'stddev': self.weight_stddev,
            'use_zero_network': use_zero_network,
        }
        
        # Add symmetry reduced parameter if the network class supports it
        # if hasattr(self, 'use_symmetry_reduced'):
        #     model_params['use_symmetry_reduced_TQ'] = self.use_symmetry_reduced
            
        # Add any additional parameters
        model_params.update(additional_params)
        
        # Create the model instance
        print("network class: ", self.network_class)
        try:
            model = self.network_class(**model_params)
        except Exception as e:
            raise ValueError(f"Failed to create model instance: {str(e)}")
        
        return model
    
    def create_optimizer(self):
        """Create an optimizer based on this configuration."""
        if self.weight_decay:
            # Create optimizer with weight decay
            return create_adam_optimizer_with_decay(
                initial_learning_rate=self.learning_rate,
                nEpochs=self.epochs,
                final_lr_factor=self.final_lr_factor
            )
        else:
            # Use standard Adam optimizer
            return tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
    
    def to_dict(self):
        """Convert configuration to dictionary for saving/loading."""
        config_dict = {
            'depth': self.depth,
            'width': self.width,
            'nEpochs': self.epochs,
            'lRate': self.learning_rate,
            'stddev': self.weight_stddev,
            'bSizes': self.batch_sizes,
            'network_function': self.network_class,
            'activation': self.activation,
            #'use_symmetry_reduced': self.use_symmetry_reduced,
            'weight_decay': self.weight_decay,
            'final_lr_factor': self.final_lr_factor,
            'name': self.name
        }
        
        # Add any additional parameters
        if self.additional_params:
            for key, value in self.additional_params.items():
                config_dict[key] = value
                
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create a NetworkConfig from a dictionary."""
        # Extract standard parameters
        name = config_dict.get('name', 'default')
        depth = config_dict.get('depth', 3)
        width = config_dict.get('width', 128)
        activation = config_dict.get('activation', 'gelu')
        learning_rate = config_dict.get('lRate', 0.001)
        weight_stddev = config_dict.get('stddev', 0.1)
        batch_sizes = config_dict.get('bSizes', [192, 150000])
        epochs = config_dict.get('nEpochs', 100)
        network_class = config_dict.get('network_function', BiholoModelFuncGENERAL)
        #use_symmetry_reduced = config_dict.get('use_symmetry_reduced', False)
        weight_decay = config_dict.get('weight_decay', None)
        final_lr_factor = config_dict.get('final_lr_factor', 2)
        
        # Create additional params dictionary from remaining keys
        additional_params = {}
        for key, value in config_dict.items():
            if key not in ['depth', 'width', 'nEpochs', 'lRate', 'stddev', 'bSizes', 
                         'network_function', 'activation', 'use_symmetry_reduced',
                         'weight_decay', 'final_lr_factor', 'name']:
                additional_params[key] = value
        
        return cls(
            name=name,
            depth=depth,
            width=width,
            activation=activation,
            learning_rate=learning_rate,
            weight_stddev=weight_stddev,
            batch_sizes=batch_sizes,
            epochs=epochs,
            network_class=network_class,
            #use_symmetry_reduced=use_symmetry_reduced,
            weight_decay=weight_decay,
            final_lr_factor=final_lr_factor,
            additional_params=additional_params
        )

    # Add this new method for creating a copy
    def copy(self, **kwargs):
        """
        Create a copy of this NetworkConfig with optional parameter overrides.
        
        Args:
            **kwargs: Any parameters to override in the new config
            
        Returns:
            A new NetworkConfig instance
        """
        params = {
            'name': self.name,
            'depth': self.depth,
            'width': self.width,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'weight_stddev': self.weight_stddev,
            'batch_sizes': self.batch_sizes.copy(),
            'epochs': self.epochs,
            'network_class': self.network_class,
            'additional_params': self.additional_params.copy() if self.additional_params else None,
            'weight_decay': self.weight_decay
        }
        
        # Override any parameters provided
        params.update(kwargs)
        
        return NetworkConfig(**params)


def create_adam_optimizer_with_decay(initial_learning_rate, nEpochs, final_lr_factor=2):
    """
    Create Adam optimizer with learning rate decay.
    
    Args:
        initial_learning_rate: Initial learning rate
        nEpochs: Total number of epochs for training
        final_lr_factor: Factor by which learning rate should be divided at the end
        
    Returns:
        Configured Adam optimizer with learning rate schedule
    """
    # Create learning rate schedule that decays from initial_learning_rate to 
    # initial_learning_rate/final_lr_factor over the course of training
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate,
        nEpochs,
        initial_learning_rate / final_lr_factor,
        power=1.0  # Linear decay
    )
    
    # Create optimizer with the schedule
    return tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)


def convert_to_tensor_dict(data):
    """Convert numpy arrays in data dictionary to TensorFlow tensors."""
    tensor_dict = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            tensor_dict[key] = tf.convert_to_tensor(value)
        else:
            tensor_dict[key] = value
    return tensor_dict


def train_and_save_phi_model(manifold_name_and_data, network_config, 
                            use_zero_network=False, alpha=None):
    """
    Train and save a phi model using the unified NetworkConfig approach.
    
    Args:
        manifold_name_and_data: Tuple containing manifold data
        network_config: NetworkConfig instance with model parameters
        use_zero_network: Whether to use zero initialization
        alpha: List of alpha values for the PhiFSModel
        
    Returns:
        Tuple of (trained model, training history, metrics)
    """
    # Use default alpha if not provided
    if alpha is None:
        alpha = [1., 1., 30., 1., 2.]
    
    # Extract data paths
    coefficients, kmoduli, ambient, monomials, type_folder, unique_id_or_coeff, manifold_name, data_path = manifold_name_and_data
    dirname = os.path.join(data_path, type_folder, manifold_name+'_pg_with_'+str(unique_id_or_coeff))
    name = f"phimodel_for_{network_config.epochs}_{network_config.batch_sizes[0]}_{network_config.batch_sizes[1]}s{network_config.depth}x{network_config.width}_{network_config.name}"
    
    print('dirname:', dirname)
    print('name:', name)
    
    # Load data
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    data = convert_to_tensor_dict(data)
    BASIS = prepare_tf_basis(np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True))
    
    # Get callbacks and metrics
    from generate_and_train_all_nnsHOLO_all import getcallbacksandmetrics
    cb_list, cmetrics = getcallbacksandmetrics(data, 'phi', wandb=True)
    
    # Create models
    from cymetric.models.tfmodels import PhiFSModel
    
    # Create the neural networks
    nn_phi = network_config.create_model(BASIS, use_zero_network)
    nn_phi_zero = network_config.create_model(BASIS, use_zero_network=True)
    
    # Create the phi models
    phimodel = PhiFSModel(nn_phi, BASIS, alpha=alpha, unique_name=network_config.name)
    phimodelzero = PhiFSModel(nn_phi_zero, BASIS, alpha=alpha, unique_name=network_config.name)
    
    # Create optimizer
    opt = network_config.create_optimizer()
    
    # Compile models
    phimodel.compile(custom_metrics=cmetrics)
    phimodelzero.compile(custom_metrics=cmetrics)
    
    # Test validation loss before training
    datacasted = [data['X_val'], data['y_val']]
    phimodel.learn_transition = False
    phimodelzero.learn_transition = False
    phimodel.learn_volk = True
    phimodelzero.learn_volk = True
    
    print("Testing models before training...")
    valzero = phimodelzero.test_step(datacasted)
    valraw = phimodel.test_step(datacasted)
    print('Testing complete')
    
    valzero = {key: float(value.numpy()) for key, value in valzero.items()}
    valraw = {key: float(value.numpy()) for key, value in valraw.items()}
    
    # Set learning flags
    phimodel.learn_volk = False
    phimodel.learn_transition = False
    
    # Train the model
    print("Starting training...")
    phimodel, training_history = train_model(
        phimodel, data, optimizer=opt, 
        epochs=network_config.epochs, 
        batch_sizes=network_config.batch_sizes, 
        verbose=1, custom_metrics=cmetrics, 
        callbacks=cb_list
    )
    print("Training complete")
    
    # Save model weights and training history
    phimodel.model.save_weights(os.path.join(dirname, name) + '.weights.h5')
    np.savez_compressed(os.path.join(dirname, 'trainingHistory-' + name), training_history)
    
    # Evaluate final model
    phimodel.learn_transition = True
    phimodel.learn_volk = True
    valfinal = phimodel.test_step(datacasted)
    valfinal = {key: value.numpy() for key, value in valfinal.items()}
    
    # Print evaluation results
    print("Zero network validation loss:", valzero)
    print("Initial validation loss:", valraw)
    print("Final validation loss:", valfinal)
    print("Ratio of final to zero:", {key + " ratio": value/(valzero[key]+1e-8) for key, value in valfinal.items()})
    print("Ratio of final to raw:", {key + " ratio": value/(valraw[key]+1e-8) for key, value in valfinal.items()})
    
    # Compute additional metrics
    averagediscrepancyinstdevs, _, mean_t_discrepancy = compute_transition_pointwise_measure(phimodel, data["X_val"])
    print("Average transition discrepancy in standard deviations:", 
          averagediscrepancyinstdevs.numpy().item(), 
          "mean discrepancy:", mean_t_discrepancy.numpy().item())
    
    return phimodel, training_history, None


def load_phi_model(manifold_name_and_data, network_config, 
                  set_weights_to_zero=False, set_weights_to_random=False, 
                  skip_measures=False, alpha=None):
    """
    Load a previously trained phi model.
    
    Args:
        manifold_name_and_data: Tuple containing manifold data
        network_config: NetworkConfig instance with model parameters
        set_weights_to_zero: Whether to use zero initialization instead of loading weights
        set_weights_to_random: Whether to use random initialization instead of loading weights
        skip_measures: Skip validation measurements
        alpha: List of alpha values for the PhiFSModel
        
    Returns:
        Tuple of (loaded model, training history, metrics)
    """
    # Use default alpha if not provided
    if alpha is None:
        alpha = [1., 1., 30., 1., 2.]
    
    # Extract data paths
    coefficients, kmoduli, ambient, monomials, type_folder, unique_id_or_coeff, manifold_name, data_path = manifold_name_and_data
    dirname = os.path.join(data_path, type_folder, manifold_name+'_pg_with_'+str(unique_id_or_coeff))
    name = f"phimodel_for_{network_config.epochs}_{network_config.batch_sizes[0]}_{network_config.batch_sizes[1]}s{network_config.depth}x{network_config.width}_{network_config.name}"
    
    print('dirname:', dirname)
    print('name:', name)
    
    # Load data
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    data = convert_to_tensor_dict(data)
    BASIS = prepare_tf_basis(np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True))
    
    # Get callbacks and metrics
    from generate_and_train_all_nnsHOLO_all import getcallbacksandmetrics
    cb_list, cmetrics = getcallbacksandmetrics(data, 'phi', wandb=False)
    
    # Create the neural networks
    nn_phi = network_config.create_model(BASIS, use_zero_network=True)
    nn_phi_zero = network_config.create_model(BASIS, use_zero_network=True)
    
    # Create the phi models
    from cymetric.models.tfmodels import PhiFSModel
    phimodel = PhiFSModel(nn_phi, BASIS, alpha=alpha, unique_name=network_config.name)
    phimodelzero = PhiFSModel(nn_phi_zero, BASIS, alpha=alpha, unique_name=network_config.name)
    
    # Initialize weights by calling once
    datacasted = [data['X_val'], data['y_val']]
    phimodel(datacasted[0][0:1])
    phimodelzero(datacasted[0][0:1])
    
    # Handle different weight loading scenarios
    if set_weights_to_zero:
        print("SETTING WEIGHTS TO ZERO")
        training_history = 0
        if skip_measures:
            return phimodelzero, training_history, None
    elif set_weights_to_random:
        print("SETTING WEIGHTS TO RANDOM")
        training_history = 0
    else:
        # Load trained weights
        phimodel.model.load_weights(os.path.join(dirname, name) + '.weights.h5')
        training_history = np.load(os.path.join(dirname, 'trainingHistory-' + name + '.npz'), 
                                  allow_pickle=True)['arr_0'].item()
    
    if skip_measures:
        return phimodel, training_history, None
    
    # Compile models for evaluation
    phimodel.compile(custom_metrics=cmetrics)
    phimodelzero.compile(custom_metrics=cmetrics)
    
    # Evaluate models
    valzero = phimodelzero.evaluate(datacasted[0], datacasted[1], return_dict=True)
    valtrained = phimodel.evaluate(datacasted[0], datacasted[1], return_dict=True)
    
    # Print evaluation results
    print("Zero network validation loss:", valzero)
    print("Trained network validation loss:", valtrained)
    print("Ratio of trained to zero:", {key + " ratio": value/(valzero[key]+1e-8) for key, value in valtrained.items()})
    
    # Compute additional metrics
    averagediscrepancyinstdevs, _, mean_t_discrepancy = compute_transition_pointwise_measure(phimodel, data["X_val"])
    print("Average transition discrepancy in standard deviations:", 
          averagediscrepancyinstdevs, "mean discrepancy:", mean_t_discrepancy)
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    
    return phimodel, training_history, None


# Predefined configurations
def get_small_phi_config(name="phi_small"):
    """Return a small phi model configuration."""
    return NetworkConfig(
        name=name,
        depth=2,
        width=64,
        activation='gelu',
        learning_rate=0.001,
        weight_stddev=0.1,
        batch_sizes=[128, 100000],
        epochs=50,
        network_class=BiholoModelFuncGENERAL,
        #use_symmetry_reduced=False,
        weight_decay=True,
        final_lr_factor=2
    )

def get_medium_phi_config(name="phi_medium"):
    """Return a medium phi model configuration."""
    return NetworkConfig(
        name=name,
        depth=3,
        width=128,
        activation='gelu',
        learning_rate=0.001,
        weight_stddev=0.1,
        batch_sizes=[192, 150000],
        epochs=100,
        network_class=BiholoModelFuncGENERAL,
        #use_symmetry_reduced=False,
        weight_decay=True,
        final_lr_factor=2
    )

def get_large_phi_config(name="phi_large"):
    """Return a large phi model configuration."""
    return NetworkConfig(
        name=name,
        depth=5,
        width=256,
        activation='gelu',
        learning_rate=0.0005,
        weight_stddev=0.1,
        batch_sizes=[256, 200000],
        epochs=200,
        network_class=BiholoModelFuncGENERAL,
        #use_symmetry_reduced=True,
        weight_decay=True,
        final_lr_factor=2
    )

# Compatibility functions to bridge with existing codebase
def phi_config_to_dict(network_config):
    """
    Convert NetworkConfig to the dictionary format expected by the original code.
    
    Args:
        network_config: NetworkConfig instance
        
    Returns:
        Dictionary compatible with original train_and_save_nn function
    """
    return network_config.to_dict()


def dict_to_phi_config(config_dict):
    """
    Convert original dictionary configuration to NetworkConfig.
    
    Args:
        config_dict: Dictionary from original codebase
        
    Returns:
        NetworkConfig instance
    """
    return NetworkConfig.from_dict(config_dict)


def use_with_original_train_and_save_nn(train_func, manifold_name_and_data, network_config, **kwargs):
    """
    Wrapper to use NetworkConfig with the original train_and_save_nn function.
    
    Args:
        train_func: The original train_and_save_nn function
        manifold_name_and_data: Data required by train_func
        network_config: NetworkConfig instance
        **kwargs: Additional arguments for train_func
        
    Returns:
        Output of train_func
    """
    # Convert NetworkConfig to dict format expected by original function
    config_dict = phi_config_to_dict(network_config)
    
    # Call the original function with the dict and additional arguments
    return train_func(manifold_name_and_data, config_dict, **kwargs)


def use_with_original_load_nn_phimodel(load_func, manifold_name_and_data, network_config, **kwargs):
    """
    Wrapper to use NetworkConfig with the original load_nn_phimodel function.
    
    Args:
        load_func: The original load_nn_phimodel function
        manifold_name_and_data: Data required by load_func
        network_config: NetworkConfig instance
        **kwargs: Additional arguments for load_func
        
    Returns:
        Output of load_func
    """
    # Convert NetworkConfig to dict format expected by original function
    config_dict = phi_config_to_dict(network_config)
    
    # Call the original function with the dict and additional arguments
    return load_func(manifold_name_and_data, config_dict, **kwargs)


def integrate_with_do_model():
    """
    Example of how to integrate NetworkConfig with the existing do_model.py file.
    
    This function shows the minimal changes needed to update do_model.py to use
    the new NetworkConfig class for phi models while maintaining compatibility
    with the rest of the code.
    """
    # Import necessary modules (this would be at the top of do_model.py)
    # from network_config import NetworkConfig, get_medium_phi_config, phi_config_to_dict
    
    # Original code in do_model.py (simplified)
    """
    # Original configuration as dictionary
    phimodel_config = {
        'depth': depthPhi, 
        'width': widthPhi, 
        'nEpochs': nEpochsPhi, 
        'lRate': lRatePhi, 
        'stddev': stddev_phi, 
        'bSizes': [tr_batchsize, SecondBSize], 
        'network_function': phi_model_load_function, 
        'activation': activationphi
    }
    
    # Train or load the model
    if not load_phi:
        phimodel, training_history, measure_phi = train_and_save_nn(
            manifold_name_and_data, phimodel_config, 
            use_zero_network=use_zero_network_phi, 
            unique_name=unique_name_phi
        )
    else:
        phimodel, training_history, measure_phi = load_nn_phimodel(
            manifold_name_and_data, phimodel_config,
            set_weights_to_zero=return_zero_phi,
            skip_measures=skip_measuresPhi,
            set_weights_to_random=return_random_phi, 
            unique_name=unique_name_phi
        )
    """
    
    # Modified code using NetworkConfig
    """
    # Create network configuration using NetworkConfig class
    phi_config = NetworkConfig(
        name=unique_name_phi,
        depth=depthPhi,
        width=widthPhi,
        activation=activationphi,
        learning_rate=lRatePhi,
        weight_stddev=stddev_phi,
        batch_sizes=[tr_batchsize, SecondBSize],
        epochs=nEpochsPhi,
        network_class=phi_model_load_function,
        #use_symmetry_reduced=use_symmetry_reduced,
        weight_decay=True
    )
    
    # Option 1: Convert to dictionary for compatibility with existing functions
    phimodel_config = phi_config_to_dict(phi_config)
    
    # Train or load using the original functions (unchanged)
    if not load_phi:
        phimodel, training_history, measure_phi = train_and_save_nn(
            manifold_name_and_data, phimodel_config, 
            use_zero_network=use_zero_network_phi, 
            unique_name=unique_name_phi
        )
    else:
        phimodel, training_history, measure_phi = load_nn_phimodel(
            manifold_name_and_data, phimodel_config,
            set_weights_to_zero=return_zero_phi,
            skip_measures=skip_measuresPhi,
            set_weights_to_random=return_random_phi, 
            unique_name=unique_name_phi
        )
    
    # Option 2: Use the new functions directly
    if not load_phi:
        phimodel, training_history, measure_phi = train_and_save_phi_model(
            manifold_name_and_data, phi_config, 
            use_zero_network=use_zero_network_phi
        )
    else:
        phimodel, training_history, measure_phi = load_phi_model(
            manifold_name_and_data, phi_config,
            set_weights_to_zero=return_zero_phi,
            skip_measures=skip_measuresPhi,
            set_weights_to_random=return_random_phi
        )
    
    # Option 3: Use wrapper functions
    if not load_phi:
        from network_config import use_with_original_train_and_save_nn
        phimodel, training_history, measure_phi = use_with_original_train_and_save_nn(
            train_and_save_nn,
            manifold_name_and_data, 
            phi_config, 
            use_zero_network=use_zero_network_phi, 
            unique_name=unique_name_phi
        )
    else:
        from network_config import use_with_original_load_nn_phimodel
        phimodel, training_history, measure_phi = use_with_original_load_nn_phimodel(
            load_nn_phimodel,
            manifold_name_and_data, 
            phi_config,
            set_weights_to_zero=return_zero_phi,
            skip_measures=skip_measuresPhi,
            set_weights_to_random=return_random_phi, 
            unique_name=unique_name_phi
        )
    """
    
    return "See docstring for integration example"


# Beta Model Configuration

def get_small_beta_config(name="beta_small"):
    """Return a small beta model configuration."""
    return NetworkConfig(
        name=name,
        depth=2,
        width=64,
        activation=tf.square,  # Default activation for beta models
        learning_rate=0.001,
        weight_stddev=0.1,
        batch_sizes=[128, 10000],
        epochs=50,
        network_class=BiholoModelFuncGENERALforHYMinv3,  # Import from custom_networks
        #use_symmetry_reduced=False,
        weight_decay=True,
        final_lr_factor=2,
        additional_params={
            "alpha": [1.0, 1.0]  # Default alpha values for beta model
        }
    )

def get_medium_beta_config(name="beta_medium"):
    """Return a medium beta model configuration."""
    return NetworkConfig(
        name=name,
        depth=3,
        width=128,
        activation=tf.square,
        learning_rate=0.001,
        weight_stddev=0.1,
        batch_sizes=[192, 15000],
        epochs=100,
        network_class=BiholoModelFuncGENERALforHYMinv3,
        #use_symmetry_reduced=False,
        weight_decay=True,
        final_lr_factor=2,
        additional_params={
            "alpha": [1.0, 1.0]
        }
    )

def get_large_beta_config(name="beta_large"):
    """Return a large beta model configuration."""
    return NetworkConfig(
        name=name,
        depth=5,
        width=256,
        activation=tf.square,
        learning_rate=0.0005,
        weight_stddev=0.1,
        batch_sizes=[256, 20000],
        epochs=200,
        network_class=BiholoModelFuncGENERALforHYMinv3,
        #use_symmetry_reduced=True,
        weight_decay=True,
        final_lr_factor=2,
        additional_params={
            "alpha": [1.0, 1.0]
        }
    )

def train_and_save_beta_model(manifold_name_and_data, linebundleforHYM, network_config,
                             phimodel, use_zero_network=False, load_network=False):
    """
    Train and save a beta model using the unified NetworkConfig approach.
    """
    # Extract network configuration parameters
    name = network_config.name
    alpha = network_config.additional_params.get('alpha', [1., 10.])
    
    # Extract data paths
    coefficients, kmoduli, ambient, monomials, type_folder, unique_id_or_coeff, manifold_name, data_path = manifold_name_and_data
    
    # Create the correct directory names matching the generate methods
    lbstring = str(linebundleforHYM).replace(' ', '')
    dirnameForMetric = os.path.join(data_path, type_folder, f"{manifold_name}_pg_with_{unique_id_or_coeff}")
    dirname = os.path.join(data_path, type_folder, f"{manifold_name}HYM_pg_with_{unique_id_or_coeff}forLB_{lbstring}_using_{phimodel.unique_name}")
    
    model_name = f"betamodel_for_{network_config.epochs}_{network_config.batch_sizes[0]}s{network_config.depth}x{network_config.width}_{name}"
    
    print('dirname:', dirname)
    print('dirnameForMetric:', dirnameForMetric)
    print('name:', model_name)
    
    # Load data from the correct locations
    try:
        data = np.load(os.path.join(dirname, 'dataset.npz'))
        data = convert_to_tensor_dict(data)
        BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Get callbacks and metrics
    from generate_and_train_all_nnsHOLO_all import getcallbacksandmetricsHYM
    cb_list, cmetrics = getcallbacksandmetricsHYM(data, name, wandb=True)
    
    # Create the neural network - much cleaner now
    nn_beta = network_config.create_model(BASIS, use_zero_network)
    nn_beta_zero = network_config.create_model(BASIS, use_zero_network=True)
    
    # Create beta model - pass alpha here
    from BetaModel import BetaModel
    betamodel = BetaModel(nn_beta, BASIS, alpha=alpha)
    betamodelzero = BetaModel(nn_beta_zero, BASIS, alpha=alpha)
    
    # Create optimizer
    opt = network_config.create_optimizer()
    
    # Compile models
    betamodel.compile(optimizer=opt, custom_metrics=cmetrics)
    betamodelzero.compile(optimizer=opt, custom_metrics=cmetrics)
    
    # Test validation loss before training
    datacasted = (data['X_val'], data['y_val'])
    
    print("Testing models before training...")
    valzero = betamodelzero.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    valraw = betamodel.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    print('Testing complete')
    
    # Train the model
    print("Starting training...")
    betamodel, training_history = train_model(
        betamodel, data, optimizer=opt, 
        epochs=network_config.epochs, 
        batch_sizes=network_config.batch_sizes, 
        verbose=1, custom_metrics=cmetrics, 
        callbacks=cb_list
    )
    print("Training complete")
    
    # Save model weights and training history
    betamodel.model.save_weights(os.path.join(dirname, model_name) + '.weights.h5')
    np.savez_compressed(os.path.join(dirname, 'trainingHistory-' + model_name), training_history)
    
    # Evaluate final model
    valfinal = betamodel.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    
    # Print evaluation results
    print("Zero network validation loss:", valzero)
    print("Initial validation loss:", valraw)
    print("Final validation loss:", valfinal)
    print("Ratio of final to zero:", {key + " ratio": valfinal[key]/(valzero[key]+1e-8) for key in valfinal.keys()})
    print("Ratio of final to raw:", {key + " ratio": valfinal[key]/(valraw[key]+1e-8) for key in valfinal.keys()})
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    
    return betamodel, training_history, None

def load_beta_model(manifold_name_and_data, linebundleforHYM, network_config,
                   phimodel, set_weights_to_zero=False, set_weights_to_random=False, 
                   skip_measures=False):
    """
    Load a previously trained beta model.
    """
    # Extract network configuration parameters
    name = network_config.name
    alpha = network_config.additional_params.get('alpha', [1., 10.])
    
    # Extract data paths
    coefficients, kmoduli, ambient, monomials, type_folder, unique_id_or_coeff, manifold_name, data_path = manifold_name_and_data
    
    # Create the correct directory names
    lbstring = str(linebundleforHYM).replace(' ', '')
    dirname = os.path.join(data_path, type_folder, manifold_name+'HYM_pg_with_'+str(unique_id_or_coeff)+'forLB_'+lbstring+'_using_'+phimodel.unique_name)
    dirname_for_metric = os.path.join(data_path, type_folder, manifold_name+'_pg_with_'+str(unique_id_or_coeff))
    
    model_name = f"betamodel_for_{network_config.epochs}_{network_config.batch_sizes[0]}s{network_config.depth}x{network_config.width}_{name}"
    
    print('dirname:', dirname)
    print('dirname_for_metric:', dirname_for_metric)
    print('name:', model_name)
    
    # Load data - use dirname for dataset, but dirname_for_metric for basis
    try:
        data = np.load(os.path.join(dirname, 'dataset.npz'))
        data = convert_to_tensor_dict(data)
        # Load basis from the metric model directory
        BASIS = prepare_tf_basis(np.load(os.path.join(dirname_for_metric, 'basis.pickle'), allow_pickle=True))
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Get callbacks and metrics
    from generate_and_train_all_nnsHOLO_all import getcallbacksandmetricsHYM
    cb_list, cmetrics = getcallbacksandmetricsHYM(data, name, wandb=False)
    
    # Create the neural network - much cleaner now
    nn_beta = network_config.create_model(BASIS, use_zero_network=True)
    nn_beta_zero = network_config.create_model(BASIS, use_zero_network=True)
    
    # Create beta model - pass alpha here
    from BetaModel import BetaModel
    betamodel = BetaModel(nn_beta, BASIS, alpha=alpha)
    betamodelzero = BetaModel(nn_beta_zero, BASIS, alpha=alpha)
    
    # Initialize by calling once
    datacasted = (data['X_val'], data['y_val'])
    betamodel(datacasted[0][0:1])
    betamodelzero(datacasted[0][0:1])
    
    # Create optimizer
    opt = network_config.create_optimizer()
    
    # Compile models
    betamodel.compile(optimizer=opt, custom_metrics=cmetrics)
    betamodelzero.compile(optimizer=opt, custom_metrics=cmetrics)
    
    # Handle different weight loading scenarios
    if set_weights_to_zero:
        print("SETTING WEIGHTS TO ZERO")
        training_history = 0
        if skip_measures:
            return betamodelzero, training_history, None
    elif set_weights_to_random:
        print("SETTING WEIGHTS TO RANDOM")
        training_history = 0
    else:
        # Load trained weights
        try:
            betamodel.model.load_weights(os.path.join(dirname, model_name) + '.weights.h5')
            training_history = np.load(os.path.join(dirname, 'trainingHistory-' + model_name + '.npz'), 
                                    allow_pickle=True)['arr_0'].item()
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using random weights instead")
            training_history = 0
    
    if skip_measures:
        return betamodel, training_history, None
    
    # Evaluate models
    valzero = betamodelzero.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    valtrained = betamodel.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    
    # Print evaluation results
    print("Zero network validation loss:", valzero)
    print("Trained network validation loss:", valtrained)
    print("Ratio of trained to zero:", {key + " ratio": valtrained[key]/(valzero[key]+1e-8) for key in valtrained.keys()})
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    
    return betamodel, training_history, None


# Sigma Model Configuration (Harmonic Form Model)

def get_small_sigma_config(name="sigma_small"):
    """Return a small sigma (harmonic form) model configuration."""
    return NetworkConfig(
        name=name,
        depth=2,
        width=64,
        activation='gelu',
        learning_rate=0.001,
        weight_stddev=0.1,
        batch_sizes=[128, 10000],
        epochs=50,
        network_class=BiholoModelFuncGENERALforSigma2_m13,  # Import from custom_networks
        use_symmetry_reduced=False,
        weight_decay=True,
        final_lr_factor=2,
        additional_params={
            "alpha": [1.0, 10.0],  # Default alpha values for sigma model
            "final_layer_scale": 0.1,
            "norm_momentum": 0.9
        }
    )

def get_medium_sigma_config(name="sigma_medium"):
    """Return a medium sigma (harmonic form) model configuration."""
    return NetworkConfig(
        name=name,
        depth=3,
        width=128,
        activation='gelu',
        learning_rate=0.001,
        weight_stddev=0.1,
        batch_sizes=[192, 15000],
        epochs=100,
        network_class=BiholoModelFuncGENERALforSigma2_m13,
        use_symmetry_reduced=False,
        weight_decay=True,
        final_lr_factor=2,
        additional_params={
            "alpha": [1.0, 10.0],
            "final_layer_scale": 0.1,
            "norm_momentum": 0.9
        }
    )

def get_large_sigma_config(name="sigma_large"):
    """Return a large sigma (harmonic form) model configuration."""
    return NetworkConfig(
        name=name,
        depth=5,
        width=256,
        activation='gelu',
        learning_rate=0.0005,
        weight_stddev=0.1,
        batch_sizes=[256, 20000],
        epochs=200,
        network_class=BiholoModelFuncGENERALforSigma2_m13,
        use_symmetry_reduced=True,
        weight_decay=True,
        final_lr_factor=2,
        additional_params={
            "alpha": [1.0, 10.0],
            "final_layer_scale": 0.1,
            "norm_momentum": 0.9
        }
    )

def train_and_save_sigma_model(manifold_name_and_data, linebundleforHYM, betamodel, 
                              phimodel, functionforbaseharmonicform_jbar, network_config, 
                              use_zero_network=False, load_network=False):
    """
    Train and save a harmonic form (sigma) model using the unified NetworkConfig approach.
    """
    # Extract network configuration parameters
    name = network_config.name
    alpha = network_config.additional_params.get('alpha', [1., 10.])
    final_layer_scale = network_config.additional_params.get('final_layer_scale', 0.01)
    norm_momentum = network_config.additional_params.get('norm_momentum', 0.999)
    
    # Extract data paths
    coefficients, kmoduli, ambient, monomials, type_folder, unique_id_or_coeff, manifold_name, data_path = manifold_name_and_data
    
    # Get the name of the base harmonic form
    nameOfBaseHF = '_' + name  # Using name from config
    lbstring = str(linebundleforHYM).replace(' ', '')
    
    # Create the correct directory names matching the generate methods
    dirnameForMetric = os.path.join(data_path, type_folder, f"{manifold_name}_pg_with_{unique_id_or_coeff}")
    dirname = os.path.join(data_path, type_folder, f"{manifold_name}_HF_pg{unique_id_or_coeff}forLB_{lbstring}{nameOfBaseHF}_using_{phimodel.unique_name}_and_{betamodel.unique_name}")
    
    model_name = f"HFmodel_for_{network_config.epochs}_{network_config.batch_sizes[0]}s{network_config.depth}x{network_config.width}_{name}"
    
    print('dirname:', dirname)
    print('dirnameForMetric:', dirnameForMetric)
    print('name:', model_name)
    
    # Load data from the correct locations
    try:
        data = np.load(os.path.join(dirname, 'dataset.npz'))
        data = convert_to_tensor_dict(data)
        BASIS = prepare_tf_basis(np.load(os.path.join(dirnameForMetric, 'basis.pickle'), allow_pickle=True))
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Get callbacks and metrics
    from generate_and_train_all_nnsHOLO_all import getcallbacksandmetricsHF
    cb_list, cmetrics = getcallbacksandmetricsHF(data, name, wandb=True)
    
    # Create the neural network - much cleaner now
    nn_sigma = network_config.create_model(BASIS, use_zero_network)
    nn_sigma_zero = network_config.create_model(BASIS, use_zero_network=True)
    
    # Create harmonic form model - pass special parameters here
    from HarmonicFormModel import HarmonicFormModel
    
    sigmamodel = HarmonicFormModel(
        nn_sigma, BASIS, betamodel, phimodel, functionforbaseharmonicform_jbar,
        alpha=alpha, final_layer_scale=final_layer_scale, norm_momentum=norm_momentum
    )
    
    sigmamodelzero = HarmonicFormModel(
        nn_sigma_zero, BASIS, betamodel, phimodel, functionforbaseharmonicform_jbar,
        alpha=alpha, final_layer_scale=final_layer_scale, norm_momentum=norm_momentum
    )
    
    # Create optimizer
    opt = network_config.create_optimizer()
    
    # Compile models
    sigmamodel.compile(optimizer=opt, custom_metrics=cmetrics)
    sigmamodelzero.compile(optimizer=opt, custom_metrics=cmetrics)
    
    # Test validation loss before training
    datacasted = (data['X_val'], data['y_val'])
    
    print("Testing models before training...")
    valzero = sigmamodelzero.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    valraw = sigmamodel.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    print('Testing complete')
    
    # Train the model
    print("Starting training...")
    sigmamodel, training_history = train_model(
        sigmamodel, data, optimizer=opt, 
        epochs=network_config.epochs, 
        batch_sizes=network_config.batch_sizes, 
        verbose=1, custom_metrics=cmetrics, 
        callbacks=cb_list
    )
    print("Training complete")
    
    # Save model weights and training history
    sigmamodel.model.save_weights(os.path.join(dirname, model_name) + '.weights.h5')
    np.savez_compressed(os.path.join(dirname, 'trainingHistory-' + model_name), training_history)
    
    # Evaluate final model
    valfinal = sigmamodel.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    
    # Print evaluation results
    print("Zero network validation loss:", valzero)
    print("Initial validation loss:", valraw)
    print("Final validation loss:", valfinal)
    print("Ratio of final to zero:", {key + " ratio": valfinal[key]/(valzero[key]+1e-8) for key in valfinal.keys()})
    print("Ratio of final to raw:", {key + " ratio": valfinal[key]/(valraw[key]+1e-8) for key in valfinal.keys()})
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    
    return sigmamodel, training_history, None

def load_sigma_model(manifold_name_and_data, linebundleforHYM, betamodel,
                    phimodel, functionforbaseharmonicform_jbar, network_config,
                    set_weights_to_zero=False, set_weights_to_random=False, 
                    skip_measures=False):
    """
    Load a previously trained harmonic form (sigma) model.
    """
    # Extract network configuration parameters
    name = network_config.name
    alpha = network_config.additional_params.get('alpha', [1., 10.])
    final_layer_scale = network_config.additional_params.get('final_layer_scale', 0.01)
    norm_momentum = network_config.additional_params.get('norm_momentum', 0.999)
    
    # Extract data paths
    coefficients, kmoduli, ambient, monomials, type_folder, unique_id_or_coeff, manifold_name, data_path = manifold_name_and_data
    
    # Create the correct directory names
    lbstring = str(linebundleforHYM).replace(' ', '')
    nameOfBaseHF = '_' + name  # This should match the unique_name passed to the function
    
    dirname = os.path.join(data_path, type_folder, manifold_name+'_HF_pg'+str(unique_id_or_coeff)+'forLB_'+lbstring+nameOfBaseHF+'_using_'+phimodel.unique_name+'_and_'+betamodel.unique_name)
    dirname_for_metric = os.path.join(data_path, type_folder, manifold_name+'_pg_with_'+str(unique_id_or_coeff))
    
    model_name = f"HFmodel_for_{network_config.epochs}_{network_config.batch_sizes[0]}s{network_config.depth}x{network_config.width}_{name}"
    
    print('dirname:', dirname)
    print('dirname_for_metric:', dirname_for_metric)
    print('name:', model_name)
    
    # Load data - use dirname for dataset, but dirname_for_metric for basis
    try:
        data = np.load(os.path.join(dirname, 'dataset.npz'))
        data = convert_to_tensor_dict(data)
        # Load basis from the metric model directory
        BASIS = prepare_tf_basis(np.load(os.path.join(dirname_for_metric, 'basis.pickle'), allow_pickle=True))
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Get callbacks and metrics
    from generate_and_train_all_nnsHOLO_all import getcallbacksandmetricsHF
    cb_list, cmetrics = getcallbacksandmetricsHF(data, name, wandb=False)
    
    # Create the neural network - much cleaner now
    nn_sigma = network_config.create_model(BASIS, use_zero_network=True)
    nn_sigma_zero = network_config.create_model(BASIS, use_zero_network=True)
    
    # Create harmonic form model - pass special parameters here
    from HarmonicFormModel import HarmonicFormModel
    
    sigmamodel = HarmonicFormModel(
        nn_sigma, BASIS, betamodel, phimodel, functionforbaseharmonicform_jbar,
        alpha=alpha, final_layer_scale=final_layer_scale, norm_momentum=norm_momentum
    )
    
    sigmamodelzero = HarmonicFormModel(
        nn_sigma_zero, BASIS, betamodel, phimodel, functionforbaseharmonicform_jbar,
        alpha=alpha, final_layer_scale=final_layer_scale, norm_momentum=norm_momentum
    )
    
    # Initialize by calling once
    datacasted = (data['X_val'], data['y_val'])
    sigmamodel(datacasted[0][0:1])
    sigmamodelzero(datacasted[0][0:1])
    
    # Create optimizer
    opt = network_config.create_optimizer()
    
    # Compile models
    sigmamodel.compile(optimizer=opt, custom_metrics=cmetrics)
    sigmamodelzero.compile(optimizer=opt, custom_metrics=cmetrics)
    
    # Handle different weight loading scenarios
    if set_weights_to_zero:
        print("SETTING WEIGHTS TO ZERO")
        training_history = 0
        if skip_measures:
            return sigmamodelzero, training_history, None
    elif set_weights_to_random:
        print("SETTING WEIGHTS TO RANDOM")
        training_history = 0
    else:
        # Load trained weights
        try:
            sigmamodel.model.load_weights(os.path.join(dirname, model_name) + '.weights.h5')
            training_history = np.load(os.path.join(dirname, 'trainingHistory-' + model_name + '.npz'), 
                                     allow_pickle=True)['arr_0'].item()
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using random weights instead")
            training_history = 0
    
    if skip_measures:
        return sigmamodel, training_history, None
    
    # Evaluate models
    valzero = sigmamodelzero.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    valtrained = sigmamodel.evaluate(datacasted[0], datacasted[1], return_dict=True, batch_size=1000)
    
    # Print evaluation results
    print("Zero network validation loss:", valzero)
    print("Trained network validation loss:", valtrained)
    print("Ratio of trained to zero:", {key + " ratio": valtrained[key]/(valzero[key]+1e-8) for key in valtrained.keys()})
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    
    return sigmamodel, training_history, None


# Compatibility functions for beta model
def beta_config_to_dict(network_config):
    """Convert NetworkConfig to the dictionary format expected by the original code for beta models."""
    return network_config.to_dict()

def dict_to_beta_config(config_dict):
    """Convert original dictionary configuration to NetworkConfig for beta models."""
    return NetworkConfig.from_dict(config_dict)

def use_with_original_train_and_save_nn_HYM(train_func, manifold_name_and_data, linebundleforHYM, 
                                         network_config, phimodel, **kwargs):
    """
    Wrapper to use NetworkConfig with the original train_and_save_nn_HYM function.
    """
    config_dict = beta_config_to_dict(network_config)
    return train_func(manifold_name_and_data, linebundleforHYM, config_dict, phimodel, **kwargs)

def use_with_original_load_nn_HYM(load_func, manifold_name_and_data, linebundleforHYM, 
                                network_config, phimodel, **kwargs):
    """
    Wrapper to use NetworkConfig with the original load_nn_HYM function.
    """
    config_dict = beta_config_to_dict(network_config)
    return load_func(manifold_name_and_data, linebundleforHYM, config_dict, phimodel, **kwargs)


# Compatibility functions for sigma model
def sigma_config_to_dict(network_config):
    """Convert NetworkConfig to the dictionary format expected by the original code for sigma models."""
    return network_config.to_dict()

def dict_to_sigma_config(config_dict):
    """Convert original dictionary configuration to NetworkConfig for sigma models."""
    return NetworkConfig.from_dict(config_dict)

def use_with_original_train_and_save_nn_HF(train_func, manifold_name_and_data, linebundleforHYM, 
                                        betamodel, metric_model, functionforbaseharmonicform_jbar, 
                                        network_config, **kwargs):
    """
    Wrapper to use NetworkConfig with the original train_and_save_nn_HF function.
    """
    config_dict = sigma_config_to_dict(network_config)
    return train_func(manifold_name_and_data, linebundleforHYM, betamodel, metric_model, 
                     functionforbaseharmonicform_jbar, config_dict, **kwargs)

def use_with_original_load_nn_HF(load_func, manifold_name_and_data, linebundleforHYM, 
                               betamodel, metric_model, functionforbaseharmonicform_jbar, 
                               network_config, **kwargs):
    """
    Wrapper to use NetworkConfig with the original load_nn_HF function.
    """
    config_dict = sigma_config_to_dict(network_config)
    return load_func(manifold_name_and_data, linebundleforHYM, betamodel, metric_model, 
                    functionforbaseharmonicform_jbar, config_dict, **kwargs) 