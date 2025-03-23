#!/usr/bin/env python3
# Example usage of the new unified neural network configuration for phimodel

import os
import sys
import numpy as np
import tensorflow as tf

# Import the new NetworkConfig class and phi model functions
from network_config import (
    NetworkConfig, 
    get_small_phi_config,
    get_medium_phi_config, 
    get_large_phi_config,
    train_and_save_phi_model,
    load_phi_model,
    use_with_original_train_and_save_nn,
    use_with_original_load_nn_phimodel
)

# Import necessary functions from original code
from custom_networks import BiholoModelFuncGENERAL
from cymetric.models.tfhelper import prepare_tf_basis

def main():
    """Demonstrate the usage of NetworkConfig for phi models."""
    
    # Example: Creating and using different phi model configurations
    print("Available preset configurations:")
    print("1. Small phi model")
    print("2. Medium phi model")
    print("3. Large phi model")
    print("4. Custom phi model")
    
    # Define a custom phi model configuration
    custom_phi_config = NetworkConfig(
        name="phi_custom",
        depth=4,                # 4 hidden layers
        width=192,              # 192 neurons per hidden layer
        activation='gelu',      # GELU activation function
        learning_rate=0.0008,   # Learning rate
        weight_stddev=0.05,     # Lower weight initialization stddev
        batch_sizes=[224, 180000],  # Custom batch sizes
        epochs=150,             # 150 training epochs
        network_class=BiholoModelFuncGENERAL,
        use_symmetry_reduced=True,  # Use symmetry reduced implementation
        weight_decay=True,      # Use weight decay
        final_lr_factor=3       # Final learning rate will be initial/3
    )
    
    # Example of converting to/from dict for compatibility with old code
    config_dict = custom_phi_config.to_dict()
    print("\nCustom configuration as dictionary:")
    for key, value in config_dict.items():
        if key != 'network_function':  # Skip printing the function object
            print(f"  {key}: {value}")
    
    # Convert back from dict
    reconstructed_config = NetworkConfig.from_dict(config_dict)
    print("\nReconstructed configuration:")
    print(f"  name: {reconstructed_config.name}")
    print(f"  depth: {reconstructed_config.depth}")
    print(f"  width: {reconstructed_config.width}")
    
    # Example of how to use with existing train_and_save_nn function
    print("\nExample of how to use with existing code:")
    print("# Original code:")
    print("phimodel, training_history, measure_phi = train_and_save_nn(")
    print("    manifold_name_and_data, phimodel_config, use_zero_network=use_zero_network_phi)")
    print("\n# New code:")
    print("# 1. Create a configuration object")
    print("phi_config = get_medium_phi_config(name='my_phi_model')")
    print("# 2. Use the new training function")
    print("phimodel, training_history, measure_phi = train_and_save_phi_model(")
    print("    manifold_name_and_data, phi_config, use_zero_network=False)")
    
    # Example of how to modify a configuration
    print("\nExample of modifying a configuration:")
    medium_config = get_medium_phi_config()
    print(f"Default medium config: depth={medium_config.depth}, width={medium_config.width}")
    
    # Modify the configuration
    medium_config.depth = 4
    medium_config.width = 160
    medium_config.learning_rate = 0.0005
    print(f"Modified medium config: depth={medium_config.depth}, width={medium_config.width}")
    
    # Example of creating a custom configuration with additional parameters
    print("\nExample with additional parameters:")
    custom_config_with_extras = NetworkConfig(
        name="phi_with_extras",
        depth=3,
        width=128,
        additional_params={
            "dropout_rate": 0.1,
            "l2_regularization": 1e-5,
            "use_batch_norm": True
        }
    )
    print(f"Custom config with extras: {custom_config_with_extras.additional_params}")
    
    # Typical usage pattern
    print("\nTypical usage pattern:")
    print("1. Select or create a configuration")
    print("2. Train/load the model using that configuration")
    print("3. The configuration controls all aspects of the model architecture and training")
    
    # Demonstrate compatibility with original codebase
    print("\nCompatibility with original codebase:")
    print("# Import compatibility functions")
    print("from network_config import (")
    print("    use_with_original_train_and_save_nn,")
    print("    use_with_original_load_nn_phimodel,")
    print("    get_medium_phi_config")
    print(")")
    print("\n# Import original functions")
    print("from generate_and_train_all_nnsHOLO_all import train_and_save_nn, load_nn_phimodel")
    print("\n# Create a network configuration")
    print("phi_config = get_medium_phi_config('phi_model_1')")
    print("\n# Train using the original function with the new configuration")
    print("phimodel, history, _ = use_with_original_train_and_save_nn(")
    print("    train_and_save_nn,")
    print("    manifold_name_and_data,")
    print("    phi_config,")
    print("    use_zero_network=False,")
    print("    unique_name='phi_custom'")
    print(")")
    print("\n# Load using the original function with the new configuration")
    print("phimodel, history, _ = use_with_original_load_nn_phimodel(")
    print("    load_nn_phimodel,")
    print("    manifold_name_and_data,")
    print("    phi_config,")
    print("    skip_measures=True")
    print(")")

if __name__ == "__main__":
    # Set TensorFlow to only use CPU if needed
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Run the demonstration
    main() 