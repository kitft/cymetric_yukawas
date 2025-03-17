#!/usr/bin/env python3
# Example usage of the new unified neural network configuration for beta and sigma models

import os
import sys
import numpy as np
import tensorflow as tf

# Import the new NetworkConfig class and model functions
from network_config import (
    # Configuration factories
    NetworkConfig,
    get_medium_phi_config,
    get_medium_beta_config,
    get_medium_sigma_config,
    
    # Training and loading functions
    train_and_save_phi_model,
    load_phi_model,
    train_and_save_beta_model,
    load_beta_model,
    train_and_save_sigma_model,
    load_sigma_model,
    
    # Compatibility wrappers
    use_with_original_train_and_save_nn_HYM,
    use_with_original_load_nn_HYM,
    use_with_original_train_and_save_nn_HF,
    use_with_original_load_nn_HF
)

# Import original functions for reference
from generate_and_train_all_nnsHOLO_all import (
    train_and_save_nn_HYM,
    load_nn_HYM,
    train_and_save_nn_HF,
    load_nn_HF
)

def example_beta_model():
    """Demonstrate the usage of NetworkConfig for beta models."""
    
    print("\n=== BETA MODEL CONFIGURATION EXAMPLES ===\n")
    
    # Create a custom beta model configuration
    custom_beta_config = NetworkConfig(
        name="beta_custom",
        depth=4,                # 4 hidden layers
        width=192,              # 192 neurons per hidden layer
        activation=tf.square,   # Square activation function (typical for beta models)
        learning_rate=0.0008,   # Learning rate
        weight_stddev=0.05,     # Lower weight initialization stddev
        batch_sizes=[224, 18000],  # Custom batch sizes
        epochs=150,             # 150 training epochs
        network_class=BiholoModelFuncGENERALforHYMinv3,  # Import from custom_networks
        use_symmetry_reduced=True,  # Use symmetry reduced implementation
        weight_decay=True,      # Use weight decay
        final_lr_factor=3,      # Final learning rate will be initial/3
        additional_params={
            "alpha": [1.0, 1.5]  # Custom alpha values for beta model
        }
    )
    
    # Show various ways to configure beta models
    print("Different beta model configurations:")
    print("\n1. Small beta config:")
    small_beta = get_small_beta_config()
    print(f"  Depth: {small_beta.depth}")
    print(f"  Width: {small_beta.width}")
    print(f"  Alpha: {small_beta.additional_params.get('alpha')}")
    
    print("\n2. Medium beta config:")
    medium_beta = get_medium_beta_config()
    print(f"  Depth: {medium_beta.depth}")
    print(f"  Width: {medium_beta.width}")
    print(f"  Alpha: {medium_beta.additional_params.get('alpha')}")
    
    print("\n3. Custom beta config:")
    print(f"  Depth: {custom_beta_config.depth}")
    print(f"  Width: {custom_beta_config.width}")
    print(f"  Alpha: {custom_beta_config.additional_params.get('alpha')}")
    
    # Training workflow example (pseudocode)
    print("\nExample training workflow (pseudocode):")
    print("# 1. Train phi model first")
    print("phi_config = get_medium_phi_config('my_phi')")
    print("phimodel, _, _ = train_and_save_phi_model(manifold_name_and_data, phi_config)")
    print("")
    print("# 2. Train beta model using the phi model")
    print("beta_config = get_medium_beta_config('my_beta')")
    print("betamodel, _, _ = train_and_save_beta_model(")
    print("    manifold_name_and_data,")
    print("    linebundleforHYM,")
    print("    beta_config,")
    print("    phimodel)")
    
    # Compatibility with original code
    print("\nCompatibility with original code:")
    print("# Using wrapper functions to integrate with existing code")
    print("# Using wrapper functions to integrate with existing code")
    print("from network_config import use_with_original_train_and_save_nn_HYM")
    print("beta_config = get_medium_beta_config('my_beta')")
    print("betamodel, _, _ = use_with_original_train_and_save_nn_HYM(")
    print("    train_and_save_nn_HYM,")
    print("    manifold_name_and_data,")
    print("    linebundleforHYM,")
    print("    beta_config,")
    print("    phimodel,")
    print("    use_zero_network=False)")
    
    return "Beta model examples completed"

def example_sigma_model():
    """Demonstrate the usage of NetworkConfig for sigma (harmonic form) models."""
    
    print("\n=== SIGMA MODEL CONFIGURATION EXAMPLES ===\n")
    
    # Create a custom sigma model configuration
    custom_sigma_config = NetworkConfig(
        name="sigma_custom",
        depth=4,                # 4 hidden layers
        width=224,              # 224 neurons per hidden layer
        activation='gelu',      # GELU activation (typical for sigma models)
        learning_rate=0.0007,   # Learning rate
        weight_stddev=0.08,     # Custom weight initialization stddev
        batch_sizes=[256, 20000],  # Custom batch sizes
        epochs=180,             # 180 training epochs
        network_class=BiholoModelFuncGENERALforSigma2_m13,
        use_symmetry_reduced=True,
        weight_decay=True,
        final_lr_factor=3,
        additional_params={
            "alpha": [1.2, 12.0],         # Custom alpha values
            "final_layer_scale": 0.08,    # Custom final layer scaling
            "norm_momentum": 0.92         # Custom normalization momentum
        }
    )
    
    # Show various ways to configure sigma models
    print("Different sigma model configurations:")
    print("\n1. Small sigma config:")
    small_sigma = get_small_sigma_config()
    print(f"  Depth: {small_sigma.depth}")
    print(f"  Width: {small_sigma.width}")
    print(f"  Alpha: {small_sigma.additional_params.get('alpha')}")
    print(f"  Final layer scale: {small_sigma.additional_params.get('final_layer_scale')}")
    
    print("\n2. Medium sigma config:")
    medium_sigma = get_medium_sigma_config()
    print(f"  Depth: {medium_sigma.depth}")
    print(f"  Width: {medium_sigma.width}")
    print(f"  Alpha: {medium_sigma.additional_params.get('alpha')}")
    print(f"  Final layer scale: {medium_sigma.additional_params.get('final_layer_scale')}")
    
    print("\n3. Custom sigma config:")
    print(f"  Depth: {custom_sigma_config.depth}")
    print(f"  Width: {custom_sigma_config.width}")
    print(f"  Alpha: {custom_sigma_config.additional_params.get('alpha')}")
    print(f"  Final layer scale: {custom_sigma_config.additional_params.get('final_layer_scale')}")
    
    # Training workflow example (pseudocode)
    print("\nExample training workflow (pseudocode):")
    print("# 1. Train phi model first")
    print("phi_config = get_medium_phi_config('my_phi')")
    print("phimodel, _, _ = train_and_save_phi_model(manifold_name_and_data, phi_config)")
    print("")
    print("# 2. Train beta model using the phi model")
    print("beta_config = get_medium_beta_config('my_beta')")
    print("betamodel, _, _ = train_and_save_beta_model(")
    print("    manifold_name_and_data,")
    print("    linebundleforHYM,")
    print("    beta_config,")
    print("    phimodel)")
    print("")
    print("# 3. Train sigma model using both phi and beta models")
    print("sigma_config = get_medium_sigma_config('my_sigma')")
    print("sigmamodel, _, _ = train_and_save_sigma_model(")
    print("    manifold_name_and_data,")
    print("    linebundleforHYM,")
    print("    betamodel,")
    print("    phimodel,")
    print("    functionforbaseharmonicform_jbar,")
    print("    sigma_config)")
    
    # Compatibility with original code
    print("\nCompatibility with original code:")
    print("# Using wrapper functions to integrate with existing code")
    print("from network_config import use_with_original_train_and_save_nn_HF")
    print("sigma_config = get_medium_sigma_config('my_sigma')")
    print("sigmamodel, _, _ = use_with_original_train_and_save_nn_HF(")
    print("    train_and_save_nn_HF,")
    print("    manifold_name_and_data,")
    print("    linebundleforHYM,")
    print("    betamodel,")
    print("    phimodel,")
    print("    functionforbaseharmonicform_jbar,")
    print("    sigma_config,")
    print("    use_zero_network=False)")
    
    return "Sigma model examples completed"

def example_full_training_pipeline():
    """Demonstrate a complete training pipeline using the unified configuration."""
    
    print("\n=== COMPLETE TRAINING PIPELINE EXAMPLE ===\n")
    
    print("# This example shows the complete training pipeline from phi to beta to sigma model")
    
    print("\n# Step 1: Define configurations for all models")
    print("phi_config = get_medium_phi_config('phi_main')")
    print("beta_config = get_medium_beta_config('beta_main')")
    print("sigma_config = get_medium_sigma_config('sigma_main')")
    
    print("\n# Step 2: Train or load phi model")
    print("if train_phi:")
    print("    phimodel, phi_history, _ = train_and_save_phi_model(")
    print("        manifold_name_and_data, phi_config)")
    print("else:")
    print("    phimodel, phi_history, _ = load_phi_model(")
    print("        manifold_name_and_data, phi_config)")
    
    print("\n# Step 3: Train or load beta model")
    print("if train_beta:")
    print("    betamodel, beta_history, _ = train_and_save_beta_model(")
    print("        manifold_name_and_data, linebundleforHYM, beta_config, phimodel)")
    print("else:")
    print("    betamodel, beta_history, _ = load_beta_model(")
    print("        manifold_name_and_data, linebundleforHYM, beta_config, phimodel)")
    
    print("\n# Step 4: Train or load sigma model")
    print("if train_sigma:")
    print("    sigmamodel, sigma_history, _ = train_and_save_sigma_model(")
    print("        manifold_name_and_data,")
    print("        linebundleforHYM,")
    print("        betamodel,")
    print("        phimodel,")
    print("        functionforbaseharmonicform_jbar,")
    print("        sigma_config)")
    print("else:")
    print("    sigmamodel, sigma_history, _ = load_sigma_model(")
    print("        manifold_name_and_data,")
    print("        linebundleforHYM,")
    print("        betamodel,")
    print("        phimodel,")
    print("        functionforbaseharmonicform_jbar,")
    print("        sigma_config)")
    
    print("\n# Step 5: Use models for inference or further calculations")
    print("# Now you can use phimodel, betamodel, and sigmamodel for calculations")
    
    return "Full training pipeline example completed"

def main():
    """Demonstrate unified network configuration approach for beta and sigma models."""
    
    print("===== UNIFIED NETWORK CONFIGURATION FOR BETA AND SIGMA MODELS =====")
    
    # Run the beta model examples
    example_beta_model()
    
    # Run the sigma model examples
    example_sigma_model()
    
    # Run the full training pipeline example
    example_full_training_pipeline()
    
    print("\n===== END OF EXAMPLES =====")

if __name__ == "__main__":
    # Set TensorFlow to only use CPU if needed
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Import these only if actually running the examples
    from custom_networks import (
        BiholoModelFuncGENERALforHYMinv3,
        BiholoModelFuncGENERALforSigma2_m13
    )
    
    # Run the demonstration
    main() 