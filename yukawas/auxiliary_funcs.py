import gc
import tensorflow as tf
import time
from datetime import datetime

def delete_all_dicts_except(except_dict_name):
    """
    Deletes all dictionary objects from the global namespace except the specified one.

    Parameters:
    except_dict_name (str): The name of the dictionary to exclude from deletion.
    """

    # Identify all dictionary variables except the one specified
    dicts_to_delete = [
        var for var, obj in globals().items()
        if isinstance(obj, dict) and var != except_dict_name
    ]

    # Delete all identified dictionary variables
    for var in dicts_to_delete:
        del globals()[var]

    # Optionally, you can clear memory by calling garbage collector
    gc.collect()



def batch_process_helper_func(func, args, batch_indices=(0,), batch_size=10000, compile_func=False):
    # Optionally compile the function for better performance
    if compile_func:
        func = tf.function(func)
        
    # Determine the number of batches based on the first batched argument
    num_batches = tf.math.ceil(tf.shape(args[batch_indices[0]])[0] / batch_size)
    results_list = []

    for i in range(tf.cast(num_batches, tf.int32)):
        print(i)
        start_idx = i * batch_size
        end_idx = tf.minimum((i + 1) * batch_size, tf.shape(args[batch_indices[0]])[0])
        
        # Create batched arguments
        batched_args = list(args)
        for idx in batch_indices:
            batched_args[idx] = args[idx][start_idx:end_idx]
        
        # Call the function with batched and static arguments
        batch_results = func(*batched_args)
        results_list.append(batch_results)

    return tf.concat(results_list, axis=0)


def determine_training_flags(start_from='phi'):
    training_flags = {
        'phi': True,
        'LB1': True,  # 02m20
        'LB2': True,  # 001m3
        'LB3': True,  # 0m213
        'vH': True,
        'vQ3': True,
        'vU3': True,
        'vQ1': True,
        'vQ2': True,
        'vU1': True,
        'vU2': True,
        'end': True
    }
    
    if start_from not in training_flags:
        raise ValueError(f"Error: '{start_from}' is not a valid starting point.")
    
    start_index = list(training_flags.keys()).index(start_from)
    print(f"Starting training from: {start_from}")
    
    for key in list(training_flags.keys())[:start_index]:
        training_flags[key] = False
        print(f"Skipping: {key}")
    
    return training_flags
import sys
    
def print_memory_usage(start_time_of_process=None, name = None):
    """
    Prints the current memory usage of the Python process.
    """
    import os
    import psutil
    
    # Get the current process
    process = psutil.Process(os.getpid())
    
    # Get memory info in bytes
    memory_info = process.memory_info()
    
    # Convert to more readable format (MB)
    memory_usage_mb = memory_info.rss / 1024 / 1024
    
    
    # Additional memory details
    virtual_memory = psutil.virtual_memory()
    if start_time_of_process is not None:
        
        start_time_str = datetime.fromtimestamp(start_time_of_process).strftime('%H:%M:%S')
        printafter_time = time.time() - start_time_of_process
    else:
        start_time_str = "unknown"
        printafter_time = 0.0
    
    print(f"Current memory usage of process {os.getpid()} for {name} started at {start_time_str}: {memory_usage_mb:.2f} MB",f", System memory: {virtual_memory.total / (1024 * 1024 * 1024):.2f} GB total, "
      f"{virtual_memory.available / (1024 * 1024 * 1024):.2f} GB available. (printed after {printafter_time:.2f} sec)")

