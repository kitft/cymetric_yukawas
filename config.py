import tensorflow as tf
#duplicate of cymetric/config.py
# not sure if we need both
import numpy as np

# Global dtype variables
real_dtype = tf.float32
complex_dtype = tf.complex64

def set_double_precision(use_double):
    global real_dtype, complex_dtype
    if use_double:
        real_dtype = tf.float64
        complex_dtype = tf.complex128
    else:
        real_dtype = tf.float32
        complex_dtype = tf.complex64

    # Optional: Set TensorFlow's default float type as well
    tf.keras.backend.set_floatx('float64' if use_double else 'float32')
    tf.keras.mixed_precision.set_global_policy('float64' if use_double else 'float32')
    global_policy = tf.keras.mixed_precision.global_policy()
    floatx = tf.keras.backend.floatx()
    if use_double:
        print(f"successfully set double precision? {'64'in global_policy.name and '64'in floatx }: {global_policy.name} and {floatx}")    
    else:
        print(f"successfully set single precision? {'32'in global_policy.name and '32'in floatx }: {global_policy.name} and {floatx}")    

def convert_to_tensor_dict(data):
   return {
    key: tf.convert_to_tensor(value, dtype=complex_dtype) if value.dtype in [np.complex64, np.complex128] 
         else tf.convert_to_tensor(value, dtype=real_dtype) if value.dtype in [np.float32, np.float64]
         else tf.convert_to_tensor(value, dtype=value.dtype) if value.dtype == np.int32
         else value
    for key, value in data.items()
   }