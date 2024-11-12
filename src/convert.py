# import tensorflow as tf
#
# # Load the TensorFlow SavedModel
# converter = tf.lite.TFLiteConverter.from_saved_model('uecfood_model_tf')
#
# # Optional: Enable optimizations if you need a smaller, faster model
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
# # Convert the model to TensorFlow Lite
# tflite_model = converter.convert()
#
# # Save the TFLite model to a file
# with open('uecfood_model.tflite', 'wb') as f:
#     f.write(tflite_model)

import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="uecfood_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a dummy input tensor with the correct input shape
input_shape = input_details[0]['shape']
input_data = np.random.rand(*input_shape).astype(np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"Model output: {output_data}")