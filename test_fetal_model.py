#!/usr/bin/env python3
"""
Test script to verify the fetal health TFLite model works correctly
"""

import numpy as np
import tensorflow as tf

def test_tflite_model():
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path='../afya_tracker/assets/models/fetal_health_model.tflite')
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Model loaded successfully!")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Create test input (21 features as expected by the model)
    test_input = np.random.rand(1, 21).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], test_input)

    # Run inference
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Model output: {output}")
    print(f"Predicted class: {np.argmax(output)}")
    print(f"Confidence: {np.max(output)}")

    return True

if __name__ == "__main__":
    try:
        test_tflite_model()
        print("✅ TFLite model test passed!")
    except Exception as e:
        print(f"❌ TFLite model test failed: {e}")
        import traceback
        traceback.print_exc()