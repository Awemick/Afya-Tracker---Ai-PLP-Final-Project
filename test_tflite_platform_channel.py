#!/usr/bin/env python3
"""
Test script to verify TFLite platform channel functionality
"""

import numpy as np
import tensorflow as tf

def test_tflite_model():
    """Test the TFLite model directly to ensure it works"""
    try:
        print("Testing TFLite model directly...")

        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path="fetal_health_model.tflite")
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")

        # Create test input (21 features)
        test_input = np.array([[
            140.0,  # baseline value
            0.1,    # accelerations
            5.0,    # fetal_movement
            0.5,    # uterine_contractions
            0.0,    # light_decelerations
            0.0,    # severe_decelerations
            0.0,    # prolongued_decelerations
            20.0,   # abnormal_short_term_variability
            10.0,   # mean_value_of_short_term_variability
            5.0,    # percentage_of_time_with_abnormal_long_term_variability
            15.0,   # mean_value_of_long_term_variability
            50.0,   # histogram_width
            120.0,  # histogram_min
            170.0,  # histogram_max
            3.0,    # histogram_number_of_peaks
            0.0,    # histogram_number_of_zeroes
            140.0,  # histogram_mode
            145.0,  # histogram_mean
            143.0,  # histogram_median
            25.0,   # histogram_variance
            0.0,    # histogram_tendency
        ]], dtype=np.float32)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_input)

        # Run inference
        interpreter.invoke()

        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Raw output: {output_data}")

        # Find predicted class
        predicted_class = np.argmax(output_data[0])
        confidence = output_data[0][predicted_class]

        print("SUCCESS: TFLite model test successful!")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")

        # Class labels: 0=Normal, 1=Suspect, 2=Pathological
        class_names = ["Normal", "Suspect", "Pathological"]
        print(f"Prediction: {class_names[predicted_class]}")

        return True

    except Exception as e:
        print(f"FAILED: TFLite model test failed: {e}")
        return False

if __name__ == "__main__":
    test_tflite_model()