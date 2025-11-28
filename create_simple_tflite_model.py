#!/usr/bin/env python3
"""
Create a simple TFLite model for fetal movement detection
This creates a basic model that can be used in the Flutter app
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_simple_fetal_movement_model():
    """Create a simple neural network for fetal movement detection"""

    # Create a simple model
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(6,)),  # 6 features: ax,ay,az,gx,gy,gz
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Create dummy training data for demonstration
    # In a real scenario, this would be trained on the MPU9250 dataset
    X_train = np.random.randn(1000, 6).astype(np.float32)
    y_train = np.random.randint(0, 2, 1000).astype(np.float32)

    # Train briefly (this is just for model structure)
    model.fit(X_train, y_train, epochs=5, verbose=0)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for quantization
    def representative_dataset_gen():
        for _ in range(100):
            yield [np.random.randn(1, 6).astype(np.float32)]

    converter.representative_dataset = representative_dataset_gen

    # Convert
    tflite_model = converter.convert()

    # Save the model
    with open('fetal_movement_mobile_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"Model created and saved as fetal_movement_mobile_model.tflite")
    print(f"Model size: {len(tflite_model)} bytes")

    return tflite_model

if __name__ == "__main__":
    create_simple_fetal_movement_model()
    print("Simple fetal movement detection model created!")
    print("Note: This is a basic model for demonstration.")
    print("For production, train on the actual MPU9250 dataset.")