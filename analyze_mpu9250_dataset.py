#!/usr/bin/env python3
"""
Analyze the MPU9250 Fetal Movement Detection Dataset
This dataset is much more relevant for mobile app deployment
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import os

def load_mpu9250_data():
    """Load all MPU9250 .mat files and combine into a single dataset"""

    data_dir = 'Ai datasets/Fetal Movement Detection Dataset Recorded Using MPU9250 Tri-Axial Accelerometer/Fetal Movement  Detection Dataset Recorded Using MPU9250 Tri-Axial Accelerometer'

    all_data = []

    print("Loading MPU9250 dataset files...")

    for i in range(1, 14):  # Mom_1.mat to Mom_13.mat
        filename = f'Mom_{i}.mat'
        filepath = os.path.join(data_dir, filename)

        try:
            mat_data = scipy.io.loadmat(filepath)
            print(f"Loaded {filename}: {list(mat_data.keys())}")

            # The .mat files appear to be in MATLAB table format
            # Let's try to extract the data differently

            # Look for the main data structure
            main_key = None
            for key in mat_data.keys():
                if not key.startswith('__'):
                    main_key = key
                    break

            if main_key:
                data_structure = mat_data[main_key]
                print(f"  Main data structure: {type(data_structure)}")
                print(f"  Data structure shape: {data_structure.shape}")

                # If it's a structured array with table data
                if hasattr(data_structure, 'dtype') and data_structure.dtype.names:
                    print(f"  Structured array with fields: {data_structure.dtype.names}")

                    # Try to extract as table
                    try:
                        # Convert structured array to DataFrame
                        df = pd.DataFrame.from_records(data_structure)
                        print(f"  Converted to DataFrame with shape: {df.shape}")
                        print(f"  Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns

                        # Look for the expected columns
                        expected_cols = ['Time', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Laugh', 'Move']
                        available_cols = [col for col in expected_cols if col in df.columns]

                        if len(available_cols) >= 6:  # At least accelerometer data
                            df_subset = df[available_cols].copy()
                            df_subset['Mother_ID'] = i
                            all_data.append(df_subset)
                            print(f"  Successfully extracted {len(available_cols)} columns")
                        else:
                            print(f"  Not enough expected columns found. Available: {df.columns.tolist()}")

                    except Exception as e2:
                        print(f"  Error converting to DataFrame: {e2}")

                else:
                    print(f"  Unexpected data structure format")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nCombined dataset shape: {combined_df.shape}")
        print(f"Columns: {combined_df.columns.tolist()}")

        # Check for movement data
        if 'Move' in combined_df.columns:
            print(f"Value counts for Move: {combined_df['Move'].value_counts()}")
        if 'Laugh' in combined_df.columns:
            print(f"Value counts for Laugh: {combined_df['Laugh'].value_counts()}")

        return combined_df
    else:
        print("No data could be loaded from the .mat files")
        return None

def analyze_dataset(df):
    """Analyze the loaded dataset"""

    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)

    print(f"Total samples: {len(df)}")
    print(f"Mothers included: {df['Mother_ID'].nunique()}")
    print(f"Sampling rate: 280 Hz (based on README)")

    # Check for missing values
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Analyze class distribution
    print("\nClass Distribution:")
    print(f"Fetal Movement (Move=1): {df['Move'].sum()} samples ({df['Move'].sum()/len(df)*100:.1f}%)")
    print(f"Mother's Laugh (Laugh=1): {df['Laugh'].sum()} samples ({df['Laugh'].sum()/len(df)*100:.1f}%)")

    # Calculate movement frequency per mother
    movements_per_mother = df.groupby('Mother_ID')['Move'].sum()
    print(f"\nFetal movements per mother:\n{movements_per_mother}")

    # Analyze sensor data ranges
    sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    print(f"\nSensor data ranges:")
    for col in sensor_cols:
        print(".2f")

    # Check for correlations
    corr_matrix = df[sensor_cols + ['Move', 'Laugh']].corr()
    print(f"\nCorrelation with fetal movement (Move):")
    print(corr_matrix['Move'].sort_values(ascending=False))

def create_movement_detection_model(df):
    """Create a model to detect fetal movements from accelerometer data"""

    print("\n" + "="*50)
    print("BUILDING FETAL MOVEMENT DETECTION MODEL")
    print("="*50)

    # Prepare features and target
    feature_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    X = df[feature_cols].values
    y = df['Move'].values  # 1 for fetal movement, 0 otherwise

    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)} (0: no movement, 1: fetal movement)")

    # Handle class imbalance - fetal movements are rare events
    movement_samples = np.sum(y)
    total_samples = len(y)
    imbalance_ratio = (total_samples - movement_samples) / movement_samples
    print(".2f")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a model suitable for movement detection
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Use weighted loss due to class imbalance
    class_weight = {0: 1.0, 1: imbalance_ratio}

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        ]
    )

    # Evaluate the model
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test_scaled, y_test)
    print(".4f")
    print(".4f")
    print(".4f")

    # Predictions
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Movement', 'Fetal Movement']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Movement', 'Fetal Movement'],
                yticklabels=['No Movement', 'Fetal Movement'])
    plt.title('Confusion Matrix - Fetal Movement Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('mpu9250_confusion_matrix.png')
    plt.show()

    # Save the model
    model.save('fetal_movement_model.h5')

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Representative dataset for quantization
    def representative_dataset_gen():
        for _ in range(100):
            yield [np.random.rand(1, 6).astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()

    with open('fetal_movement_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"\nModel saved as fetal_movement_model.h5")
    print(f"TFLite model saved as fetal_movement_model.tflite ({len(tflite_model)} bytes)")

    # Save scaler for app use
    import joblib
    joblib.dump(scaler, 'mpu9250_scaler.pkl')

    return model, scaler, history

def compare_datasets():
    """Compare the MPU9250 dataset with the current CTG dataset"""

    print("\n" + "="*50)
    print("DATASET COMPARISON")
    print("="*50)

    print("Current CTG Dataset (fetal_health.csv):")
    ctg_df = pd.read_csv('Ai datasets/fetal_health.csv')
    print(f"  - Samples: {len(ctg_df)}")
    print(f"  - Features: {ctg_df.shape[1] - 1} (CTG measurements)")
    print(f"  - Classes: {ctg_df['fetal_health'].nunique()} (Normal/Suspect/Pathological)")
    print(f"  - Equipment needed: Expensive CTG machine")

    print("\nMPU9250 Accelerometer Dataset:")
    print("  - Samples: ~13 mothers × 10-20 minutes × 280Hz")
    print("  - Features: 6 (3-axis accelerometer + 3-axis gyroscope)")
    print("  - Classes: 2 (Fetal movement detection)")
    print("  - Equipment needed: Smartphone/wearable device")

    print("\nRelevance to Mobile App:")
    print("  ✅ MPU9250: Perfect for mobile - uses phone sensors")
    print("  ❌ CTG: Requires medical equipment - not mobile-friendly")
    print("  ✅ MPU9250: Real-time fetal movement tracking")
    print("  ❌ CTG: Retrospective analysis only")

if __name__ == "__main__":
    # Load and analyze the MPU9250 dataset
    df = load_mpu9250_data()

    if df is not None:
        # Analyze the dataset
        analyze_dataset(df)

        # Compare with current dataset
        compare_datasets()

        # Create movement detection model
        model, scaler, history = create_movement_detection_model(df)

        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print("*** REPLACE the CTG-based model with MPU9250-based fetal movement detection!")
        print("*** This will make your app truly mobile-friendly and practical for users.")
        print("*** The new model can run on smartphone sensors in real-time.")
        print("="*60)

    else:
        print("*** Failed to load MPU9250 dataset. Check file formats and paths.")