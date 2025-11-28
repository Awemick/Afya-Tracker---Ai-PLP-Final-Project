import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Ai datasets/fetal_health.csv')

# Display basic info
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Target distribution:")
print(df['fetal_health'].value_counts())

# Prepare features and target
X = df.drop('fetal_health', axis=1)
y = df['fetal_health'] - 1  # Convert to 0,1,2 for classification

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5)
    ]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Suspect', 'Pathological'],
            yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# Representative dataset for quantization
def representative_dataset_gen():
    for _ in range(100):
        yield [np.random.rand(1, 21).astype(np.float32)]

# Save the model
model.save('fetal_health_model.h5')

# Convert to TensorFlow Lite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_model_quantized = converter.convert()

# Save the quantized TFLite model
with open('fetal_health_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model_quantized)

# Also save unquantized version
converter_unquantized = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model_unquantized = converter_unquantized.convert()

with open('fetal_health_model.tflite', 'wb') as f:
    f.write(tflite_model_unquantized)

print("Model saved as fetal_health_model.h5")
print(f"TFLite model saved as fetal_health_model.tflite ({len(tflite_model_unquantized)} bytes)")
print(f"Quantized TFLite model saved as fetal_health_model_quantized.tflite ({len(tflite_model_quantized)} bytes)")

# Save scaler for later use in app
import joblib
joblib.dump(scaler, 'scaler.pkl')