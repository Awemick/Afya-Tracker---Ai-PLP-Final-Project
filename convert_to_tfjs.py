import tensorflow as tf
import os

# Load the Keras model
print("Loading Keras model...")
model = tf.keras.models.load_model('fetal_health_model.h5')

# Create tfjs_model directory if it doesn't exist
if not os.path.exists('tfjs_model'):
    os.makedirs('tfjs_model')

# Save as SavedModel format first (more compatible)
saved_model_dir = 'saved_model'
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

print("Saving as SavedModel format...")
tf.saved_model.save(model, saved_model_dir)

# Try to convert using the CLI approach
print("Converting to TensorFlow.js format...")
try:
    # Use the command line tool if available
    import subprocess
    result = subprocess.run([
        'python', '-m', 'tensorflowjs.converters.converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--saved_model_tags=serve',
        f'--input_path={saved_model_dir}',
        '--output_path=tfjs_model'
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Model converted successfully to TensorFlow.js format!")
        print("📁 Saved in: tfjs_model/ directory")
    else:
        print("❌ CLI conversion failed, trying alternative method...")
        raise Exception(result.stderr)

except Exception as e:
    print(f"CLI conversion failed: {e}")
    print("Trying direct Python conversion...")

    try:
        # Fallback: try direct conversion
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, 'tfjs_model')
        print("✅ Model converted successfully using direct method!")
        print("📁 Saved in: tfjs_model/ directory")
    except Exception as e2:
        print(f"❌ Direct conversion also failed: {e2}")
        print("💡 You may need to manually convert using:")
        print("   pip install tensorflowjs")
        print("   tensorflowjs_converter --input_format=keras fetal_health_model.h5 tfjs_model")

print("🎉 Conversion process completed!")