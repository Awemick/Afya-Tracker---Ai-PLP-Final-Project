const tf = require('@tensorflow/tfjs-node');
const path = require('path');

async function testFetalHealthModel() {
  try {
    // Load the model from file system
    const modelPath = path.join(__dirname, '../afya_tracker_web/public/models/fetal_health_model/model.json');
    const model = await tf.loadLayersModel(`file://${modelPath}`);

    console.log('Model loaded successfully');

    // Test data - 21 features as expected by the model
    const testInputs = [
      // Normal case
      [
        140.0, 0.1, 0.8, 0.5, 0.0, 0.0, 0.0, 20.0, 8.0, 5.0, 12.0,
        50.0, 120.0, 170.0, 3.0, 0.0, 140.0, 145.0, 143.0, 25.0, 0.0
      ],
      // Suspect case
      [
        140.0, 0.0, 0.3, 0.5, 0.1, 0.0, 0.0, 50.0, 3.0, 30.0, 4.5,
        50.0, 120.0, 170.0, 3.0, 0.0, 140.0, 145.0, 143.0, 25.0, 0.0
      ],
      // Pathological case
      [
        140.0, 0.0, 0.1, 0.5, 0.2, 0.1, 0.1, 80.0, 1.0, 60.0, 1.5,
        50.0, 120.0, 170.0, 3.0, 0.0, 140.0, 145.0, 143.0, 25.0, 0.0
      ]
    ];

    const classLabels = ['Normal', 'Suspect', 'Pathological'];

    for (let i = 0; i < testInputs.length; i++) {
      const inputTensor = tf.tensor2d([testInputs[i]], [1, 21]);
      const prediction = model.predict(inputTensor);
      const probabilities = await prediction.data();

      const predictedClass = tf.argMax(probabilities).dataSync()[0];
      const confidence = probabilities[predictedClass];

      console.log(`\nTest Case ${i + 1}:`);
      console.log(`Predicted Class: ${predictedClass} (${classLabels[predictedClass]})`);
      console.log(`Confidence: ${(confidence * 100).toFixed(2)}%`);
      console.log(`Probabilities: [${probabilities.map(p => (p * 100).toFixed(1) + '%').join(', ')}]`);

      // Cleanup
      inputTensor.dispose();
      prediction.dispose();
    }

    model.dispose();
    console.log('\nModel testing completed');

  } catch (error) {
    console.error('Error testing model:', error);
  }
}

testFetalHealthModel();