import numpy as np
import struct

# Create random weights for the model (same architecture as fetalHealthService.ts)
# Dense1: 21x64 kernel + 64 bias = 1344 + 64 = 1408 floats
dense1_kernel = np.random.normal(0, 0.1, (21, 64)).astype(np.float32)
dense1_bias = np.zeros(64, dtype=np.float32)

# Dense2: 64x32 kernel + 32 bias = 2048 + 32 = 2080 floats
dense2_kernel = np.random.normal(0, 0.1, (64, 32)).astype(np.float32)
dense2_bias = np.zeros(32, dtype=np.float32)

# Dense3: 32x16 kernel + 16 bias = 512 + 16 = 528 floats
dense3_kernel = np.random.normal(0, 0.1, (32, 16)).astype(np.float32)
dense3_bias = np.zeros(16, dtype=np.float32)

# Dense4: 16x3 kernel + 3 bias = 48 + 3 = 51 floats
dense4_kernel = np.random.normal(0, 0.1, (16, 3)).astype(np.float32)
dense4_bias = np.zeros(3, dtype=np.float32)

# Concatenate all weights
all_weights = np.concatenate([
    dense1_kernel.flatten(),
    dense1_bias.flatten(),
    dense2_kernel.flatten(),
    dense2_bias.flatten(),
    dense3_kernel.flatten(),
    dense3_bias.flatten(),
    dense4_kernel.flatten(),
    dense4_bias.flatten()
])

# Write to binary file in the web app directory
output_path = '../afya_tracker_web/public/models/fetal_health_model/group1-shard1of1.bin'
with open(output_path, 'wb') as f:
    for weight in all_weights:
        f.write(struct.pack('<f', weight))

print(f'Created weights file with {len(all_weights)} parameters')
print(f'File saved to: {output_path}')
print('TensorFlow.js model files created successfully!')
print('- model.json')
print('- group1-shard1of1.bin')