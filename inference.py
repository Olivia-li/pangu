import os
import numpy as np
import torch

# The directory of your input and output data
input_data_dir = 'input_data'
output_data_dir = 'output_data'

# Load the PyTorch model
model = torch.load('models/model_1.pth')

# Ensure the model is in evaluation mode
model.eval()

# Load the upper-air numpy arrays
input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)

# Convert the numpy arrays to PyTorch tensors
input = torch.from_numpy(input)
input_surface = torch.from_numpy(input_surface)

# Run the inference session
with torch.no_grad():
    output, output_surface = model(input, input_surface)

# Convert the output tensors to numpy arrays
output = output.numpy()
output_surface = output_surface.numpy()

# Save the results
np.save(os.path.join(output_data_dir, 'output_upper'), output)
np.save(os.path.join(output_data_dir, 'output_surface'), output_surface)