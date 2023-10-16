import onnx
import torch
import numpy as np

def LoadData(step):
    """
    Loads the input, input_surface, target, and target_surface data for a given step.
    """
    # Replace these file paths with the actual paths to your .pth files
    input_file = f"data/input_{step}.pth"
    input_surface_file = f"data/input_surface_{step}.pth"
    target_file = f"data/target_{step}.pth"
    target_surface_file = f"data/target_surface_{step}.pth"

    # Load the data from the .pth files
    input = torch.load(input_file)
    input_surface = torch.load(input_surface_file)
    target = torch.load(target_file)
    target_surface = torch.load(target_surface_file)

    return input, input_surface, target, target_surface

def LoadConstantMask(filename):
    """
    Loads a constant mask (e.g., soil type) from an ONNX file.
    """
    onnx_model = onnx.load(filename)
    # Assuming the mask is stored as a parameter of the ONNX model
    mask = onnx_model.graph.initializer[0]
    mask = torch.from_numpy(np.array(mask.float_data)).reshape(mask.dims)
    return mask