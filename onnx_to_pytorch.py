import onnx
import torch
from onnx2pytorch import ConvertModel

onnx_model = onnx.load("models/model_1.onnx")
pytorch_model = ConvertModel(onnx_model)

# Save the converted model
torch.save(pytorch_model.state_dict(), "model_1.pth")

