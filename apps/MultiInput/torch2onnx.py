import torch.nn as nn
import torch.onnx


class MultiInputNet(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2


# Create the super-resolution model by using the above model definition.
torch_model = MultiInputNet()
torch_model.eval()

# Input to the model
dummy_x1 = torch.randn(3, 3)
dummy_x2 = torch.randn(3, 3)
torch_out = torch_model(dummy_x1, dummy_x2)

# Export the model
torch.onnx.export(
    torch_model,  # model being run
    (dummy_x1, dummy_x2),  # model input (or a tuple for multiple inputs)
    "multi_input.onnx",  # where to save the model (can be a file or file-like object)
    verbose=True,
    opset_version=11,  # the ONNX version to export the model to
    input_names=["x1", "x2"],  # the model's input names
    output_names=["output"],  # the model's output names
)
