import torch
import torch.nn as nn
import torch.onnx

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 20), 
            nn.ReLU(),         
            nn.Linear(20, 3)
        )

    def forward(self, x):
        return self.network(x)

model = SimpleModel()
model.eval()

# Dummy input
dummy_input = torch.randn(1, 10)

model_name = "sensor_model.onnx"
torch.onnx.export(
    model,                  
    dummy_input,            # Input shape
    model_name,             # model name
    export_params=True,     
    opset_version=18,       # ONNX version
    input_names=['input'],  
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Success! Model '{model_name}' exported.")