import onnx

model_path = "sensor_model.onnx"
model = onnx.load(model_path)
graph = model.graph

print(f"--- Inspect ONNX model ---")

for input in graph.input:
    shape = []
    for dim in input.type.tensor_type.shape.dim:
        val = dim.dim_value if dim.dim_value > 0 else "Dynamic"
        shape.append(val)
    print(f"input: name='{input.name}', shape={shape}")

for output in graph.output:
    shape = []
    for dim in output.type.tensor_type.shape.dim:
        val = dim.dim_value if dim.dim_value > 0 else "Dynamic"
        shape.append(val)
    print(f"Output: name='{output.name}', shape={shape}")