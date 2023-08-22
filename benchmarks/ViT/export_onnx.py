# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import sys

import numpy as np
import torch
import transformers

import poprt

sys.path.append('../')
import helper

# Load ViT
model = transformers.ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224'
)
model.eval()

# Export ONNX model
print("Export ONNX model with dynamic batch size:")
input_data = torch.ones(1, 3, 224, 224).float()
torch.onnx.export(
    model,
    input_data,
    "./ViT.onnx",
    export_params=True,
    opset_version=11,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={'pixel_values': {0: 'batch_size'}},
)

print("Check precision between Pytorch and ONNX:")
# Generate input data
input_info = {
    "pixel_values": ([1, 3, 224, 224], np.float32),
}
input_data = helper.generate_data(input_info)

# Run Torch
with torch.no_grad():
    torch_data = {"pixel_values": torch.from_numpy(input_data["pixel_values"]).float()}
    torch_res = [model(**torch_data).logits.numpy()]

# Run ONNXRUNTIME
sess = poprt.backend.get_session("ViT.onnx", 1, "onnxruntime")
sess.load()
inputs_info, outputs_info = sess.get_io_info()
outputs_name = [o for o in outputs_info]
ort_res = sess.run(outputs_name, input_data)
helper.accuracy(torch_res, ort_res)
