# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import sys

import clip
import numpy as np
import onnx
import torch

import poprt

sys.path.append('../')
import helper

# Load CLIP
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
input_data = (torch.ones(1, 3, 224, 224).float(), torch.ones(100, 77).long())
print("Export ONNX model with dynamic batch size:")
torch.onnx.export(
    model,
    input_data,
    "./CLIP.onnx",
    export_params=True,
    opset_version=11,
    input_names=["input", "input/1"],
    output_names=["output", "output/1"],
    dynamic_axes={'input': {0: 'batch_size'}},
)
print("Check precision between Pytorch and ONNX:")
# Generate input data
input_info = {
    "input": ([1, 3, 224, 224], np.float32),
    "input/1": ([100, 77], np.int64),
}
input_data = helper.generate_data(input_info)

# Run Torch
with torch.no_grad():
    image = torch.from_numpy(input_data["input"]).float()
    text = torch.from_numpy(input_data["input/1"]).long()
    logits_per_image, logits_per_text = model(image, text)
    torch_res = [logits_per_image.detach().numpy(), logits_per_text.detach().numpy()]

model = onnx.load("CLIP.onnx")
# ONNXRUNTIME CPU only support ArgMax with INT32, but the original model uses INT64
model.graph.input[1].type.tensor_type.elem_type = onnx.TensorProto.INT32
# Overwrite ONNX model
onnx.save(model, "CLIP.onnx")

input_data["input/1"] = input_data["input/1"].astype(np.int32)
# Run ONNXRUNTIME
sess = poprt.backend.get_session(model.SerializeToString(), 1, "onnxruntime")
sess.load()
inputs_info, outputs_info = sess.get_io_info()
outputs_name = [o for o in outputs_info]
ort_res = sess.run(outputs_name, input_data)
helper.accuracy(torch_res, ort_res)
