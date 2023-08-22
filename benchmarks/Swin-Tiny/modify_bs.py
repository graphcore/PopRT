# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import copy

import onnx

from poprt.optimizer import Optimizer
from poprt.passes.onnx_helper import get_constant

model = onnx.load('open_swim_transformer/swim-transformer-batch2.onnx')

# Modify the input batch size
model = Optimizer(["extract_constant_to_initializer"]).run(model)
model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 24
model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 24


# Fix Reshape Ops
for n in model.graph.node:
    if n.op_type == "Reshape":
        shape = get_constant(model.graph, n.input[1])
        if shape is not None:
            shape_array = copy.deepcopy(onnx.numpy_helper.to_array(shape))
            if shape_array[0] == 4:
                shape_array[0] = 24
                shape.CopyFrom(
                    onnx.numpy_helper.from_array(shape_array, name=shape.name)
                )
            if shape_array[0] == 256:
                shape_array[0] = 1536
                shape.CopyFrom(
                    onnx.numpy_helper.from_array(shape_array, name=shape.name)
                )
            if shape_array[0] == 64:
                shape_array[0] = 384
                shape.CopyFrom(
                    onnx.numpy_helper.from_array(shape_array, name=shape.name)
                )
            if shape_array[0] == 16:
                shape_array[0] = 96
                shape.CopyFrom(
                    onnx.numpy_helper.from_array(shape_array, name=shape.name)
                )

model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, "open_swin-Tiny-BS24-fp32.onnx")
