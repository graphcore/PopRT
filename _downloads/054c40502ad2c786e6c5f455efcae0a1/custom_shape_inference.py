# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Tuple

import onnx

from poprt.passes.onnx_helper import get_dtype, get_shape
from poprt.passes.shape_inference import ShapeInference, register


@register(['LeakyRelu'])
class LeakyRelu(ShapeInference):
    """Function based on ONNX to infer the shape and dtype of Custom Op."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        model: onnx.ModelProto,
        node: onnx.NodeProto,
    ) -> Tuple[onnx.ModelProto, bool]:
        graph = model.graph
        input_name = node.input[0]
        output_name = node.output[0]
        # If the Op already has known shape and dtype of output, return True
        if get_shape(model.graph, output_name) and get_dtype(model.graph, output_name):
            return model, True

        input_dtype = get_dtype(graph, input_name)
        input_shape = get_shape(graph, input_name)
        # If the Op is able to be inferred shape and dtype, return True
        if input_dtype and input_shape and 0 not in input_shape:
            # ![Shape-Inference Function begin]

            # Step.1: Write the method following ONNX-Protobuf standard,
            #         to calc shape and dtype of output in terms of shape and dtype of input
            # The LeakyRelu Op has same shape and dtype with input and output

            # Step.2: Create new TensorProto with inferred shape and dtype of output
            output_tensor = onnx.helper.make_tensor_value_info(
                output_name, input_dtype, input_shape
            )
            # Step.3: Call update_value_info to update
            model = self.update_value_info(model, output_tensor)
            # Step.4: Call infer_shapes function
            model = onnx.shape_inference.infer_shapes(model)
            # ![Shape-Inference Function end]
            return model, True
        # If the Op is not able to be inferred, return False
        else:
            return model, False
