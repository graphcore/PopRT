# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse

import onnx

from onnx import helper

from poprt.profile import get_model_flops


def default_model():
    TensorProto = onnx.TensorProto
    matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
    add = helper.make_node("Add", ["Z", "A"], ["B"])
    graph = helper.make_graph(
        [matmul, add],
        "test",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, (4, 4, 8, 16)),
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (4, 4, 16, 8)),
            helper.make_tensor_value_info("A", TensorProto.FLOAT, (4, 4, 8, 8)),
        ],
        [helper.make_tensor_value_info("B", TensorProto.FLOAT, (4, 4, 8, 8))],
    )
    opset_imports = [helper.make_opsetid("", 11)]
    original_model = helper.make_model(graph, opset_imports=opset_imports)
    return original_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show flops of onnx model.')
    parser.add_argument('--onnx_model', type=str, help="Full path of the onnx model.")
    args = parser.parse_args()

    if not args.onnx_model:
        print("No onnx model provided, run default model.")
        model = default_model()
    else:
        print(f"Run onnx model {args.onnx_model}")
        model = onnx.load(args.onnx_model)

    flops = get_model_flops(model, False)
    print(f"FLOPs of model is {flops}")
