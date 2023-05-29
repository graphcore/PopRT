# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import os

import onnx

from onnx import helper


def create_onnx_model_with_custom_op():
    TensorProto = onnx.TensorProto

    attributes = {"alpha": 0.01}
    leaky_relu = helper.make_node(
        "LeakyRelu", ["X"], ["Y"], domain="ai.graphcore", **attributes
    )
    relu = helper.make_node("Relu", ["Y"], ["Z"])

    graph = helper.make_graph(
        [leaky_relu, relu],
        "custom_op_test",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, (8, 8)),
        ],
        [
            helper.make_tensor_value_info("Z", TensorProto.FLOAT, (8, 8)),
        ],
    )
    opset_imports = [helper.make_opsetid("", 11)]
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.opset_import.append(onnx.helper.make_opsetid("ai.graphcore", 1))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert onnx model and run it on IPU.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./',
        help="Full path of the onnx model will be saved to.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise ValueError("--output_dir should be an exist folder")

    model_path = os.path.join(args.output_dir, 'custom_op_test.onnx')

    model = create_onnx_model_with_custom_op()
    onnx.save(model, model_path)

    # Convert and Run
    compile_cmd = "bash build.sh"
    os.system(compile_cmd)
    abs_path = os.path.abspath(os.path.dirname(__file__))
    run_cmd = rf"""poprt \
--input_model {model_path} \
--custom_shape_inference {abs_path}/custom_shape_inference.py \
--custom_library_so_paths {abs_path}/custom_ops.so \
--run"""
    os.system(run_cmd)
    # 2022-12-30 07:01:54,408 INFO cli.py:446] Bs: 8
    # 2022-12-30 07:01:54,408 INFO cli.py:449] Latency: 0.23ms
    # 2022-12-30 07:01:54,408 INFO cli.py:450] Tput: 35469
