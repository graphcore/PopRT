# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os

import onnx

from onnx import TensorProto, helper

from poprt import runtime


def make_single_relu(onnx_file):
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])

    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])

    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        "Relu",  # name
        ["X"],  # inputs
        ["Y"],  # outputs
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],  # nodes
        "single-relu",  # name
        [X],  # inputs
        [Y],  # outputs
    )

    # Create the model (ModelProto)
    opset_imports = [helper.make_opsetid("", 11)]
    model_def = helper.make_model(
        graph_def, producer_name="single-relu", opset_imports=opset_imports
    )

    print(f"The model is:\n{onnx.helper.printable_graph(model_def.graph)}")
    onnx.checker.check_model(model_def)
    print("The model is checked!")
    onnx.save(model_def, onnx_file)

    return model_def


if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))
    model_path = abs_path + "/model"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    onnx_file = str(model_path + "/single_relu.onnx")
    popef_file = str(model_path + "/executable.popef")

    model_proto = make_single_relu(onnx_file)

    ipu_version = runtime.DeviceManager().ipu_hardware_version()
    # Compile onnx to popef from cmd cli
    cmd = rf"""poprt \
--input_model {abs_path}/model/single_relu.onnx \
--output_model {abs_path}/single_relu_export.onnx \
--export_popef \
--output_dir model \
--ipu_version {ipu_version} \
--custom_library_so_paths {abs_path}/custom_transform.so \
--compiler_options custom_transform_applier_settings="{"{"}'Fwd0': ['IrSerialise']{"}"}"
"""
    os.system(cmd)
