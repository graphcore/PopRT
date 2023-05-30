# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import json
import os

import numpy as np
import numpy.testing as npt
import onnx

from onnx import TensorProto, helper

from poprt import runtime


def make_single_relu(onnx_file):
    # Create one input (ValueInfoProto)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])

    # Create one output (ValueInfoProto)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])

    # Create a node (NodeProto)
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
    onnx.save(model_def, onnx_file)

    return model_def


def check_ir(serilaized_ir_dest):
    with open(serilaized_ir_dest, "r") as f:
        ir = json.load(f)
        assert (
            len(
                list(
                    filter(
                        lambda op: "Neg" in op["type"],
                        ir["single-relu"],
                    )
                )
            )
            == 1
        )


if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))
    model_path = abs_path + "/model"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    onnx_file = str(model_path + "/single_relu.onnx")
    popef_file = str(model_path + "/executable.popef")

    ir_path = abs_path + "/ir"
    if not os.path.exists(ir_path):
        os.mkdir(ir_path)
    ir_file = str(ir_path + "/ir.json")

    model_proto = make_single_relu(onnx_file)

    ipu_version = runtime.DeviceManager().ipu_hardware_version()
    # Compile onnx to popef from cmd cli
    cmd = rf"""poprt \
--input_model {abs_path}/model/single_relu.onnx \
--output_model {abs_path}/single_relu_export.onnx \
--export_popef \
--output_dir model \
--ipu_version {ipu_version} \
--custom_library_so_paths {abs_path}/custom_pattern.so \
--compiler_options custom_patterns=ReplaceReluWithNeg \
serialize_ir=True \
serialized_ir_dest={abs_path}/ir/ir.json"""

    os.system(cmd)

    # check ir
    check_ir(ir_file)

    # Create model runner
    model_runner = runtime.ModelRunner(popef_file)

    # Check results
    data = np.random.uniform(0, 1, (3, 2)).astype(np.float32)
    result = np.zeros([3, 2], dtype=np.float32)

    model_runner.execute({'X': data}, {'Y': result})
    npt.assert_allclose(np.negative(data), result, rtol=1e-6, atol=1e-6)
