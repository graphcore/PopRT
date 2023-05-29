# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os

from onnx import TensorProto, helper

from poprt import compiler, runtime


def make_matmul_with_tile_mapping():
    matmul = helper.make_node("MatMul", ["X", "Y"], ["Z"])
    mapping = helper.make_node(
        "PrintTileMapping", ["Z"], ["Out"], domain="ai.graphcore"
    )
    graph = helper.make_graph(
        [matmul, mapping],
        "test",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, (4, 4, 8, 8)),
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (4, 4, 8, 8)),
        ],
        [helper.make_tensor_value_info("Out", TensorProto.FLOAT, (4, 4, 8, 8))],
    )
    opset_imports = [helper.make_opsetid("", 11)]
    model = helper.make_model(graph, opset_imports=opset_imports)
    return model


if __name__ == '__main__':
    popef_file = "./test.popef"

    model_proto = make_matmul_with_tile_mapping()
    model = model_proto.SerializeToString()
    outputs = [o.name for o in model_proto.graph.output]
    opts = compiler.CompilerOptions()
    opts.ipu_version = runtime.DeviceManager().ipu_hardware_version()
    compiler.Compiler.compile_and_export(model, outputs, popef_file, opts)
    os.remove(popef_file)
