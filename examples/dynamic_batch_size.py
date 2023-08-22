# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import numpy.testing as npt
import onnx

from onnx import helper

from poprt import runtime
from poprt.compiler import Compiler
from poprt.runtime import RuntimeConfig


def default_model():
    """Create a test model."""
    TensorProto = onnx.TensorProto
    add = helper.make_node("Add", ["X", "Y"], ["O"])
    graph = helper.make_graph(
        [add],
        "test",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, (4, 2)),
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, (4, 2)),
        ],
        [helper.make_tensor_value_info("O", TensorProto.FLOAT, (4, 2))],
    )
    opset_imports = [helper.make_opsetid("", 11)]
    original_model = helper.make_model(graph, opset_imports=opset_imports)
    return original_model


def compile(model: onnx.ModelProto):
    """Compile ONNX to PopEF."""
    model_bytes = model.SerializeToString()
    outputs = [o.name for o in model.graph.output]
    executable = Compiler.compile(model_bytes, outputs)
    return executable


def run(executable):
    """Run PopEF."""
    config = RuntimeConfig()
    config.timeout_ns = 300 * 1000  # 300us
    config.batching_dim = 0
    model_runner = runtime.Runner(executable, config)
    batch_sizes = [1, 4, 7]
    for batch_size in batch_sizes:
        inputs = {}
        inputs['X'] = np.random.uniform(0, 1, [batch_size, 2]).astype(np.float32)
        inputs['Y'] = np.random.uniform(0, 1, [batch_size, 2]).astype(np.float32)

        outputs = {}
        outputs['O'] = np.zeros([batch_size, 2], dtype=np.float32)
        model_runner.execute(inputs, outputs)
        expected = inputs['X'] + inputs['Y']
        npt.assert_array_equal(
            outputs['O'],
            expected,
            f"Result: outputs['O'] not equal with expected: {expected}",
        )
        print(f'Successfully run with input data in batch size {batch_size}')


if __name__ == '__main__':
    model = default_model()
    executable = compile(model)
    run(executable)
