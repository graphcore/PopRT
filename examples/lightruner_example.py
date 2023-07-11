# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Dict, List

import numpy as np
import onnx

from onnx import helper

from poprt import runtime
from poprt.compiler import Compiler, CompilerOptions

np.random.seed(1984)


def get_weights_info(infos: List[runtime.DataDesc]) -> Dict[str, runtime.DataDesc]:
    weights_info = {}
    for info in infos:
        if info.popef_contains_tensor_data:
            weights_info[info.name] = info

    return weights_info


def compile(model: onnx.ModelProto):
    """Compile ONNX to PopEF."""
    outputs = [o.name for o in model.graph.output]

    options = CompilerOptions()
    options.constant_weights = False
    options.enable_engine_caching = True
    options.cache_path = './cache'

    executable = Compiler.compile(model, outputs, options)
    return executable


def run(executable):
    """Run PopEF."""
    # get device
    dm = runtime.DeviceManager()
    replica_to_device = runtime.ReplicaIdToDevice()
    for device_id in range(dm.get_num_devices()):
        try:
            device = dm.get_specific_device(device_id)
            replica_to_device[0] = device
            break
        except:
            pass
    if not replica_to_device:
        raise RuntimeError("No IPU can be attached.")

    config = runtime.LightRunnerConfig()
    config.replica_to_device = replica_to_device

    runner = runtime.LightRunner(executable, config)

    # generate random inputs
    inputs = {}
    outputs = {}
    for info in runner.get_execute_inputs():
        inputs[info.name] = np.random.uniform(0, 1, info.shape).astype(
            info.numpy_data_type()
        )
    for info in runner.get_execute_outputs():
        outputs[info.name] = np.zeros(info.shape, dtype=info.numpy_data_type())

    # run with original weights(from popef)
    future = runner.execute_async(inputs, outputs)
    future.wait()
    print(outputs["output"].flatten()[1000:1020])

    # generate random weights
    weights_info = get_weights_info(runner.get_model_inputs())
    weights = {}
    for name, info in weights_info.items():
        weights[name] = np.random.uniform(-1, 1, info.shape).astype(
            info.numpy_data_type()
        )

    # update weights
    for k, v in weights.items():
        runner.set_input_anchor(k, v)
    runner.update_weights()
    print(f"fin update_weights")

    # run with updated weights
    future = runner.execute_async(inputs, outputs)
    future.wait()
    print(outputs["output"].flatten()[1000:1020])


def default_model():
    TensorProto = onnx.TensorProto

    nodes = []
    num_matmuls = 4
    nodes.append(helper.make_node("Expand", ["input", "shape"], ["Act0"]))
    for i in range(num_matmuls):
        nodes.append(helper.make_node("MatMul", [f"Act{i}", "Weight"], [f"Act{i+1}"]))
    nodes.append(
        helper.make_node("ReduceMean", [f"Act{num_matmuls}"], ["output"], axes=[0, 1])
    )

    graph = helper.make_graph(
        nodes,
        "matmul_test",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, (256, 256)),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, (256, 256))],
        [
            helper.make_tensor(
                "shape",
                TensorProto.INT64,
                [4],
                np.array([4, 4, 256, 256], dtype=np.int64),
            ),
            helper.make_tensor(
                "Weight",
                TensorProto.FLOAT,
                (4, 4, 256, 256),
                np.random.randn(4, 4, 256, 256),
            ),
        ],
    )
    opset_imports = [helper.make_opsetid("", 11)]
    original_model = helper.make_model(graph, opset_imports=opset_imports)
    return original_model


if __name__ == '__main__':
    model = default_model()
    exe = compile(model)
    run(exe)
