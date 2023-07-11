# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import time

from typing import Dict

import numpy as np
import onnx

from onnx import helper

from poprt import Pass, runtime
from poprt.compiler import Compiler, CompilerOptions
from poprt.converter import Converter

RuntimeInput = Dict[str, np.ndarray]


def convert(model_proto: onnx.ModelProto, args) -> onnx.ModelProto:
    """Convert ONNX model to a new optimized ONNX model."""
    converter = Converter(convert_version=11, precision='fp16')
    converted_model = converter.convert(model_proto)
    # Add other passes here
    converted_model = Pass.get_pass('int64_to_int32')(converted_model)
    converted_model = Pass.get_pass('gelu_pattern')(converted_model)

    return converted_model


def compile(model: onnx.ModelProto, args):
    """Compile ONNX to PopEF."""
    model_bytes = model.SerializeToString()
    outputs = [o.name for o in model.graph.output]

    options = CompilerOptions()
    options.num_ipus = 1
    options.ipu_version = runtime.DeviceManager().ipu_hardware_version()
    options.batches_per_step = args.batches_per_step
    options.partials_type = 'half'
    executable = Compiler.compile(model_bytes, outputs, options)

    return executable


def run_synchronous(
    model_runner: runtime.Runner,
    input: RuntimeInput,
    output: RuntimeInput,
    iterations: int,
) -> None:
    deltas = []
    sess_start = time.time()
    for _ in range(iterations):
        start = time.time()
        model_runner.execute(input, output)
        end = time.time()
        deltas.append(end - start)
    sess_end = time.time()

    latency = sum(deltas) / len(deltas) * 1000
    print(f'Latency : {latency:.3f}ms')
    avg_sess_time = (sess_end - sess_start) / iterations * 1000
    print(f'Synchorous avg Session Time : {avg_sess_time:.3f}ms')


def run_asynchronous(
    model_runner: runtime.Runner,
    input: RuntimeInput,
    output: RuntimeInput,
    iterations: int,
) -> None:
    # precreate multiple numbers of outputs
    async_inputs = [input] * iterations
    async_outputs = [output] * iterations
    futures = []

    sess_start = time.time()
    for i in range(iterations):
        f = model_runner.execute_async(async_inputs[i], async_outputs[i])
        futures.append(f)

    # waits all execution ends
    for i, future in enumerate(futures):
        future.wait()
    sess_end = time.time()

    avg_sess_time = (sess_end - sess_start) / iterations * 1000
    print(f'Asyncronous avg Session Time : {avg_sess_time:.3f}ms')


def run(executable, args):
    """Run PopEF."""
    # Create model runner
    model_runner = runtime.Runner(executable)

    # fix random number generation
    np.random.seed(2022)

    # Prepare inputs and outpus
    inputs = {}
    inputs_info = model_runner.get_execute_inputs()
    for input in inputs_info:
        inputs[input.name] = np.random.uniform(0, 1, input.shape).astype(
            input.numpy_data_type()
        )

    outputs = {}
    outputs_info = model_runner.get_execute_outputs()
    for output in outputs_info:
        outputs[output.name] = np.zeros(output.shape, dtype=output.numpy_data_type())

    # Run
    # To correctly generate the popvision report, iteration must be a
    # multiple of batches_per_step and greater than 2 * batches_per_step
    iteration = args.batches_per_step * 10

    # warm up device
    for _ in range(10):
        model_runner.execute(inputs, outputs)

    run_synchronous(model_runner, inputs, outputs, iteration)
    run_asynchronous(model_runner, inputs, outputs, iteration)


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
    parser = argparse.ArgumentParser(
        description='Convert onnx model and run it on IPU.'
    )
    parser.add_argument('--onnx_model', type=str, help="Full path of the onnx model.")
    parser.add_argument(
        '--batches_per_step',
        type=int,
        default=100,
        help="The number of on-chip loop count.",
    )
    parser.add_argument('--popef', type=str, help="Full path of the popef file")
    args = parser.parse_args()

    if args.popef:
        run(args.popef, args)
    else:
        if not args.onnx_model:
            print("No onnx model provided, run default model.")
            model = default_model()
        else:
            print(f"Run onnx model {args.onnx_model}")
            model = onnx.load(args.onnx_model)

        converted_model = convert(model, args)
        executable = compile(converted_model, args)
        run(executable, args)
