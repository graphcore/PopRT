# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import threading

import numpy as np
import onnx

from onnx import helper

from poprt import runtime
from poprt.compiler import Compiler, CompilerOptions
from poprt.runtime import RuntimeConfig

'''
PopRT use OverlapInnerLoop strategy as default exchange strategy.
There are two loops in the main program: outer loop and inner loop.
Each batch data needs to be processed in three pipeline stages: load/compute/store.
Therefore, in order to enable the pipeline to run normally, at least three threads
are required to feed data to the pipeline at the same time.
==============================================================
OverlapInnerLoop:
- Boxes denote subgraphs / subgraph Ops / loops
- Inputs/outputs are loop carried in order

.- outer loop ----------------------------------------.
|                  .- inner loop -.                   |
| load - compute - | - store      |                   |
|           load - | - compute -- | - store           |
|                  |   load ----- | - compute - store |
|                  '--------------'                   |
'-----------------------------------------------------'
         ^^^^^^^       ^^^^^^^        ^^^^^^^
         overlap       overlap        overlap

==============================================================
'''


def compile(model: onnx.ModelProto, args):
    """Compile ONNX to PopEF."""
    model_bytes = model.SerializeToString()
    outputs = [o.name for o in model.graph.output]

    options = CompilerOptions()
    options.batches_per_step = args.batches_per_step
    options.num_io_tiles = args.num_io_tiles

    executable = Compiler.compile(model_bytes, outputs, options)
    return executable


def run(executable, args):
    """Run PopEF."""
    # Create model runner
    config = RuntimeConfig()
    config.timeout_ns = 0
    # Create model runner
    model_runner = runtime.ModelRunner(executable, config)

    inputs_info = model_runner.get_model_inputs()
    outputs_info = model_runner.get_model_outputs()

    # Run in multiple threads
    def execute(bps, inputs_info, outputs_info):
        inputs = {}
        outputs = {}

        for input in inputs_info:
            inputs[input.name] = np.random.uniform(0, 1, input.shape).astype(
                input.numpy_data_type()
            )
        for output in outputs_info:
            outputs[output.name] = np.zeros(
                output.shape, dtype=output.numpy_data_type()
            )

        # To correctly generate the popvision report, iteration must be a
        # multiple of batches_per_step and greater than 2 * batches_per_step
        # There are 3 threads, so the total number feed into IPU is 3 * iteration
        iteration = bps
        for _ in range(iteration):
            model_runner.execute(inputs, outputs)

    threads = []
    num_threads = 3
    print(f"Run PopEF with {num_threads} threads.")
    for _ in range(num_threads):
        threads.append(
            threading.Thread(
                target=execute, args=(args.batches_per_step, inputs_info, outputs_info)
            )
        )

    for t in threads:
        t.start()

    for t in threads:
        t.join()
    print(f"Complete.")


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
    parser = argparse.ArgumentParser(
        description='Convert onnx model and run it on IPU.'
    )
    parser.add_argument(
        '--batches_per_step',
        type=int,
        default=16,
        help="The number of on-chip loop count.",
    )
    parser.add_argument(
        '--num_io_tiles',
        type=int,
        default=192,
        help="The number of IO tiles.",
    )
    args = parser.parse_args()
    model = default_model()
    exec = compile(model, args)
    run(exec, args)
