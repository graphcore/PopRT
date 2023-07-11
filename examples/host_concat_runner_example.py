# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import copy

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.shape_inference

from onnx import TensorProto, helper

from poprt import Pass, runtime
from poprt._names import sHostConcatRunnerConfig
from poprt.compiler import Compiler, CompilerOptions, Executable
from poprt.passes.onnx_helper import topological_sort

np.random.seed(0)

sHostConcatRunnerConfig = 'HostConcatRunnerConfig'


def compile_overlapio(model: onnx.ModelProto, fp: str = ''):
    """Compile ONNX to PopEF."""
    outputs = [o.name for o in model.graph.output]

    options = CompilerOptions()
    options.batches_per_step = 1000
    options.num_io_tiles = 128
    # options.group_host_sync = True
    options.stream_buffering_depth = 2
    options.enable_prefetch_datastreams = True
    options.rearrange_anchors_on_host = False
    options.partials_type = 'float'
    # options.enable_outlining = False
    options.time_limit_scheduler = 0
    options.swap_limit_scheduler = 0
    options.transitive_closure_optimization_threshold = 0
    options.ipu_version = runtime.DeviceManager().ipu_hardware_version()
    options.constant_weights = False
    # options.enable_engine_caching = True
    # options.cache_path = './cache'

    for meta in model.metadata_props:
        if meta.key == sHostConcatRunnerConfig:
            options.opaque_blobs = {sHostConcatRunnerConfig: meta.value}
            break

    if fp:
        Compiler.compile_and_export(model, outputs, fp, options)
        executable = Executable(fp)
    else:
        executable = Compiler.compile(model, outputs, options)
    return executable


def compile_plain(model: onnx.ModelProto, fp: str = ''):
    """Compile ONNX to PopEF."""
    outputs = [o.name for o in model.graph.output]

    options = CompilerOptions()
    options.constant_weights = False
    options.ipu_version = runtime.DeviceManager().ipu_hardware_version()
    # options.enable_engine_caching = True
    # options.cache_path = './cache'

    for meta in model.metadata_props:
        if meta.key == sHostConcatRunnerConfig:
            options.opaque_blobs = {sHostConcatRunnerConfig: meta.value}
            break

    if fp:
        Compiler.compile_and_export(model, outputs, fp, options)
        executable = Executable(fp)
    else:
        executable = Compiler.compile(model, outputs, options)
    return executable


def run(executable, merge_executable, merged_infos):
    # run with LightRunner
    config = runtime.LightRunnerConfig()
    config.check_package_hash = False
    runner = runtime.LightRunner(executable, config)

    # generate random inputs
    inputs = {}
    outputs = {}
    host_concat_outputs = {}
    for info in runner.get_execute_inputs():
        inputs[info.name] = np.random.uniform(0, 1, info.shape).astype(
            info.numpy_data_type()
        )
    for info in runner.get_execute_outputs():
        outputs[info.name] = np.zeros(info.shape, dtype=info.numpy_data_type())
        host_concat_outputs[info.name] = np.zeros(
            info.shape, dtype=info.numpy_data_type()
        )

    def run_with_light_runner(runnerx, inputsx, outputsx):
        future = runnerx.execute_async(inputsx, outputsx)
        future.wait()
        okey = next(iter(outputsx.keys()))
        print(f"run_with_light_runner:")
        print(outputsx[okey].flatten()[0:20])

    def run_with_host_concat_runner(runnerx, inputsx, outputsx):
        future = runnerx.execute_async(inputsx, outputsx)
        future.wait()
        okey = next(iter(outputsx.keys()))
        print(f"run_with_host_concat_runner:")
        print(outputsx[okey].flatten()[0:20])

    config = runtime.HostConcatRunnerConfig()
    config.check_package_hash = False
    config.ring_buffer_size_multiplier = 30
    config.timeout_ns = 100

    hccs = {}
    for merged_info in merged_infos:
        hcc = runtime.HostConcatConfig()
        hcc.tensor_name = merged_info[0]
        hcc.input_tensor_names = merged_info[1]
        hcc.shape = merged_info[2]
        hcc.data_size_bytes = merged_info[3]
        hccs[hcc.tensor_name] = hcc

    config.host_concat_config = hccs
    host_concat_runner = runtime.HostConcatRunner(merge_executable, config)

    run_with_light_runner(runner, inputs, outputs)
    run_with_host_concat_runner(host_concat_runner, inputs, host_concat_outputs)


def run_with_lightrunner(executable):
    config = runtime.LightRunnerConfig()
    config.check_package_hash = False

    runner = runtime.LightRunner(executable, config)

    # generate random inputs
    inputs = {}
    outputs = {}
    host_concat_outputs = {}
    for info in runner.get_execute_inputs():
        inputs[info.name] = np.random.uniform(0, 1, info.shape).astype(
            info.numpy_data_type()
        )
    for info in runner.get_execute_outputs():
        outputs[info.name] = np.zeros(info.shape, dtype=info.numpy_data_type())
        host_concat_outputs[info.name] = np.zeros(
            info.shape, dtype=info.numpy_data_type()
        )

    def run_with_light_runner(runnerx, inputsx, outputsx):
        future = runnerx.execute_async(inputsx, outputsx)
        future.wait()
        okey = next(iter(outputsx.keys()))
        print(f"run_with_light_runner:")
        print(outputsx[okey].flatten()[0:20])

    run_with_light_runner(runner, inputs, outputs)
    return inputs, outputs


def run_with_host_concat_runner(executable, inputs, outputs):
    config = runtime.HostConcatRunnerConfig()
    config.check_package_hash = False
    config.ring_buffer_size_multiplier = 30
    config.timeout_ns = 100

    runner = runtime.HostConcatRunner(executable, config)

    def run_with_host_concat_runner(runnerx, inputsx, outputsx):
        future = runnerx.execute_async(inputsx, outputsx)
        future.wait()
        okey = next(iter(outputsx.keys()))
        print(f"run_with_host_concat_runner:")
        print(outputsx[okey].flatten()[0:20])

    run_with_host_concat_runner(runner, inputs, outputs)


def make_model():
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, (512, 512))]

    inputs_0, inputs_1, inputs_2 = [], [], []
    # N0 = N1 * 5 + N2 * 50
    # N0, N1, N2 = 600, 100, 2
    # N0, N1, N2 = 300, 50, 1
    # N0, N1, N2 = 150, 20, 1
    N0, N1, N2 = 60, 2, 1
    for i in range(N0):
        inputs_0.append(
            helper.make_tensor_value_info(f"input_0_{i}", TensorProto.FLOAT, (512, 1))
        )
    for i in range(N1):
        inputs_1.append(
            helper.make_tensor_value_info(f"input_1_{i}", TensorProto.FLOAT, (512, 5))
        )
    for i in range(N2):
        inputs_2.append(
            helper.make_tensor_value_info(f"input_2_{i}", TensorProto.FLOAT, (512, 50))
        )

    inits = []
    inits.append(
        helper.make_tensor(
            "w0",
            TensorProto.FLOAT,
            (N0, 512),
            np.random.uniform(0, 1, (N0, 512)).astype(np.float32),
        )
    )
    inits.append(
        helper.make_tensor(
            "w1",
            TensorProto.FLOAT,
            (N0, 512),
            np.random.uniform(0, 1, (N0, 512)).astype(np.float32),
        )
    )

    nodes = []
    nodes.append(
        helper.make_node("Concat", [x.name for x in inputs_0], ["concat_0"], axis=1)
    )
    nodes.append(
        helper.make_node("Concat", [x.name for x in inputs_1], ["concat_1"], axis=1)
    )
    nodes.append(
        helper.make_node("Concat", [x.name for x in inputs_2], ["concat_2"], axis=1)
    )
    nodes.append(
        helper.make_node("Concat", ['concat_1', 'concat_2'], ["concat_3"], axis=1)
    )
    nodes.append(helper.make_node("MatMul", ['concat_0', 'w0'], ["matmul_0"]))
    nodes.append(helper.make_node("MatMul", ['concat_3', 'w1'], ["matmul_1"]))
    nodes.append(helper.make_node("Add", ["matmul_0", "matmul_1"], ["output"]))

    graph = helper.make_graph(
        nodes,
        "matmul_test",
        inputs_0 + inputs_1 + inputs_2,
        outputs,
        inits,
    )

    opset_imports = [helper.make_opsetid("", 11)]
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.opset_import.append(onnx.helper.make_opsetid("ai.graphcore", 1))

    # skip shape inference
    # model = onnx.shape_inference.infer_shapes(model)

    sorted_nodes = topological_sort(model.graph)
    model.graph.ClearField('node')
    for node in sorted_nodes:
        model.graph.node.append(node)
    onnx.checker.check_model(model)
    return model


if __name__ == '__main__':
    model = make_model()

    merged_inputs = []
    # (dtype, shape)
    merged_inputs.append((onnx.TensorProto.FLOAT, (512, 1)))
    merged_inputs.append((onnx.TensorProto.FLOAT, (512, 5)))

    apass = Pass.get_pass('apply_host_concat_split', merged_inputs)
    merge_model = apass.run(copy.copy(model))

    print(f"model len(inputs): {len(model.graph.input)}")
    print(f"merged model len(inputs): {len(merge_model.graph.input)}")

    # debug
    # onnx.save(model, "host_concat_runner_example.onnx")
    # onnx.save(merge_model, "host_concat_runner_example.merge.onnx")

    # compile = compile_overlapio
    compile = compile_plain

    # raw model cost more compile time
    fp0 = 'host_concat_runner_example.popef'
    exe = compile(model, fp0)

    # merged model cost less compile time
    fp1 = 'host_concat_runner_example.merge.popef'
    merge_exe = compile(merge_model, fp1)

    # use merged_infos
    # run(fp0, fp1, merged_infos_list)

    # use merged_infos from popef model
    inputs, outputs = run_with_lightrunner(fp0)
    run_with_host_concat_runner(fp1, inputs, outputs)
