# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import sys
import time

import numpy as np
import onnx

import poprt

sys.path.append('../')
import helper


def run_poprt(model_proto, input_data, bs):
    # Convert
    input_shape = {}
    for i in input_data:
        input_shape[i] = list(input_data[i].shape)
        # Convert float32 to float16
        if input_data[i].dtype == np.float32:
            input_data[i] = input_data[i].astype(np.float16)
        # Convert int64 to int32
        if input_data[i].dtype == np.int64:
            input_data[i] = input_data[i].astype(np.int32)
    converted_model = poprt.converter.Converter(
        input_shape=input_shape, precision="fp16"
    ).convert(model_proto)

    # Compile
    options = poprt.compiler.CompilerOptions()
    options.rearrange_streams_on_host = False
    options.rearrange_anchors_on_host = False
    options.group_host_sync = False
    options.stream_buffering_depth = 2
    options.batches_per_step = 1024
    options.use_128bit_conv_unit_load = True
    options.enable_fast_reduce = True
    if bs == 8000:
        options.available_memory_proportion = 1.0
        options.num_io_tiles = 128
        options.enable_outlining = True
        options.outline_threshold = 1
    if bs == 51200:
        options.available_memory_proportion = 0.6
        options.num_io_tiles = 32
    outputs = [o.name for o in converted_model.graph.output]

    executable = poprt.compiler.Compiler.compile(converted_model, outputs, options)

    runner = poprt.runtime.Runner(executable)
    outputs = runner.get_execute_outputs()
    outputs_dict = {
        o.name: np.zeros(o.shape).astype(o.numpy_data_type()) for o in outputs
    }

    # Warmup
    for _ in range(100):
        runner.execute(input_data, outputs_dict)

    # Run performance
    futures = []
    sess_start = time.time()
    for i in range(5000):
        f = runner.execute_async(input_data, outputs_dict)
        futures.append(f)

    for _, future in enumerate(futures):
        future.wait()
    sess_end = time.time()
    interval = (sess_end - sess_start) / 5000
    tput = bs / interval
    print(f'Asyncronous avg Session Time : {interval * 1000:.3f}ms')
    print("Tput: %s " % (tput))
    runner.execute(input_data, outputs_dict)
    return outputs_dict


if __name__ == "__main__":
    for bs in [8000, 51200]:
        print(f"Run batch size {bs}:")
        # Load ONNX model
        model = onnx.load("WideDeep.onnx")
        # Generate input data
        input_info = {
            "new_categorical_placeholder_0": ([bs * 26, 2], np.int64),
            "new_numeric_placeholder_0": ([bs, 13], np.float32),
        }
        input_data = helper.generate_data(input_info)
        ort_output_name, ort_res = helper.run_onnxruntime(model, input_data)
        poprt_dict = run_poprt(model, input_data, bs)
        poprt_res = [poprt_dict[o] for o in ort_output_name]
        print(
            "Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:"
        )
        helper.accuracy(ort_res, poprt_res)
