# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import sys
import time

import numpy as np
import onnx

import poprt

sys.path.append('../')
import helper


def run_poprt(model_proto, input_data):
    # Convert
    input_shape = {}
    for i in input_data:
        input_shape[i] = list(input_data[i].shape)
        # Convert float32 to float16
        if input_data[i].dtype == np.float32:
            input_data[i] = input_data[i].astype(np.float16)
    converted_model = poprt.converter.Converter(
        input_shape=input_shape, precision="fp16", enable_insert_remap=True
    ).convert(model_proto)

    # Compile
    bps = 128
    options = poprt.compiler.CompilerOptions()
    options.rearrange_streams_on_host = False
    options.group_host_sync = False
    options.available_memory_proportion = 0.2
    options.enable_non_stable_softmax = True
    options.stream_buffering_depth = 2
    options.enable_prefetch_datastreams = True
    options.batches_per_step = bps
    outputs = [o.name for o in converted_model.graph.output]
    executable = poprt.compiler.Compiler.compile(converted_model, outputs, options)

    runner = poprt.runtime.Runner(executable)
    outputs = runner.get_execute_outputs()
    outputs_dict = {
        o.name: np.zeros(o.shape).astype(o.numpy_data_type()) for o in outputs
    }
    # Warmup
    for _ in range(10):
        runner.execute(input_data, outputs_dict)

    # Run performance
    futures = []
    sess_start = time.time()
    for i in range(100):
        f = runner.execute_async(input_data, outputs_dict)
        futures.append(f)

    for _, future in enumerate(futures):
        future.wait()
    sess_end = time.time()
    interval = (sess_end - sess_start) / 100
    tput = input_data["input"].shape[0] / interval
    print(f'Asyncronous avg Session Time : {interval * 1000:.3f}ms')
    print("Tput: %s " % (tput))

    runner.execute(input_data, outputs_dict)
    return outputs_dict


if __name__ == "__main__":
    for bs in [32, 56]:
        print(f"Run batch size {bs}:")
        # Load ONNX model
        model = onnx.load("CLIP.onnx")
        # Generate input data
        input_info = {
            "input": ([bs, 3, 224, 224], np.float32),
            "input/1": ([100, 77], np.int32),
        }
        input_data = helper.generate_data(input_info)
        ort_output_name, ort_res = helper.run_onnxruntime(model, input_data)
        poprt_dict = run_poprt(model, input_data)
        poprt_res = [poprt_dict[o] for o in ort_output_name]
        print(
            "Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:"
        )
        helper.accuracy(ort_res, poprt_res)
