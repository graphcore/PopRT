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
        precision="fp16",
        enable_insert_remap=True,
        enable_erf_gelu=True,
        used_passes=["attention_padding", "merge_multi_slice"],
    ).convert(model_proto)

    # Compile
    options = poprt.compiler.CompilerOptions()
    options.rearrange_anchors_on_host = False
    options.group_host_sync = False
    options.batches_per_step = 128
    options.available_memory_proportion = 0.3
    options.stream_buffering_depth = 2
    options.num_io_tiles = 32
    options.enable_outlining = True
    options.outline_threshold = 0.5

    outputs = [o.name for o in converted_model.graph.output]
    executable = poprt.compiler.Compiler.compile(converted_model, outputs, options)

    runner = poprt.runtime.Runner(executable)
    outputs = runner.get_execute_outputs()
    outputs_dict = {
        o.name: np.zeros(o.shape).astype(o.numpy_data_type()) for o in outputs
    }
    # Warmup
    for _ in range(50):
        runner.execute(input_data, outputs_dict)

    # Run performance
    futures = []
    sess_start = time.time()
    for i in range(1500):
        f = runner.execute_async(input_data, outputs_dict)
        futures.append(f)

    for _, future in enumerate(futures):
        future.wait()
    sess_end = time.time()
    interval = (sess_end - sess_start) / 1500
    tput = input_data["input.1"].shape[0] / interval
    print(f'Asyncronous avg Session Time : {interval * 1000:.3f}ms')
    print("Tput: %s " % (tput))

    runner.execute(input_data, outputs_dict)
    return outputs_dict


if __name__ == "__main__":
    for bs in [24]:
        print(f"Run batch size {bs}:")
        # Load ONNX model
        model = onnx.load("open_swin-Tiny-BS24-fp32.onnx")

        # Generate input data
        input_info = {
            "input.1": ([bs, 3, 224, 224], np.float32),
        }
        input_data = helper.generate_data(input_info)
        ort_output_name, ort_res = helper.run_onnxruntime(model, input_data)
        poprt_dict = run_poprt(model, input_data)
        poprt_res = [poprt_dict[o] for o in ort_output_name]
        print(
            "Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:"
        )
        helper.accuracy(ort_res, poprt_res)
