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
    if bs == 1:
        converted_model = poprt.converter.Converter(
            input_shape=input_shape, precision="fp16"
        ).convert(model_proto)
    elif bs == 64:
        converted_model = poprt.converter.Converter(
            input_shape=input_shape,
            precision="fp16",
            used_passes=["matmul_rotary_embedding"],
        ).convert(model_proto)
    elif bs == 356:
        converted_model = poprt.converter.Converter(
            input_shape=input_shape,
            precision="fp16",
            used_passes=["matmul_rotary_embedding", "pre_scale"],
        ).convert(model_proto)

    # Compile
    options = poprt.compiler.CompilerOptions()
    options.stream_buffering_depth = 2
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
    tput = input_data["input_ids"].shape[0] / interval
    print(f'Asyncronous avg Session Time : {interval * 1000:.3f}ms')
    print("Tput: %s " % (tput))

    runner.execute(input_data, outputs_dict)
    return outputs_dict


if __name__ == "__main__":
    for bs in [1, 64, 356]:
        print(f"Run batch size {bs}:")
        # Load ONNX model
        model = onnx.load("RoformerV2.onnx")
        # Generate input data
        input_data = {}
        input_data.update(
            helper.generate_data({"input_ids": ([bs, 128], np.int64)}, 0, 2000)
        )
        input_data.update(
            helper.generate_data({"token_type_ids": ([bs, 128], np.int64)}, 1, 1)
        )
        input_data.update(
            helper.generate_data({"attention_mask": ([bs, 128], np.int64)}, 1, 1)
        )
        ort_output_name, ort_res = helper.run_onnxruntime(model, input_data)
        poprt_dict = run_poprt(model, input_data, bs)
        poprt_res = [poprt_dict[o] for o in ort_output_name]
        print(
            "Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:"
        )
        helper.accuracy(ort_res, poprt_res)
