# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import sys
import time

import numpy as np
import onnx

import poprt

sys.path.append('../')
import helper


def run_poprt(model_proto, input_data):
    # Each iteration processes 1344 pixels
    # An image is a 800 * 800 rgb, so the num of iteration is 800 * 800 / 1344 = 476.19 = 477
    iteration = 477

    # Convert
    input_shape = {}
    for i in input_data:
        input_shape[i] = list(input_data[i].shape)
        # Convert float32 to float16
        if input_data[i].dtype == np.float32:
            input_data[i] = input_data[i].astype(np.float16)
    converted_model = poprt.converter.Converter(
        input_shape=input_shape, precision="fp16"
    ).convert(model_proto)

    # Compile
    options = poprt.compiler.CompilerOptions()
    outputs = [o.name for o in converted_model.graph.output]
    executable = poprt.compiler.Compiler.compile(converted_model, outputs, options)

    runner = poprt.runtime.ModelRunner(executable)
    outputs = runner.get_model_outputs()
    outputs_dict = {
        o.name: np.zeros(o.shape).astype(o.numpy_data_type()) for o in outputs
    }
    # Warmup
    for _ in range(10):
        runner.execute(input_data, outputs_dict)

    # Run performance
    time_cost = []
    for i in range(50):
        futures = []
        sess_start = time.time()
        for i in range(iteration):
            f = runner.executeAsync(input_data, outputs_dict)
            futures.append(f)

        for _, future in enumerate(futures):
            future.wait()
        sess_end = time.time()
        time_cost.append(sess_end - sess_start)
    avg_per_image = np.mean(time_cost)
    tput = 800 * 800 / avg_per_image
    print("Tput (800 * 800 / time_per_image): %s " % (tput))

    runner.execute(input_data, outputs_dict)
    return outputs_dict


if __name__ == "__main__":
    for bs in [1344]:
        print(f"Run batch size {bs}:")
        # Load ONNX model
        model = onnx.load("NeRF.onnx")
        # Generate input data
        input_info = {
            "input_1": ([bs, 3], np.float32),
            "input_2": ([bs, 3], np.float32),
            "input_3": ([bs, 3], np.float32),
        }
        input_data = helper.generate_data(input_info, 0.0, 0.2)

        ort_output_name, ort_res = helper.run_onnxruntime(model, input_data)
        poprt_dict = run_poprt(model, input_data)
        poprt_res = [poprt_dict[o] for o in ort_output_name]
        print(
            "Check precision between original ONNX with ONNXRUNTIME and converted ONNX with PopRT:"
        )
        helper.accuracy(ort_res, poprt_res)
