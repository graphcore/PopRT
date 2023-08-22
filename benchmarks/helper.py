# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import onnx

from sklearn import metrics

import poprt


def get_inputs_info(model, bs):
    input_info = {}
    for i in model.graph.input:
        shape = [
            bs if d.dim_value <= 0 else d.dim_value
            for d in i.type.tensor_type.shape.dim
        ]
        dtype = onnx.helper.tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type)
        input_info[i.name] = (shape, dtype)
    return input_info


def generate_data(input_info, low=0.0, high=10.0):
    np.random.seed(2023)
    input_data = {}
    for i in input_info:
        shape = input_info[i][0]
        dtype = input_info[i][1]
        # Return random input data
        input_data[i] = ((high - low) * np.random.rand(*shape) + low).astype(dtype)
    return input_data


def run_onnxruntime(model_proto, input_data):
    # Run ONNXRUNTIME
    sess = poprt.backend.get_session(model_proto.SerializeToString(), 1, "onnxruntime")
    sess.load()
    _, outputs_info = sess.get_io_info()
    output_name = [o for o in outputs_info]
    # Generate random input data
    ort_res = sess.run(output_name, input_data)
    return output_name, ort_res


def accuracy(cpu_res, ipu_res):
    ipu_total_res = []
    cpu_total_res = []
    for i in range(len(cpu_res)):
        if ipu_res[i].shape != cpu_res[i].shape:
            raise Exception("ipu_res shape does not equal to cpu_res shape")

        ipu_total_res.append(ipu_res[i].flatten().astype(np.float32))
        cpu_total_res.append(cpu_res[i].flatten().astype(np.float32))
    ipu_total_res = np.concatenate(ipu_total_res, axis=0)
    cpu_total_res = np.concatenate(cpu_total_res, axis=0)

    # MSE
    print("mse: {}".format(metrics.mean_squared_error(ipu_total_res, cpu_total_res)))
    # MAE
    print("mae: {}".format(metrics.mean_absolute_error(ipu_total_res, cpu_total_res)))
