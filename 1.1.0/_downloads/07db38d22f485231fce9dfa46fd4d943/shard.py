# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import datetime

import numpy as np
import onnx
import requests

from poprt import runtime
from poprt.compiler import Compiler, CompilerOptions
from poprt.converter import Sharder


def load_model():
    # Download model
    url = 'https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-7.onnx'
    response = requests.get(url)
    if response.status_code == 200:
        model = onnx.load_model_from_string(response.content)
    else:
        raise Exception(
            f"Failed to download model with status_code {response.status_code}"
        )
    return model


def manual_sharding(model):
    # Fix the batch size to 1
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

    # Sharding and pipelining info
    sharding_info = {
        "resnetv17_stage1__plus0": 0,
        "resnetv17_stage4_batchnorm2_fwd": 1,
        "resnetv17_stage4__plus0": 2,
    }
    pipelining_info = {
        "resnetv17_stage1__plus0": 0,
        "resnetv17_stage4_batchnorm2_fwd": 1,
        "resnetv17_stage4__plus0": 2,
    }
    model = Sharder(sharding_info=sharding_info, pipelining_info=pipelining_info).run(
        model
    )

    return model


def compile(model):
    # Compile the model with backend options
    model_bytes = model.SerializeToString()
    outputs = [o.name for o in model.graph.output]

    options = CompilerOptions()
    options.ipu_version = runtime.DeviceManager().ipu_hardware_version()
    # Sharding into 4 IPUs
    options.num_ipus = 4
    # Enable Sharding and Pipelining
    options.enable_pipelining = True
    options.virtual_graph_mode = "manual"
    options.batches_per_step = 16

    executable = Compiler.compile(model_bytes, outputs, options)
    runner_config = runtime.RuntimeConfig()
    runner_config.timeout_ns = datetime.timedelta(microseconds=0)
    runner = runtime.Runner(executable, runner_config)
    return runner


def run(runner):
    inputs_info = runner.get_execute_inputs()
    outputs_info = runner.get_execute_outputs()

    inputs = {}
    for i in inputs_info:
        inputs[i.name] = np.ones(i.shape, dtype=i.numpy_data_type())

    outputs = {}
    for o in outputs_info:
        outputs[o.name] = np.zeros(o.shape, dtype=o.numpy_data_type())

    runner.execute(inputs, outputs)


if __name__ == '__main__':
    model = load_model()
    model = manual_sharding(model)
    runner = compile(model)
    run(runner)
