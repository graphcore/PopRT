# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import onnx

from poprt import runtime
from poprt.compiler import Compiler, CompilerOptions
from poprt.converter import Converter


class InferenceBase(object):
    def __init__(
        self, model_path, input_shape, precision, fp8_params, fp8_skip_op_names
    ):
        self.model_path = model_path
        self.precision = precision
        self.fp8_skip_op_names = fp8_skip_op_names
        self.fp8_params = fp8_params
        self.input_shape = input_shape

    def convert_compile(self):
        # load onnx model
        model = onnx.load(self.model_path)
        # Convert ONNX model to a new optimized ONNX model.
        converter = Converter(
            convert_version=11,
            precision=self.precision,
            input_shape=self.input_shape,
            fp8_params=self.fp8_params,
            fp8_skip_op_names=self.fp8_skip_op_names,
        )
        model = converter.convert(model)

        # Compile ONNX to PopEF.
        model_bytes = model.SerializeToString()
        outputs = [o.name for o in model.graph.output]

        options = CompilerOptions()
        options.num_ipus = 1
        options.ipu_version = runtime.DeviceManager().ipu_hardware_version()
        options.batches_per_step = 128
        if self.precision == 'fp16':
            options.partials_type = 'half'
        executable = Compiler.compile(model_bytes, outputs, options)

        # Create model runner
        self.model_runner = runtime.ModelRunner(executable)
        self.inputs_info = self.model_runner.get_model_inputs()
        self.outputs = {}
        self.outputs_info = self.model_runner.get_model_outputs()
        for output in self.outputs_info:
            self.outputs[output.name] = np.zeros(
                output.shape, dtype=output.numpy_data_type()
            )

    def format_feeds(self, feeds):
        inputs = {}
        for input in self.inputs_info:
            inputs[input.name] = np.array(feeds[input.name]).astype(
                input.numpy_data_type()
            )
        return inputs

    def predict(self, feeds):
        feeds = self.format_feeds(feeds)
        self.model_runner.execute(feeds, self.outputs)

        outputs = {}
        for output in self.outputs_info:
            outputs[output.name] = self.outputs[output.name].copy()
        return outputs


def convert_input_shape(input_shape: str):
    results = {}
    inputs = input_shape.split(" ")
    for input in inputs:
        key, value = input.split("=")
        value = value.split(",")
        results[key] = [int(x) for x in value]
    return results


def compile(
    model_path,
    input_shape: str,
    precision: str,
    fp8_params: str = None,
    fp8_skip_op_names: str = None,
):
    input_shape = convert_input_shape(input_shape)
    inference = InferenceBase(
        model_path=model_path,
        input_shape=input_shape,
        precision=precision,
        fp8_params=fp8_params,
        fp8_skip_op_names=fp8_skip_op_names,
    )
    inference.convert_compile()
    return inference
