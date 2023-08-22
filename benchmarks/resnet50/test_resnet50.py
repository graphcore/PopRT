# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import logging
import sys
import tempfile

from typing import Any, Dict

import numpy as np
import onnx
import torch
import torchvision.models as models

import poprt

from poprt.compiler import CompilerOptions
from poprt.utils import get_input_info_from_model_proto

sys.path.append("..")
from benchmark import get_raw_inputs, run_benchmark

np.random.seed(2023)


# online normalize
class Model(torch.nn.Module):
    def __init__(self, resnet):
        super(Model, self).__init__()
        self.resnet = resnet

        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])

        self.mul = 1.0 / (self.std * 255.0).unsqueeze(0)
        self.sub = (self.mean / self.std).unsqueeze(0)

    def forward(self, x):
        x = x.to(torch.float)
        x = x * self.mul - self.sub
        o = self.resnet(x)
        return o


def get_inputs(
    input_shape: Dict[str, Any],
    input_dtype: Dict[str, Any],
):
    inputs = {}
    # use random data
    for name, shape in input_shape.items():
        inputs[name] = np.random.randint(0, 255, shape).astype(input_dtype[name])
    return inputs


def export_resnet50(onnx_file, batch, enable_8bit_input):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()

    dummy_input = torch.randint(0, 255, (batch, 3, 224, 224))

    input_names = ["input"]
    output_names = ["output"]
    if enable_8bit_input:
        dummy_input = dummy_input.to(torch.uint8).byte()
        model = Model(model)
    else:
        dummy_input = dummy_input.to(torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )


def test_func(
    batch_size: int,
    precision: str,
    bps: int,
    enable_overlapio: bool,
    enable_8bit_input: bool,
):
    onnx_file = "resnet50.onnx"
    size = (3, 224, 224)
    raw_input_dtype = {'input': np.uint8 if enable_8bit_input else np.float32}
    input_shape = {'input': [batch_size, *size]}

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_onnx_file = tmpdir + "/batch_size_" + str(batch_size) + "_" + onnx_file
        export_resnet50(
            onnx_file=raw_onnx_file,
            batch=batch_size,
            enable_8bit_input=enable_8bit_input,
        )
        logging.info(
            f"start benchmark for model: {raw_onnx_file},input_shape {input_shape}, precision: {precision}"
        )
        raw_model = onnx.load(raw_onnx_file)
        model = poprt.Converter(
            input_shape=input_shape,
            precision=precision,
            batch_size=batch_size,
            batch_axis=0,
        ).convert(raw_model)

        options = CompilerOptions()
        options.stream_buffering_depth = 2
        if enable_overlapio:
            options.num_io_tiles = 32
        options.batches_per_step = bps
        options.rearrange_anchors_on_host = False
        options.partials_type = 'half'

        input_shape, input_dtype = get_input_info_from_model_proto(model)
        inputs = get_inputs(input_shape, input_dtype)
        raw_inputs = get_raw_inputs(inputs, raw_input_dtype)
        benchmark_info = run_benchmark(
            model,
            inputs,
            raw_model=raw_onnx_file,
            raw_inputs=raw_inputs,
            compiler_options=options,
        )
        benchmark_info.add_attrs(
            precision=precision, batch_size=batch_size, size=size, onnx_file=onnx_file
        )
        logging.info(benchmark_info)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Inference resnet50 with Poprt')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch size of inference'
    )
    parser.add_argument('--bps', type=int, default=128, help='on-chip loop times')
    parser.add_argument(
        '--precision', choices=['fp16', 'fp32'], help='enable inference with fp16'
    )
    parser.add_argument(
        '--enable_overlapio', action='store_true', help='enable io tile'
    )
    parser.add_argument(
        '--enable_8bit_input',
        action='store_true',
        help='enable inference with 8-bit input',
    )
    args = parser.parse_args()
    test_func(
        batch_size=args.batch_size,
        precision=args.precision,
        bps=args.bps,
        enable_overlapio=args.enable_overlapio,
        enable_8bit_input=args.enable_8bit_input,
    )
