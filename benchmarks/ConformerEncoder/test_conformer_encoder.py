# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import logging
import sys
import tempfile

from typing import Any, Dict

import numpy as np
import onnx

import poprt

from poprt.compiler import CompilerOptions
from poprt.utils import get_input_info_from_model_proto

sys.path.append("..")
import os

from benchmark import get_raw_inputs, run_benchmark

np.random.seed(2023)


def get_inputs(
    input_shape: Dict[str, Any],
    input_dtype: Dict[str, Any],
):
    inputs = {}
    # use random data
    for name, shape in input_shape.items():
        if name == "src_pad_mask":
            inputs[name] = np.random.randint(0, 2, shape).astype(input_dtype[name])
        else:
            inputs[name] = np.random.rand(*shape).astype(input_dtype[name])

    return inputs


def export_conformer_onnx(dir):
    # onnx file comes from byte mlperf repo:
    # https://github.com/bytedance/ByteMLPerf/blob/main/byte_mlperf/prepare_model_and_dataset.sh
    os.system(
        f"wget -O {dir}/open_conformer.tar https://lf-bytemlperf.17mh.cn/obj/bytemlperf-zoo/open_conformer.tar"
    )
    os.system(f"tar xf open_conformer.tar -C {dir}")


def test_func_fp16(batch_size: int, bps: int, enable_overlapio: bool):
    size = (3, 64, 512)
    raw_input_dtype = {'src': np.float32, 'src_pad_mask': bool}
    input_shape = {'input': [batch_size, *size]}

    with tempfile.TemporaryDirectory() as tmpdir:
        export_conformer_onnx(tmpdir)
        raw_onnx_file = tmpdir + "/" + "open_conformer/conformer_encoder.onnx"
        logging.info(
            f"start benchmark for model: {raw_onnx_file},input_shape {input_shape}, precision: fp16"
        )
        raw_model = onnx.load(raw_onnx_file)
        model = poprt.Converter(
            input_shape=input_shape,
            batch_size=batch_size,
            batch_axis=0,
            precision='fp16',
            enable_insert_remap=True,
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
            precision='fp16', batch_size=batch_size, size=size, onnx_file=raw_onnx_file
        )
        logging.info(benchmark_info)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Inference conformer encoder with Poprt'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch size of inference'
    )
    parser.add_argument('--bps', type=int, default=128, help='on-chip loop times')
    parser.add_argument(
        '--enable_overlapio', action='store_true', help='enable io tile'
    )
    args = parser.parse_args()
    test_func_fp16(
        batch_size=args.batch_size,
        bps=args.bps,
        enable_overlapio=args.enable_overlapio,
    )
