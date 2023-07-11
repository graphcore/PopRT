# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse

import onnx

from poprt import runtime
from poprt.compiler import Compiler, CompilerOptions
from poprt.converter import Converter


def convert(model_proto: onnx.ModelProto, args) -> onnx.ModelProto:
    """Convert ONNX model to a new optimized ONNX model."""
    converter = Converter()
    converted_model = converter.convert(model_proto)
    return converted_model


def compile(model: onnx.ModelProto, args):
    """Compile ONNX to PopEF."""
    model_bytes = model.SerializeToString()
    outputs = [o.name for o in model.graph.output]

    options = CompilerOptions()
    options.batches_per_step = args.batches_per_step
    options.num_io_tiles = args.num_io_tiles
    options.group_host_sync = True
    options.stream_buffering_depth = 2
    options.enable_prefetch_datastreams = True
    options.rearrange_anchors_on_host = False
    options.partials_type = 'float'
    options.time_limit_scheduler = 0
    options.swap_limit_scheduler = 0
    options.transitive_closure_optimization_threshold = 0
    options.ipu_version = runtime.DeviceManager().ipu_hardware_version()

    Compiler.compile_and_export(model_bytes, outputs, args.popef_path, options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile onnx model to PopEF.')
    parser.add_argument(
        '--onnx_model',
        type=str,
        required=True,
        help="Full path of the onnx model.",
    )
    parser.add_argument(
        '--popef_path',
        type=str,
        default=None,
        help="Full path of the output popef model.",
    )
    parser.add_argument(
        '--batches_per_step',
        type=int,
        default=1000,
        help="The number of on-chip loop count.",
    )
    parser.add_argument(
        '--num_io_tiles',
        type=int,
        default=192,
        help="The number of IO tiles.",
    )
    parser.add_argument(
        '--skip_convert',
        action='store_true',
    )
    args = parser.parse_args()
    if args.popef_path is None:
        args.popef_path = f"{args.onnx_model}.popef"

    model = onnx.load(args.onnx_model)
    if not args.skip_convert:
        model = convert(model, args)

    compile(model, args)
