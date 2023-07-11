# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse

import onnx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract subgraph from onnx.')
    parser.add_argument(
        '--src_onnx', type=str, help="Full path of the source onnx model."
    )
    parser.add_argument(
        '--dst_onnx', type=str, help="Full path of the dest onnx model."
    )
    parser.add_argument(
        '--input_names',
        type=str,
        nargs='+',
        required=True,
        help="Input names.",
    )
    parser.add_argument(
        '--output_names',
        type=str,
        nargs='+',
        required=True,
        help="Output names.",
    )
    args = parser.parse_args()

    onnx.utils.extract_model(
        args.src_onnx, args.dst_onnx, args.input_names, args.output_names
    )
