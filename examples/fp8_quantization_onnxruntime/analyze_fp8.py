# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import math

import pandas as pd
import torch

from ppq import layerwise_error_analyse
from ppq.core.ffi_cpu import CPU_COMPLIER
from utils.convert_compile_and_run import compile
from utils.eval_imagenet import evaluate_ipu_with_imagenet, load_imagenet_from_directory
from utils.ppq_quantize import get_first_last_conv_name, quantize

CPU_COMPLIER.complie()


def get_dataloader(args):
    dataloader = load_imagenet_from_directory(args.val_dir, batchsize=args.batch_size)
    return dataloader


def get_calibrate_dataloader(args):
    """for data in dataloader.

    # data: Union[dict, torch.Tensor] # data not include label
    """
    dataloader = load_imagenet_from_directory(
        args.val_dir,
        batchsize=args.batch_size,
        shuffle=False,
        subset=512,
        require_label=False,
        num_of_workers=8,
    )
    return dataloader


def eval_ipu_model(args, inference, dataloader):
    result = evaluate_ipu_with_imagenet(
        inference, '', batchsize=args.batch_size, imagenet_validation_loader=dataloader
    )
    return result


def get_dummy_input(input_shapes: str):
    input_shape_list = input_shapes.split(" ")
    if len(input_shape_list) == 1:
        value = input_shape_list[0].split("=")[1].split(",")
        value = [int(x) for x in value]
        dummy_input = torch.zeros(size=value, device='cpu')
        return dummy_input
    else:
        dummy_input = {}
        for input_shape in input_shape_list:
            key = input_shape.split("=")[0]
            value = input_shape.split("=")[1].split(",")
            value = [int(x) for x in value]
            dummy_input[key] = torch.zeros(size=value, device='cpu')
        return dummy_input


def eval_ipu_scale(args, dataloader, dataframe):
    fp8_params = 'F143,F143,{},{}'
    fp8_skip_op_names = get_first_last_conv_name(
        args.model_path, args.default_skip_layer
    )
    best_result = -10000
    best_scale = -10000
    for scale in [-5, -4, -3, -2, -1, 0, 1, 2]:
        inference = compile(
            args.model_path,
            args.input_shapes,
            'fp8',
            fp8_params.format(scale, scale),
            fp8_skip_op_names=fp8_skip_op_names,
        )
        result = eval_ipu_model(args, inference, dataloader)
        dataframe.loc[len(dataframe)] = [scale, fp8_skip_op_names, '%.3f' % result]
        if result > best_result:
            best_result = result
            best_scale = scale
    return best_scale


def ppq_error_analyse(args, dataloader, scale):
    dummy_input = get_dummy_input(args.input_shapes)
    graph = quantize(
        args.model_path, dataloader, dummy_input, math.pow(2, scale - 1), []
    )
    error = layerwise_error_analyse(
        graph=graph, running_device='cpu', method='snr', dataloader=dataloader
    )
    # Extract the top 8 ops with the largest error.
    skip_layers = []
    collection = [(name, value) for name, value in error.items()]
    collection = sorted(collection, key=lambda x: x[1], reverse=True)
    for i in range(8):
        if i < len(collection):
            skip_layers.append(collection[i][0])
    return skip_layers


def eval_ipu_skip_layers(args, dataloader, skip_layers, scale, dataframe):
    fp8_params = 'F143,F143,{},{}'
    fp8_skip_op_names = get_first_last_conv_name(
        args.model_path, args.default_skip_layer
    )
    for layer_name in skip_layers:
        if layer_name in fp8_skip_op_names:
            continue
        fp8_skip_op_names += ','
        fp8_skip_op_names += layer_name
        inference = compile(
            args.model_path,
            args.input_shapes,
            'fp8',
            fp8_params.format(scale, scale),
            fp8_skip_op_names=fp8_skip_op_names,
        )
        result = eval_ipu_model(args, inference, dataloader)
        dataframe.loc[len(dataframe)] = [scale, fp8_skip_op_names, '%.3f' % result]
    return None


def eval_ipu_fp16(args, dataloader, dataframe):
    inference = compile(args.model_path, args.input_shapes, 'fp16')
    result = eval_ipu_model(args, inference, dataloader)
    dataframe.loc[len(dataframe)] = ["fp16", None, '%.3f' % result]
    return None


def analyze_fp8(args):
    dataloader = get_dataloader(args)
    cali_dataloader = get_calibrate_dataloader(args)
    dataframe = pd.DataFrame(columns=('scale', 'skip layer', 'acc'))
    # test the accuracy of multiple scales
    scale = eval_ipu_scale(args, dataloader, dataframe)
    # test the quantization error of the highest precision scale
    skip_layers = ppq_error_analyse(args, cali_dataloader, scale)
    # skip layers with high quantization errors one by one
    eval_ipu_skip_layers(args, dataloader, skip_layers, scale, dataframe)
    # test the accuracy of FP16
    eval_ipu_fp16(args, dataloader, dataframe)
    dataframe.to_csv(args.output_path, index=False)
    return dataframe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze FP8')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of model')

    parser.add_argument(
        '--input_shapes',
        type=str,
        default='input=8,3,224,224',
        help='onnx model input shapes, \
            bert example: input_ids:0=1,256 input_mask:0=1,256 segment_ids:0=1,256',
    )

    parser.add_argument(
        '--model_path', type=str, default='./inception.onnx', help='onnx model path'
    )

    parser.add_argument(
        '--val_dir', type=str, default='/fp8_test/validation', help='onnx model path'
    )

    parser.add_argument(
        '--output_path', type=str, default='./output.csv', help='output dataframe path'
    )

    parser.add_argument(
        '--default_skip_layer',
        type=str,
        choices=['first', 'last', 'both', ''],
        default='both',
        help='default fp8 skip layers',
    )

    args = parser.parse_args()
    analyze_fp8(args)
