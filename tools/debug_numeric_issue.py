# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse

import numpy as np
import onnx

from poprt.backend import get_session
from poprt.passes import Pass

np.random.seed(2022)


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        input_func = {}
        for kv in values:
            k, v = kv.split("=")
            if v not in ['ones', 'zeros', 'rand']:
                raise ValueError('Only support ones, zeros and rand')
            input_func[k] = v
        setattr(namespace, self.dest, input_func)


def get_synthetic_data(inputs_meta, input_func):
    feed_dicts = {}
    for input in inputs_meta.items():
        shape = input[1][0]
        dtype = input[1][1]

        func_name = input_func[input[0]]
        if func_name in ['rand']:
            data = np.random.rand(*shape)
        elif func_name in ['ones', 'zeros']:
            func = getattr(np, func_name)
            data = func(shape)

        feed_dicts[input[0]] = data.astype(dtype)

    return feed_dicts


def get_candidates(src_onnx: onnx.ModelProto, dst_onnx: onnx.ModelProto, op_type):
    src_tensors = []
    for node in src_onnx.graph.node:
        if node.op_type == op_type:
            for np_out in node.output:
                src_tensors.append(np_out)

    dst_tensors = []
    for node in dst_onnx.graph.node:
        for np_out in node.output:
            if np_out in src_tensors:
                dst_tensors.append(np_out)

    return dst_tensors


def get_compared_model(src_onnx: onnx.ModelProto, dst_onnx: onnx.ModelProto, output):
    input_names = [mp_in.name for mp_in in src_onnx.graph.input]
    output_names = [output]

    e = onnx.utils.Extractor(src_onnx)
    src_sub_model = e.extract_model(input_names, output_names)

    e = onnx.utils.Extractor(dst_onnx)
    dst_sub_model = e.extract_model(input_names, output_names)

    return src_sub_model, dst_sub_model


def check_precision(src_model, dst_model, input_func):
    ort_sess = get_session(src_model.SerializeToString(), 1, 'onnxruntime').load()
    prt_sess = get_session(dst_model.SerializeToString(), 1, "poprt").load()

    ort_inputs_info, ort_outputs_info = ort_sess.get_io_info()
    prt_inputs_info, prt_outputs_info = prt_sess.get_io_info()

    ort_out_names = [o for o in ort_outputs_info]
    prt_out_names = [o for o in prt_outputs_info]

    ort_feeds_dict = get_synthetic_data(ort_inputs_info, input_func)
    ort_outs = ort_sess.run(ort_out_names, ort_feeds_dict)

    prt_feeds_dict = {}
    for key in ort_feeds_dict.keys():
        # popart may optimize some input away
        if key in prt_inputs_info.keys():
            prt_feeds_dict[key] = ort_feeds_dict[key].astype(prt_inputs_info[key][1])
    prt_outs = prt_sess.run(prt_out_names, prt_feeds_dict)

    for ort_out, prt_out, key in zip(ort_outs, prt_outs, ort_out_names):
        dif = np.abs(ort_out - prt_out.astype(ort_out.dtype))
        dif_max = dif.max()
        ort_max = ort_out[dif == dif_max][0]
        prt_max = prt_out[dif == dif_max][0]

        dif_min = dif.min()
        ort_min = ort_out[dif == dif_min][0]
        prt_min = prt_out[dif == dif_min][0]

        print(f'{key} - shape: {ort_out.shape}')
        print(
            '    max: (%.6f, %.6f, %.6f), min: (%.6f, %.6f, %.6f), mean: (%.6f)'
            % (dif_max, ort_max, prt_max, dif_min, ort_min, prt_min, dif.mean())
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Numerical debugging assistant')
    parser.add_argument(
        '--src_onnx',
        type=str,
        required=True,
        help='Specify the original model',
    )
    parser.add_argument(
        '--dst_onnx',
        type=str,
        required=True,
        help='Specify the converted model',
    )
    parser.add_argument(
        '--input_func',
        type=str,
        metavar="KEY=VAL",
        nargs="+",
        action=StoreDictKeyPair,
        required=True,
        help='Specify func for each input tensor',
    )
    parser.add_argument(
        '--op_type', type=str, default='MatMul', help='Specify the operator types'
    )
    parser.add_argument(
        '--every_n', type=int, default=1, help='Specify sample rate 1/n'
    )
    args = parser.parse_args()

    src_onnx = onnx.load(args.src_onnx)
    dst_onnx = onnx.load(args.dst_onnx)

    # remove initializer from model.graph.input
    rm_initializer_from_input = Pass.get_pass('remove_initializer_from_input')
    src_onnx = rm_initializer_from_input(src_onnx)
    dst_onnx = rm_initializer_from_input(dst_onnx)

    # copy input_shape from dst_onnx to src_onnx
    for src_mp_in in src_onnx.graph.input:
        for dst_mp_in in dst_onnx.graph.input:
            if src_mp_in.name == dst_mp_in.name:
                src_mp_in.type.tensor_type.shape.CopyFrom(
                    dst_mp_in.type.tensor_type.shape
                )

    tensors = get_candidates(src_onnx, dst_onnx, args.op_type)
    filtered_tensors = tensors[0 :: args.every_n]

    for tensor in filtered_tensors:
        src_sub_model, dst_sub_model = get_compared_model(src_onnx, dst_onnx, tensor)
        check_precision(src_sub_model, dst_sub_model, args.input_func)

    print('model output:')
    check_precision(src_onnx, dst_onnx, args.input_func)
