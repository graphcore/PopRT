# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import os

import numpy as np
import onnx

from onnx import numpy_helper

from poprt.passes.onnx_helper import topological_sort


def create_new_model(model):
    new_onnx_model = onnx.ModelProto()
    new_onnx_model.CopyFrom(model)
    new_onnx_model.graph.ClearField('node')
    new_onnx_model.graph.ClearField('initializer')
    new_onnx_model.graph.ClearField('value_info')
    return new_onnx_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert model from fp16 to fp32.')
    parser.add_argument('--fp16_model', type=str, help="path of the fp16 model.")
    parser.add_argument('--fp32_model', type=str, help="path of the fp32 model.")
    args = parser.parse_args()

    if os.path.exists(args.fp16_model):
        model = onnx.load(args.fp16_model)
    else:
        raise FileNotFoundError(
            f"{args.fp16_model} doesn't exists, please check carefully."
        )
    new_model = create_new_model(model)

    # insert cast on model input
    input_remap_dicts = {}
    for mp_in in new_model.graph.input:
        node_name = mp_in.name + '_to_fp32'
        input_remap_dicts[mp_in.name] = [node_name]
        node = onnx.helper.make_node(
            'Cast',
            name=node_name,
            inputs=[mp_in.name],
            outputs=[node_name],
            to=onnx.TensorProto.FLOAT,
        )
        new_model.graph.node.append(node)

    # convert initializer
    for mp_w in model.graph.initializer:
        np_array = numpy_helper.to_array(mp_w)
        if np_array.dtype in [np.float16]:
            new_mp_w = onnx.helper.make_tensor(
                mp_w.name,
                data_type=onnx.TensorProto.FLOAT,
                dims=list(np_array.shape),
                vals=list(np_array.astype(np.float32).flatten()),
            )
            new_model.graph.initializer.append(new_mp_w)
        else:
            new_model.graph.initializer.append(mp_w)

    # insert node
    outputs = [mp_out.name for mp_out in new_model.graph.output]
    output_remap_dicts = {}

    # topologically sort if needed
    try:
        onnx.checker.check_model(model)
        sorted_nodes = [node for node in model.graph.node]
    except onnx.checker.ValidationError:
        sorted_nodes = topological_sort(model.graph)

    for node in sorted_nodes:
        new_node = onnx.NodeProto()
        new_node.CopyFrom(node)

        for i, np_in in enumerate(new_node.input):
            if np_in in input_remap_dicts.keys():
                new_node.input[i] = input_remap_dicts[np_in][0]
            # the output tensor may have multiple consumers, so we should record the name mapping
            elif np_in in output_remap_dicts.keys():
                new_node.input[i] = output_remap_dicts[np_in][0]

        if (
            new_node.op_type == 'Cast'
            and new_node.attribute[0].i == onnx.TensorProto.FLOAT16
        ):
            new_node.attribute[0].i = onnx.TensorProto.FLOAT

        if (
            new_node.op_type == 'ConstantOfShape'
            and new_node.attribute[0].t.data_type == onnx.TensorProto.FLOAT16
        ):
            np_array = numpy_helper.to_array(node.attribute[0].t).astype(np.float32)
            new_tensor = onnx.helper.make_tensor(
                name=node.attribute[0].t.name,
                data_type=onnx.TensorProto.FLOAT32,
                dims=np_array.shape,
                vals=np_array.flatten().tobytes(),
                raw=True,
            )
            new_node.attribute[0].t.CopyFrom(new_tensor)

        new_model.graph.node.append(new_node)

        new_node = new_model.graph.node[-1]
        for i, np_out in enumerate(new_node.output):
            if np_out in outputs:
                new_node.output[i] = np_out + '_fp32'
                node_name = new_node.name + '_to_fp16'
                cast_node = onnx.helper.make_node(
                    'Cast',
                    name=node_name,
                    inputs=[new_node.output[i]],
                    outputs=[np_out],
                    to=onnx.TensorProto.FLOAT16,
                )
                new_model.graph.node.append(cast_node)
                output_remap_dicts[np_out] = [new_node.output[i]]

    onnx.save(new_model, args.fp32_model)
