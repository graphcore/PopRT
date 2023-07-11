# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import os

import numpy as np
import onnxruntime as ort

np.random.seed(2022)


def get_synthetic_data(inputs_meta):
    """Get synthetic data according to inputs_meta.

    :param inputs_meta: input info, include name, dtype, shape

    :return: {name, data}, will set batch_size to 1 if not provided
    """
    feed_dicts = {}
    for input in inputs_meta:
        dtype = np.float32
        if input.type == 'tensor(float16)':
            dtype = np.float16
        if input.type == 'tensor(int64)':
            dtype = np.int64
        if input.type == 'tensor(bool)':
            dtype = np.bool
        data = np.random.rand(*input.shape).astype(dtype)
        feed_dicts[input.name] = data

    return feed_dicts


def get_sess_and_data(model_name):
    """Get sess and synthetic data.

    :param onnx_model: the imported onnx model

    :return sess: InferenceSession
    :return feed_dicts: {input_name, input_data}, will set batch-size to 1
    """

    sess = ort.InferenceSession(model_name)

    inputs_meta = sess.get_inputs()
    feed_dicts = get_synthetic_data(inputs_meta)

    return sess, feed_dicts


def run(args):
    """Run and print the output.

    :param args: config
    """
    model_name = args.model
    if not os.path.exists(model_name):
        raise RuntimeError(f"{model_name} does not exist, please double check")

    sess, feed_dicts = get_sess_and_data(model_name)

    output_names = [output.name for output in sess.get_outputs()]

    outputs = sess.run(output_names, feed_dicts)

    if args.print:
        print(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run model with onnxruntime')
    parser.add_argument(
        '--model', type=str, required=True, help="full path of the onnx model"
    )
    parser.add_argument(
        '--print', action='store_true', help="print the output of model"
    )
    args = parser.parse_args()

    run(args)
