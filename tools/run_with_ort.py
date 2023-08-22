# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import os

import numpy as np
import onnxruntime as ort

np.random.seed(2022)


def get_synthetic_data(inputs_meta):
    """Get synthetic data based on input information.

    :param inputs_meta: Input information, including name, data type and shape.

    :return: {name, data}, will set the batch size to 1, if this is not
        provided.
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
    """Get session and synthetic data.

    :param onnx_model: The imported ONNX model

    :return sess: The inference session.
    :return feed_dicts: {input_name, input_data} and will set batch size to 1.
    """

    sess = ort.InferenceSession(model_name)

    inputs_meta = sess.get_inputs()
    feed_dicts = get_synthetic_data(inputs_meta)

    return sess, feed_dicts


def run(args):
    """Run and print the output.

    :param args: Config.
    """
    model_name = args.model
    if not os.path.exists(model_name):
        raise RuntimeError(f"{model_name} does not exist. Please check the model name.")

    sess, feed_dicts = get_sess_and_data(model_name)

    output_names = [output.name for output in sess.get_outputs()]

    outputs = sess.run(output_names, feed_dicts)

    if args.print:
        print(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run model with onnxruntime')
    parser.add_argument(
        '--model', type=str, required=True, help="Full path to the ONNX model"
    )
    parser.add_argument('--print', action='store_true', help="Print the model output.")
    args = parser.parse_args()

    run(args)
