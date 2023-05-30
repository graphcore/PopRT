# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import time

import imageio
import numpy as np
import onnx

from PIL import Image

from poprt import runtime
from poprt.compiler import Compiler


def preprocess(img_path):
    img = imageio.imread(img_path, pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def run(model_path):
    # Compile
    model = onnx.load(model_path)
    model_bytes = model.SerializeToString()
    outputs = [o.name for o in model.graph.output]
    executable = Compiler.compile(model_bytes, outputs)
    runner_config = runtime.RuntimeConfig()
    runner_config.timeout_ns = 0
    runner = runtime.Runner(executable, runner_config)

    # Get input and output info
    inputs_info = runner.get_execute_inputs()
    outputs_info = runner.get_execute_outputs()
    outputs = {}
    for output in outputs_info:
        outputs[output.name] = np.zeros(output.shape, dtype=output.numpy_data_type())

    top_1 = 0
    top_5 = 0
    num_img = 0
    start = 0
    for line in open("img.txt"):
        line = line.strip()
        img, label = line.split(" ")
        label = int(label)

        input_data = preprocess(img)
        input_data = input_data.flatten()
        feed_dicts = {
            inputs_info[0]
            .name: input_data.astype(inputs_info[0].numpy_data_type())
            .reshape(inputs_info[0].shape)
        }

        # Run
        end = time.time()
        print((end - start) * 1000)
        runner.execute(feed_dicts, outputs)
        start = time.time()
        prob = np.squeeze(outputs[outputs_info[0].name])
        index = np.argsort(prob)[::-1]
        if index[0] == label:
            top_1 += 1
        if label in index[:5]:
            top_5 += 1
        num_img += 1
        print(
            "{} top-1: {}, top-5: {}, label: {}".format(
                num_img, index[0], index[:5], label
            )
        )

    print("top-1: {}".format(top_1))
    print("top-5: {}".format(top_5))
    print("num_img: {}".format(num_img))
    print("top-1-acc: {}".format(top_1 / num_img))
    print("top-5-acc: {}".format(top_5 / num_img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='googlenet accuracy test')
    parser.add_argument('--model_path', type=str, help='model file path')
    args = parser.parse_args()
    run(args.model_path)
