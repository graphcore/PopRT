# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import os
import pickle
import sys

from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision

from data_preprocess import get_resnet50_preprocess_data

from poprt.backend import get_session
from poprt.converter import Converter
from poprt.passes import layer_precision_compare
from poprt.quantizer import quantize

sys.path.append(str(Path(__file__).resolve().parent.parent))


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


def export_model(resnet50_save_path):
    dummy_input = torch.randint(0, 255, (1, 3, 224, 224))
    dummy_input = dummy_input.to(torch.uint8).byte()
    model = torchvision.models.resnet.resnet50(pretrained=True).eval()
    model = Model(model)
    torch.onnx.export(
        model,
        dummy_input,
        resnet50_save_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )


def run_resnet50(model_path, half_partial, data_root, data_file):
    options = {'num_io_tiles': 32}
    if half_partial:
        options['partials_type'] = 'half'
    sess = get_session(model_path, 1, "poprt", options=options).load()
    inputs_info, outputs_info = sess.get_io_info()
    outputs_name = [o for o in outputs_info]
    input_name = list(inputs_info.keys())[0]

    for input_info in inputs_info.values():
        bs = input_info[0][0]
        break
    all_data, all_label = get_resnet50_preprocess_data(data_root, data_file)
    num_data = all_data.shape[0]

    pred_num = 0
    count = 0
    for idx in range(0, num_data, bs):
        batch_data = all_data[idx : idx + bs]
        batch_label = all_label[idx : idx + bs]
        feed_dicts = {input_name: batch_data}
        outputs = sess.run(outputs_name, feed_dicts)[0]
        pred = np.argmax(outputs, 1)
        pred_num += np.sum(pred == batch_label)
        count += bs
        print('acc: ', pred_num / count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='onnx squad')
    parser.add_argument(
        '--model_dir', type=str, default='model', help='model saved path'
    )
    parser.add_argument(
        '--model_run_type',
        type=str,
        default='fp8',
        choices=['fp8', 'fp16', 'fp8', 'fp8_weight'],
        help='model run type',
    )
    parser.add_argument(
        '--half_partial', action='store_true', help='enable inference with half partial'
    )
    parser.add_argument(
        '--calibration_with_data',
        action='store_true',
        help='Calibrate the fp8 model using the calibration data',
    )
    parser.add_argument(
        '--calibration_loss_type',
        type=str,
        default='kld',
        choices=['mse', 'mae', 'snr', 'kld', 'cos_dist'],
        help='Choose the calibration method, default is kld.',
    )
    parser.add_argument(
        '--num_of_layers_keep_fp16',
        type=int,
        default=0,
        help='Set the layer whose loss is topk to fp16 in fp8 quantization.',
    )
    parser.add_argument(
        '--precision_compare',
        action='store_true',
        help='Compare the output precision of conv/matmul/gemm between the origin and the converted model.',
    )
    args = parser.parse_args()
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    resnet50_save_path = args.model_dir + '/resnet50.onnx'
    resnet50_fp16_save_path = args.model_dir + '/resnet50_fp16.onnx'
    resnet50_fp8_save_path = args.model_dir + '/resnet50_fp8.onnx'
    # save model
    export_model(resnet50_save_path)

    data_root = 'data/'
    data_file = 'data/val_official_clean.csv'
    # save calibration data for quantize or precision compare
    calibration_data, _ = get_resnet50_preprocess_data(data_root, data_file)
    calibration_dict = {'input': calibration_data}
    calibration_save_path = f'{args.model_dir}/calibration_dict.pickle'
    with open(calibration_save_path, 'wb') as file:
        pickle.dump(calibration_dict, file)

    resnet50 = onnx.load(resnet50_save_path)
    model_run_path = resnet50_save_path
    converted_model = None
    if args.model_run_type == 'fp16':
        converter = Converter(convert_version=11, precision=args.model_run_type)
        converted_model = converter.convert(resnet50)
        onnx.save(converted_model, resnet50_fp16_save_path)
        model_run_path = resnet50_fp16_save_path
    elif args.model_run_type in ['fp8', 'fp8_weight']:
        # Keep the first conv_7x7 as fp16, because the speed of fp8 conv_7x7 with 3 input channels is very slow
        # Keep Conv_13 and the last Gemm to fp16, it can improve accuracy and hardly affect the speed
        if not args.calibration_with_data:
            converter = Converter(
                convert_version=11,
                precision=args.model_run_type,
                fp8_params='F143,F143,-3,-3',
                fp8_skip_op_names='Conv_5,Conv_13,Gemm_126',
            )
            converted_model = converter.convert(resnet50)
        else:
            # start quantization
            converter = Converter(
                convert_version=11,
                precision=args.model_run_type,
                quantize=True,
            )
            converter_model = converter.convert(resnet50)
            converted_model = quantize(
                onnx_model=converter_model,
                input_model=None,
                output_dir=args.model_dir,
                data_preprocess=calibration_save_path,
                precision=args.model_run_type,
                quantize_loss_type=args.calibration_loss_type,
                num_of_layers_keep_fp16=args.num_of_layers_keep_fp16,
            )

        onnx.save(converted_model, resnet50_fp8_save_path)
        model_run_path = resnet50_fp8_save_path

    if args.precision_compare:
        layer_precision_compare(
            origin_model_path=resnet50_save_path,
            converted_model=converted_model,
            data_preprocess=calibration_save_path,
            output_dir=args.model_dir,
        )

    run_resnet50(model_run_path, args.half_partial, data_root, data_file)
