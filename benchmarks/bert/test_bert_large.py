# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import logging
import os
import sys
import tempfile

from typing import Any

import numpy as np
import onnx

import poprt

from poprt.compiler import CompilerOptions

sys.path.append("..")
from benchmark import get_raw_inputs, run_benchmark

np.random.seed(2023)


def export_bert(dir):
    os.system(
        f"python -m transformers.onnx --model=bert-large-uncased {dir} --feature question-answering"
    )


class BertInputs(object):
    def __init__(
        self, input_ids, attention_mask, token_type_ids, input_len, max_seq_len
    ):
        # pad 0
        self.input_ids = np.expand_dims(input_ids, axis=0)
        self.attention_mask = np.expand_dims(attention_mask, axis=0)
        self.token_type_ids = np.expand_dims(token_type_ids, axis=0)
        self.input_len = [input_len]

    # merge elements
    def __add__(self, other):
        self.input_ids = np.vstack((self.input_ids, other.input_ids))
        self.attention_mask = np.vstack((self.attention_mask, other.attention_mask))
        self.token_type_ids = np.vstack((self.token_type_ids, other.token_type_ids))
        self.input_len = self.input_len + other.input_len
        return self


def gen_bert_inputs(batch_size: int, seq_length: int, dtype: Any):
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    text = [
        "Where is london located",
    ]
    text = text * 100
    raw_inputs = []
    for elem in text[:batch_size]:
        feature = tokenizer(
            elem,
            return_tensors='np',
            padding='max_length',
            max_length=seq_length,
            truncation=True,
        )
        trans_feature = BertInputs(
            input_ids=np.array(feature['input_ids'][0], dtype),
            token_type_ids=np.array(feature['token_type_ids'][0], dtype),
            attention_mask=np.array(feature['attention_mask'][0], dtype),
            input_len=len(feature['input_ids'][0]),
            max_seq_len=seq_length,
        )
        raw_inputs.append(trans_feature)

    input = raw_inputs[0]
    for i in range(1, len(raw_inputs)):
        input = input + raw_inputs[i]
    return input


def test_func(batch_size: int, bps: int, enable_overlapio: bool, precision='fp16'):
    onnx_file = "model.onnx"
    sequence_len = 128
    input_shape = {
        'input_ids': [batch_size, sequence_len],
        'attention_mask': [batch_size, sequence_len],
        'token_type_ids': [batch_size, sequence_len],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        export_bert(dir=tmpdir)
        raw_onnx_file = tmpdir + "/" + onnx_file
        logging.info(
            f"start benchmark for model: {raw_onnx_file},input_shape {input_shape}"
        )
        raw_model = onnx.load(raw_onnx_file)
        model = poprt.Converter(
            input_shape=input_shape,
            precision=precision,
            used_passes=["pre_scale", "fused_attention", "erf_gelu_pattern"],
        ).convert(raw_model)
        options = CompilerOptions()
        options.group_host_sync = True
        if enable_overlapio:
            options.num_io_tiles = 32
        options.batches_per_step = bps
        options.rearrange_anchors_on_host = False
        options.partials_type = 'half'

        options.available_memory_proportion = 0.15
        options.enable_outlining = True
        options.outline_threshold = 10.0

        options.use_128bit_conv_unit_load = True
        options.enable_multi_stage_reduce = False
        options.enable_fast_reduce = True

        # options.enable_engine_caching = True
        # options.cache_path = './cache'

        raw_input_dtype = {
            'input_ids': np.int64,
            "attention_mask": np.int64,
            "token_type_ids": np.int64,
        }

        batch_inputs = gen_bert_inputs(batch_size, sequence_len, np.int32)
        input_lens = batch_inputs.__dict__.pop("input_len")
        batch_inputs = batch_inputs.__dict__

        raw_inputs = get_raw_inputs(batch_inputs, raw_input_dtype)
        benchmark_info = run_benchmark(
            model,
            batch_inputs,
            raw_model=raw_onnx_file,
            raw_inputs=raw_inputs,
            compiler_options=options,
            sequence_lens=input_lens,
        )
        benchmark_info.add_attrs(
            precision=precision,
            batch_size=batch_size,
            size=(batch_size, sequence_len),
            onnx_file=onnx_file,
        )
        logging.info(benchmark_info)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Inference bert with Poprt')
    parser.add_argument(
        '--batch_size', type=int, default=18, help='batch size of inference'
    )
    parser.add_argument('--bps', type=int, default=100, help='batch size of inference')
    parser.add_argument(
        '--enable_overlapio', action='store_true', help='enable io tile'
    )
    args = parser.parse_args()
    test_func(
        batch_size=args.batch_size,
        bps=args.bps,
        enable_overlapio=args.enable_overlapio,
    )
