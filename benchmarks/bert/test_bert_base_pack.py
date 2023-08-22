# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import logging
import os
import queue
import sys
import tempfile
import time

from typing import Any

import numpy as np
import onnx

import poprt

from poprt import runtime
from poprt.compiler import Compiler, CompilerOptions

sys.path.append("..")
import helper
import onnxruntime

np.random.seed(2023)


def export_bert(dir):
    os.system(
        f"python -m transformers.onnx --model=csarron/bert-base-uncased-squad-v1 {dir} --feature question-answering"
    )
    os.system(
        f"python ../../examples/packed_bert_example/add_position_ids.py --input_model {dir}/model.onnx"
    )


class BertInputs(object):
    def __init__(
        self, input_ids, attention_mask, token_type_ids, position_ids, input_len
    ):
        self.input_ids = np.expand_dims(input_ids, axis=0)
        self.attention_mask = np.expand_dims(attention_mask, axis=0)
        self.token_type_ids = np.expand_dims(token_type_ids, axis=0)
        self.position_ids = np.expand_dims(position_ids, axis=0)
        self.input_len = [input_len]


def gen_datasets(args, dtype: Any, random_data: bool = True):
    raw_inputs = []
    if random_data:
        print(
            f"use random data,avg_seq_len: {args.avg_seq_len}, max_seq_len: {args.max_seq_len}"
        )
        input_len = np.random.normal(
            args.avg_seq_len, args.avg_seq_len, size=args.dataset_size
        ).astype(np.int32)
        input_len = np.clip(input_len, 1, args.max_seq_len)

        for s_len in input_len:
            input_ids = np.random.randint(0, args.emb_size, (s_len)).astype(np.int32)

            attention_mask = np.ones(s_len).astype(np.int32)
            token_type_ids = np.random.randint(0, 1, (s_len)).astype(np.int32)

            position_ids = np.arange(s_len).astype(np.int32)
            feature = BertInputs(
                input_ids, attention_mask, token_type_ids, position_ids, s_len
            )
            raw_inputs.append(feature)
        return raw_inputs
    else:
        from datasets import load_dataset
        from transformers import BertTokenizer

        raw_question = load_dataset("squad")
        tokenizer = BertTokenizer.from_pretrained('csarron/bert-base-uncased-squad-v1')
        text = [
            raw_question["validation"][i]["question"] for i in range(args.dataset_size)
        ]
        tokenizer = BertTokenizer.from_pretrained('csarron/bert-base-uncased-squad-v1')

        for elem in text[: args.dataset_size]:
            feature = tokenizer(
                elem,
                return_tensors='np',
                max_length=args.max_seq_len,
                truncation=True,
            )
            trans_feature = BertInputs(
                input_ids=np.array(feature['input_ids'][0], dtype),
                token_type_ids=np.array(feature['token_type_ids'][0], dtype),
                attention_mask=np.array(feature['attention_mask'][0], dtype),
                input_len=len(feature['input_ids'][0]),
                position_ids=np.arange(len(feature['input_ids'][0])).astype(dtype),
            )

            raw_inputs.append(trans_feature)

    return raw_inputs


def run_bert_pack(args, popef, datasets, raw_model, out_names):
    config = runtime.PackRunnerConfig(
        timeout_microseconds=args.timeout_microseconds,
        max_valid_num=args.max_valid_num,
        dynamic_input_name=args.dynamic_input_name,
    )
    # here we use first_fit algo
    config.algorithm = runtime.PackAlgorithm.first_fit
    config.enable_input_single_row_mode("attention_mask")
    pack_runner = runtime.Runner(popef, config)

    result_queue = queue.Queue()
    results = []
    start_time = time.time()
    for i in range(args.dataset_size):
        feed_dicts = {
            "input_ids": datasets[i].input_ids,
            "attention_mask": datasets[i].attention_mask,
            "token_type_ids": datasets[i].token_type_ids,
            "position_ids": datasets[i].position_ids,
        }
        out_dict = {
            "start_logits": np.zeros([args.max_seq_len]).astype(np.float16),
            "end_logits": np.zeros([args.max_seq_len]).astype(np.float16),
        }
        future = pack_runner.execute_async(feed_dicts, out_dict)
        out_dict = {k: v[: datasets[i].input_len[0]] for k, v in out_dict.items()}
        result_queue.put((future, out_dict))
    result_queue.put((None, None))
    while True:
        future, out_dict = result_queue.get()
        if future == None:
            break
        future.wait()
        results.append(out_dict)
    end_time = time.time()
    tput = args.dataset_size / (end_time - start_time)
    latency_ms = (end_time - start_time) / args.dataset_size
    logging.info(
        f"Pack bert_base, Batch Size: {args.batch_size} Throughput: {tput} samples/s, Latency : {latency_ms * 1000} ms"
    )

    if not args.random_data:
        ort_outputs = []
        for raw_input in datasets[:5]:
            raw_input = raw_input.__dict__
            raw_input.pop("input_len")
            raw_input.pop("position_ids")

            input = {}
            for k, v in raw_input.items():
                input[k] = np.array(v).astype(np.int64)
            ort_session = onnxruntime.InferenceSession(raw_model)
            ort_output = ort_session.run(out_names, input)
            ort_outputs.append([np.array(elem).flatten() for elem in ort_output])

        ipu_outputs = [list(e.values()) for e in results[:5]]
        ipu_outputs = np.concatenate(
            [np.concatenate((sl, el)) for sl, el in ipu_outputs]
        )
        ort_outputs = np.concatenate(
            [np.concatenate((sl, el)) for sl, el in ort_outputs]
        )
        helper.accuracy(ort_outputs, ipu_outputs)


def test_func(args):
    batch_size = args.batch_size
    bps = args.bps
    enable_overlapio = args.enable_overlapio
    sequence_len = args.max_seq_len
    input_shape = {
        'input_ids': [batch_size, sequence_len],
        'attention_mask': [batch_size, sequence_len],
        'token_type_ids': [batch_size, sequence_len],
        'position_ids': [batch_size, sequence_len],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = "model.onnx"
        bert_base_pos_name = "bert_base_squad_pos.onnx"
        export_bert(dir=tmpdir)
        raw_onnx_file = tmpdir + "/" + onnx_file

        logging.info(
            f"start benchmark for model: {raw_onnx_file},input_shape {input_shape}"
        )
        bert_base_pos_onnx = onnx.load(bert_base_pos_name)
        model = poprt.Converter(
            input_shape=input_shape,
            precision='fp16',
            used_passes=[
                "insert_attention_mask",
                "pre_scale",
                "fused_attention",
                "erf_gelu_pattern",
            ],
        ).convert(bert_base_pos_onnx)
        options = CompilerOptions()
        options.group_host_sync = False
        options.enable_prefetch_datastreams = False
        if enable_overlapio:
            options.num_io_tiles = 32
        options.batches_per_step = bps
        options.partials_type = 'half'

        options.use_128bit_conv_unit_load = True
        options.enable_multi_stage_reduce = False
        options.enable_fast_reduce = True
        options.available_memory_proportion = 0.4

        out_names = [o.name for o in model.graph.output]
        popef = Compiler.compile(model, out_names, options)
        datasets = gen_datasets(args, np.int32, random_data=args.random_data)
        run_bert_pack(args, popef, datasets, raw_onnx_file, out_names)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Inference bert with Poprt')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch size of inference'
    )
    parser.add_argument('--random_data', action='store_true', help='if use random data')
    parser.add_argument('--bps', type=int, default=100, help='batch per step')
    parser.add_argument('--dataset_size', type=int, default=1000, help='dataset size')
    parser.add_argument(
        '--dynamic_input_name',
        type=str,
        default="input_ids",
        help='input lenght that is dynamic',
    )
    parser.add_argument(
        '--max_seq_len', type=int, default=128, help='max sequence length'
    )
    parser.add_argument(
        '--avg_seq_len', type=int, default=64, help='average sequence length of input'
    )
    parser.add_argument('--max_valid_num', type=int, default=64, help='max valid num')
    parser.add_argument(
        '--emb_size', type=int, default=30522, help='word embedding table size'
    )
    parser.add_argument(
        '--timeout_microseconds',
        type=int,
        default=1500,
        help='longest time cpu pack wait',
    )
    parser.add_argument(
        '--enable_overlapio', action='store_true', help='enable io tile'
    )
    args = parser.parse_args()
    test_func(args)
