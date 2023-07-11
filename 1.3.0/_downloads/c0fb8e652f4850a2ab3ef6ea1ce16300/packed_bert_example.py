# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse
import csv
import os
import queue
import random
import sys
import tempfile
import threading
import time

from multiprocessing.pool import ThreadPool
from queue import Queue

import numpy as np
import packing_utils

from sklearn.metrics import mean_absolute_error

from poprt import runtime
from poprt.backend import get_session

np.random.seed(2023)
INPUT_IDS = "input_ids"
POSITION_IDS = "position_ids"
ATTENTION_MASK = "attention_mask"
TOKEN_TYPE_IDS = "token_type_ids"
UNPACK_INFO = "unpack_info"
OUTPUT2 = "start_logits"
OUTPUT1 = "end_logits"


class BertInputs(object):
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        unpack_info,
        input_len,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.input_len = input_len
        self.unpack_info = unpack_info


def get_synthetic_data(args):
    input_len = np.random.normal(
        args.avg_seq_len, args.avg_seq_len, size=args.dataset_size
    ).astype(np.int32)
    input_len = np.clip(input_len, 1, args.max_seq_len)

    datasets = []
    for s_len in input_len:
        input_ids = np.random.randint(0, args.emb_size, (s_len)).astype(np.int32)

        attention_mask = np.ones(s_len).astype(np.int32)
        token_type_ids = np.random.randint(0, 2, (s_len)).astype(np.int32)

        position_ids = np.arange(s_len).astype(np.int32)
        unpack_info = np.zeros(args.max_valid_num).astype(np.int32)

        feature = BertInputs(
            input_ids, attention_mask, token_type_ids, position_ids, unpack_info, s_len
        )
        datasets.append(feature)

    return datasets


def dump_results(model_name, results):
    fieldnames = [OUTPUT1, OUTPUT2]
    filename = os.path.basename(model_name)[:-4] + 'csv'
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for result in results:
            dict_name2list = {
                OUTPUT1: result[OUTPUT1],
                OUTPUT2: result[OUTPUT2],
            }
            writer.writerow(dict_name2list)


## create batched inputs and pad samples to max_seq_len
def padding_data(datasets, index, args):
    feed_dicts = {}
    feed_dicts[INPUT_IDS] = np.zeros(
        (args.batch_size, args.max_seq_len), dtype=np.int32
    )
    feed_dicts[ATTENTION_MASK] = np.zeros(
        (args.batch_size, args.max_seq_len), dtype=np.int32
    )
    feed_dicts[POSITION_IDS] = np.zeros(
        (args.batch_size, args.max_seq_len), dtype=np.int32
    )
    feed_dicts[TOKEN_TYPE_IDS] = np.zeros(
        (args.batch_size, args.max_seq_len), dtype=np.int32
    )

    for i in range(args.batch_size):
        input_len = datasets[index].input_len
        feed_dicts[INPUT_IDS][i][:input_len] = datasets[index].input_ids
        feed_dicts[ATTENTION_MASK][i][:input_len] = datasets[index].attention_mask
        feed_dicts[POSITION_IDS][i][:input_len] = datasets[index].position_ids
        feed_dicts[TOKEN_TYPE_IDS][i][:input_len] = datasets[index].token_type_ids
        index = index + 1
    return feed_dicts


# online pack, samples feeded to IPU can reach to maximum num of batches in each running turn
def run_packing_model_with_pack_runner_unpack_repack(args, datasets):
    tmpdir = tempfile.TemporaryDirectory()
    # export popef for PackRunner
    get_session(
        args.model_with_packing_unpack_repack,
        1,
        "poprt",
        output_dir=tmpdir.name,
        export_popef=True,
    ).load()
    config = runtime.PackRunnerConfig(
        timeout_microseconds=args.timeout_microseconds,
        # max_valid_num=args.max_valid_num,
        # dynamic_input_name=args.dynamic_input_name,
    )

    popef_path = tmpdir.name + '/executable.popef'
    # popef_path = "/popconverter/examples/packed_bert_example/executable.popef"
    pack_runner = runtime.Runner(popef_path, config)

    result_queue = queue.Queue()
    results = []
    start_time = time.time()
    for i in range(args.dataset_size):
        feed_dicts = {
            INPUT_IDS: datasets[i].input_ids,
            ATTENTION_MASK: datasets[i].attention_mask,
            TOKEN_TYPE_IDS: datasets[i].token_type_ids,
            POSITION_IDS: datasets[i].position_ids,
            # unpack_info should be hidden from user in the future
            UNPACK_INFO: np.zeros(args.max_valid_num).astype(np.int32),
        }
        out_dict = {
            OUTPUT1: np.zeros([args.max_seq_len]).astype(np.float16),
            OUTPUT2: np.zeros([args.max_seq_len]).astype(np.float16),
        }
        future = pack_runner.execute_async(feed_dicts, out_dict)
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
    print(
        f"[Pack Online Unpack Repack] Throughput: {tput} samples/s, Latency : {latency_ms * 1000} ms"
    )

    if args.dump_results:
        dump_results(
            "online_unpack_repack" + args.model_with_packing_unpack_repack, results
        )

    tmpdir.cleanup()
    return results


# offline pack, samples feeded to IPU can reach to maximum num of batches in each running turn
# model with pack / unpack ops
def run_packing_model_with_model_runner(args, datasets, model_path, across_rows):
    run_queue = queue.Queue()
    start_time = time.time()
    index = 0
    for i in range(0, args.dataset_size):
        transfer = packing_utils.pack_data(
            datasets,
            index,
            args.batch_size,
            seq_len=256,
            max_valid_num=args.max_valid_num,
            segment_num=1,
            across_rows=across_rows,
        )

        run_queue.put(transfer)
        index = transfer.count
        if index == args.dataset_size:
            break
    run_queue.put(None)
    duration_of_packing = time.time() - start_time
    mean_latency_of_padding_us = duration_of_packing * 1e6 / args.dataset_size

    print(f"Mean latency of packing data: {mean_latency_of_padding_us} us/sam")
    print(f"Total latency of packing data: {duration_of_packing} s")

    sess = get_session(model_path, 1, "poprt").load()

    pool = ThreadPool(processes=1)

    def execute(feed_dicts, valid_num):
        outputs = sess.run([OUTPUT1, OUTPUT2], feed_dicts)
        res = []
        if across_rows:
            for i in range(valid_num):
                res1 = outputs[0][i].copy().tolist()
                res2 = outputs[1][i].copy().tolist()
                res.append({OUTPUT1: res1, OUTPUT2: res2})
        else:
            outlen = len(outputs[0][0])
            for index in range(len(feed_dicts[ATTENTION_MASK])):
                start = 0
                arr = np.array(feed_dicts[ATTENTION_MASK][index])
                while start < outlen and arr[start] > 0:
                    arr = arr - 1
                    zero_num = len(arr) - np.count_nonzero(arr)
                    out1 = [0] * outlen
                    out2 = [0] * outlen
                    out1[:zero_num] = outputs[0][index][start : start + zero_num]
                    out2[:zero_num] = outputs[1][index][start : start + zero_num]
                    res.append({OUTPUT1: out1, OUTPUT2: out2})
                    start += zero_num
        return res

    asy_results = []

    total_start_time = time.time()
    while True:
        input_data = run_queue.get()
        if input_data is None:
            break

        feed_dicts = {
            INPUT_IDS: input_data.data[INPUT_IDS],
            ATTENTION_MASK: input_data.data[ATTENTION_MASK],
            TOKEN_TYPE_IDS: input_data.data[TOKEN_TYPE_IDS],
            POSITION_IDS: input_data.data[POSITION_IDS],
            # unpack_info should be hidden from user in the future
            UNPACK_INFO: input_data.unpack_info,
        }
        if not across_rows:
            feed_dicts.pop(UNPACK_INFO)

        valid_num = len(input_data.specs)
        async_result = pool.apply_async(execute, (feed_dicts, valid_num))
        asy_results.append(async_result)

    results = []
    for asy in asy_results:
        for res in asy.get():
            results.append(res)
    total_end_time = time.time()

    tput = len(results) / (total_end_time - total_start_time)
    latency = (total_end_time - total_start_time) / len(results)
    if across_rows:
        print(
            f"[Pack Offline Unpack Repack] Throughput: {tput} samples/s, Latency: {latency*1000} ms"
        )
    else:
        print(
            f"[Pack Offline AttentionMask] Throughput: {tput} samples/s, Latency: {latency*1000} ms"
        )

    if args.dump_results:
        dump_results("offline_" + model_path, results)

    return results


# online pack, samples feeded to IPU can reach to maximum num of batches in each running turn
# model only add AttentionMask op in this mode
def run_packing_model_with_pack_runner_attention_mask(args, datasets, algo):
    tmpdir = tempfile.TemporaryDirectory()
    # export popef for PackRunner
    get_session(
        args.model_with_packing_attention_mask,
        1,
        "poprt",
        output_dir=tmpdir.name,
        export_popef=True,
    ).load()
    config = runtime.PackRunnerConfig(
        timeout_microseconds=args.timeout_microseconds,
        max_valid_num=args.max_valid_num,
        dynamic_input_name=args.dynamic_input_name,
    )

    if algo == "next_fit":
        config.algorithm = runtime.PackAlgorithm.next_fit
    else:
        config.algorithm = runtime.PackAlgorithm.first_fit

    config.enable_input_single_row_mode("attention_mask")
    popef_path = tmpdir.name + '/executable.popef'
    # popef_path = "/popconverter/examples/packed_bert_example/executable.popef"
    pack_runner = runtime.Runner(popef_path, config)

    result_queue = queue.Queue()
    results = []
    start_time = time.time()
    for i in range(args.dataset_size):
        feed_dicts = {
            INPUT_IDS: datasets[i].input_ids,
            ATTENTION_MASK: datasets[i].attention_mask,
            TOKEN_TYPE_IDS: datasets[i].token_type_ids,
            # position_ids is an optional input for first_fit/next_fit mode
            # POSITION_IDS: datasets[i].position_ids,
        }
        out_dict = {
            OUTPUT1: np.zeros([args.max_seq_len]).astype(np.float16),
            OUTPUT2: np.zeros([args.max_seq_len]).astype(np.float16),
        }
        future = pack_runner.execute_async(feed_dicts, out_dict)
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
    print(
        f"[Pack Online AttentionMask({algo})] Throughput: {tput} samples/s, Latency : {latency_ms * 1000} ms"
    )

    if args.dump_results:
        dump_results(
            "online_attention_mask_"
            + algo
            + "_"
            + args.model_with_packing_attention_mask,
            results,
        )

    tmpdir.cleanup()
    return results


def latency_distribuion_with_pack_runner_attention_mask(args, datasets, algo):
    tmpdir = tempfile.TemporaryDirectory()
    # export popef for PackRunner
    get_session(
        args.model_with_packing_attention_mask,
        1,
        "poprt",
        output_dir=tmpdir.name,
        export_popef=True,
    ).load()
    config = runtime.PackRunnerConfig(
        timeout_microseconds=args.timeout_microseconds,
        max_valid_num=args.max_valid_num,
        dynamic_input_name=args.dynamic_input_name,
    )

    if algo == "next_fit":
        config.algorithm = runtime.PackAlgorithm.next_fit
    else:
        config.algorithm = runtime.PackAlgorithm.first_fit

    config.enable_input_single_row_mode("attention_mask")
    popef_path = tmpdir.name + '/executable.popef'
    # popef_path = "/popconverter/examples/packed_bert_example/executable.popef"
    pack_runner = runtime.Runner(popef_path, config)

    sample_num = args.batch_size * args.iterations
    clients = int(args.batch_size * 3.5)
    count_percent = 0.6

    q = Queue()

    def perf_count(model_runner, iteration):
        durations = []
        for i in range(sample_num):
            start_time = time.time()
            random.randint(0, sample_num)
            feed_dicts = {
                INPUT_IDS: datasets[i].input_ids,
                ATTENTION_MASK: datasets[i].attention_mask,
                TOKEN_TYPE_IDS: datasets[i].token_type_ids,
            }
            out_dict = {
                OUTPUT1: np.zeros([args.max_seq_len]).astype(np.float16),
                OUTPUT2: np.zeros([args.max_seq_len]).astype(np.float16),
            }
            pack_runner.execute(feed_dicts, out_dict)
            end_time = time.time()
            durations.append((start_time, end_time))
        # remove first and last example's time counter
        ignored_samples = int(sample_num * (1 - count_percent) / 2)
        durations = durations[ignored_samples:-ignored_samples]
        q.put(durations, timeout=10)

    thp = [
        threading.Thread(target=perf_count, args=(pack_runner, args.iterations))
        for _ in range(clients)
    ]
    for t in thp:
        t.start()
    for t in thp:
        t.join()

    durations_from_th = []
    while not q.empty():
        durations_from_th += q.get()
    max_timestamp = max(y for _, y in durations_from_th)
    min_timestamp = min(x for x, _ in durations_from_th)
    clients * (sample_num * count_percent) / (max_timestamp - min_timestamp)
    times_range = [y - x for x, y in durations_from_th]

    times_range.sort()
    tail_latency = round(times_range[int(len(times_range) * 0.99)] * 1000, 2)
    avg_latency = round(sum(times_range) / len(times_range) * 1000, 2)

    print(f"Average Latency: {avg_latency}ms, P99 latency: {tail_latency}ms.")
    return tail_latency, avg_latency


# no pack, padding each line with 0 if input length is not long enough.
# samples num equals to batch at every running turn
def run_original_model_with_model_runner(args, datasets):
    run_queue = queue.Queue()
    start_time = time.time()
    for i in range(0, args.dataset_size, args.batch_size):
        feed_dicts = padding_data(datasets, i, args)
        run_queue.put((args.batch_size, feed_dicts))
    run_queue.put((0, None))
    duration_of_padding_s = time.time() - start_time

    mean_latency_of_padding_us = duration_of_padding_s * 1e6 / args.dataset_size
    print(f"Mean latency of padding data: {mean_latency_of_padding_us} us/sam")
    print(f"Total latency of padding data: {duration_of_padding_s} s")

    sess = get_session(args.model_without_packing, 1, "poprt").load()

    asy_results = []

    def execute(feed_dicts, valid_num):
        outputs = sess.run([OUTPUT1, OUTPUT2], feed_dicts)
        res = []
        for i in range(valid_num):
            res1 = outputs[0][i].copy().tolist()
            res2 = outputs[1][i].copy().tolist()
            res.append({OUTPUT1: res1, OUTPUT2: res2})
        return res

    # execute
    pool = ThreadPool(processes=1)
    total_start_time = time.time()
    while True:
        valid_num, feed_dicts = run_queue.get()
        if feed_dicts is None:
            break
        async_result = pool.apply_async(execute, (feed_dicts, valid_num))
        asy_results.append(async_result)
    results = []
    for asy in asy_results:
        for res in asy.get():
            results.append(res)
    total_end_time = time.time()

    tput = len(results) / (total_end_time - total_start_time)
    latency = (total_end_time - total_start_time) / len(results)

    if args.dump_results:
        dump_results("original_" + args.model_without_packing, results)

    print(f"[Original] Throughput: {tput} samples/s, Latency: {latency *1000} ms")

    return results


def calculate_mae(expected_results, output_results, datasets, enable_debug):
    assert len(datasets) == len(expected_results)
    assert len(datasets) == len(output_results)
    maes = []
    zipped_data = zip(datasets, expected_results, output_results)
    for i, (data, expected, output) in enumerate(zipped_data):
        np.testing.assert_equal(len(expected), len(output))
        input_len = data.input_len
        output_1_mae = mean_absolute_error(
            expected[OUTPUT1][:input_len], output[OUTPUT1][:input_len]
        )
        output_2_mae = mean_absolute_error(
            expected[OUTPUT2][:input_len], output[OUTPUT2][:input_len]
        )
        maes.append([i, output_1_mae, output_2_mae])

    k = 10 if len(datasets) > 10 else len(datasets)

    def print_topk(k, out_name, out_index):
        for i in range(1, k + 1):
            print(f"Sample: {maes[-i][0]}, {out_name} mae : {maes[-i][out_index]}")

    if enable_debug:
        maes.sort(key=lambda e: e[1])
        print(f"\n***** Top {k} mae of output: {OUTPUT1} *****")
        print_topk(k, OUTPUT1, 1)

        maes.sort(key=lambda e: e[2])
        print(f"\n***** Top {k} mae of output: {OUTPUT2} *****")
        print_topk(k, OUTPUT2, 2)

    print(f"{OUTPUT1} average mae: {np.mean(maes,axis=0)[1]}")
    print(f"{OUTPUT2} average mae: {np.mean(maes,axis=0)[2]}")


def main():
    parser = argparse.ArgumentParser(description='packed bert-base-squad')
    parser.add_argument(
        '--avg_seq_len', type=int, default=128, help='average sequence length of input'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16, help='batch size of model'
    )
    parser.add_argument('--dump_results', action='store_true', help='dump results')
    parser.add_argument(
        '--dynamic_input_name', type=str, default=INPUT_IDS, help='dynamic input name'
    )
    parser.add_argument(
        '--emb_size', type=int, default=30522, help='word embedding table size'
    )
    parser.add_argument(
        '--enable_debug', action='store_true', help='enable output debug info'
    )
    parser.add_argument(
        '--iterations', type=int, default=100, help='number of batches to run'
    )
    parser.add_argument(
        '--max_seq_len', type=int, default=256, help='max sequence length of input'
    )
    parser.add_argument(
        '--max_valid_num', type=int, default=40, help='max valid num for pack'
    )
    parser.add_argument(
        '--model_without_packing', help='model without pack, unpack, repack op'
    )
    parser.add_argument(
        '--model_with_packing_unpack_repack',
        help='model with pack, unpack, repack op converted by PopRT',
    )
    parser.add_argument(
        '--model_with_packing_attention_mask',
        help='model with AttentionMask op converted by PopRT',
    )
    parser.add_argument(
        '--timeout_microseconds',
        type=int,
        default=15000,
        help='timeout in microseconds',
    )

    args = parser.parse_args()
    args.dataset_size = args.iterations * args.batch_size

    # generate synthetic dataset
    datasets = get_synthetic_data(args)
    original_result = run_original_model_with_model_runner(args, datasets)

    offline_pack_result_unpack_repack = run_packing_model_with_model_runner(
        args, datasets, args.model_with_packing_unpack_repack, True
    )
    online_pack_result_unpack_repack = run_packing_model_with_pack_runner_unpack_repack(
        args, datasets
    )

    offline_pack_result_attention_mask = run_packing_model_with_model_runner(
        args, datasets, args.model_with_packing_attention_mask, False
    )
    online_pack_result_attention_mask_first_fit = (
        run_packing_model_with_pack_runner_attention_mask(args, datasets, "first_fit")
    )
    online_pack_result_attention_mask_next_fit = (
        run_packing_model_with_pack_runner_attention_mask(args, datasets, "next_fit")
    )
    latency_distribuion_with_pack_runner_attention_mask(args, datasets, "first_fit")

    # compare the results
    print("\nCompare results between original and online pack(with unpack repack)")
    calculate_mae(
        original_result, online_pack_result_unpack_repack, datasets, args.enable_debug
    )
    print("\nCompare results between offline and online pack with unpack repack op")
    calculate_mae(
        offline_pack_result_unpack_repack,
        online_pack_result_unpack_repack,
        datasets,
        args.enable_debug,
    )

    print(
        "\nCompare results between original and online_first_fit pack with attention_mask op"
    )
    calculate_mae(
        original_result,
        online_pack_result_attention_mask_first_fit,
        datasets,
        args.enable_debug,
    )
    print(
        "\nCompare results between original and online_next_fit pack with attention_mask op"
    )
    calculate_mae(
        original_result,
        online_pack_result_attention_mask_next_fit,
        datasets,
        args.enable_debug,
    )

    print(
        "\nCompare results between offline and online_next_fit pack with attenttion_mask op"
    )
    calculate_mae(
        offline_pack_result_attention_mask,
        online_pack_result_attention_mask_next_fit,
        datasets,
        args.enable_debug,
    )


if __name__ == "__main__":
    sys.exit(main())
