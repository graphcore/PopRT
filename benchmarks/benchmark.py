# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import json
import logging
import time

from multiprocessing.pool import ThreadPool
from typing import Any, AnyStr, Dict, List, Optional, Union

import numpy as np
import onnx
import onnxruntime

from sklearn import metrics

from poprt import runtime
from poprt.compiler import Compiler, CompilerOptions
from poprt.runtime import RuntimeConfig


def accuracy(cpu_res, ipu_res, sequence_lens: list = None, dump_res: bool = False):
    ipu_total_res = []
    cpu_total_res = []
    for i in range(len(cpu_res)):
        if ipu_res[i].shape != cpu_res[i].shape:
            raise Exception("ipu_res shape does not match to cpu_res shape")
        if sequence_lens is not None:
            for idx in range(len(sequence_lens)):
                seq_len = sequence_lens[idx]
                ipu_total_res.append(
                    ipu_res[i][idx][:seq_len].flatten().astype(np.float32)
                )
                cpu_total_res.append(
                    cpu_res[i][idx][:seq_len].flatten().astype(np.float32)
                )
        else:
            ipu_total_res.append(ipu_res[i].flatten().astype(np.float32))
            cpu_total_res.append(cpu_res[i].flatten().astype(np.float32))

    ipu_total_res = np.concatenate(ipu_total_res, axis=0)
    cpu_total_res = np.concatenate(cpu_total_res, axis=0)
    if dump_res:
        np.savetxt("ipu_result.txt", ipu_total_res)
        np.savetxt("cpu_result.txt", cpu_total_res)
    mse = metrics.mean_squared_error(ipu_total_res, cpu_total_res)
    mae = metrics.mean_absolute_error(ipu_total_res, cpu_total_res)
    dict = {
        'mse': mse,
        'mae': mae,
    }
    return dict


def get_raw_inputs(
    inputs: Dict[str, np.ndarray],
    idtypes: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    raw_inputs = {}
    for k, v in inputs.items():
        raw_inputs[k] = v.astype(idtypes[k])
    return raw_inputs


class BenchmarkOutput:
    def __init__(
        self,
        time_costs: Dict[int, float],
        pop_outputs: Dict[int, List[np.ndarray]],
        ort_outputs: Dict[int, List[np.ndarray]],
        iteration: int,
        sequence_lens: list,
    ):
        self.iteration = iteration

        self.total_time_costs = time_costs[-2]
        self.compile_cost = time_costs[-1]
        self._time_costs = time_costs

        _time_costs = list(time_costs.values())[1:-2]
        self.latency_mean = np.mean(_time_costs)
        self.latency_max = np.max(_time_costs)
        self.latency_min = np.min(_time_costs)
        self.latency_median = np.median(_time_costs)

        self._pop_outputs = pop_outputs
        self._ort_outputs = ort_outputs

        acc = accuracy(ort_outputs[0], pop_outputs[0], sequence_lens)
        for k, v in acc.items():
            k = f'accuracy/{k}'
            self.__setattr__(k, v)

        self.log_type = type(self).__name__

        # optional attributes
        self.precision: Optional[str] = None
        self.batch_size: Optional[int] = None
        # image input shape
        self.size: Optional[List[int]] = None
        self.onnx_file: Optional[str] = None

    def add_attrs(self, **kwargs):
        # optional attributes
        self.precision: Optional[str] = kwargs.pop('precision', self.precision)
        self.batch_size: Optional[int] = kwargs.pop('batch_size', self.batch_size)
        # image input shape
        self.size: Optional[List[int]] = kwargs.pop('size', self.size)
        if self.size:
            self.size = list(self.size)
        self.onnx_file: Optional[str] = kwargs.pop('onnx_file', self.onnx_file)

        # get tput
        if self.batch_size is not None:
            self.tput = self.batch_size * self.iteration / self.total_time_costs

    def __str__(self):
        attrs = {}
        for attr in dir(self):
            if attr.startswith('_') or attr in ['add_attrs']:
                continue
            attrs[attr] = getattr(self, attr)
        attrs = dict(sorted(attrs.items()))
        attrs_str = attrs.__str__()
        attrs_str = attrs_str.replace('None', "'None'")
        return json.dumps(attrs_str)


def log_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f'Exception: {e}')
            raise e

    return wrapper


def init_outputs(outputs_info):
    outputs = {}
    for out_info in outputs_info:
        outputs[out_info.name] = np.zeros(
            out_info.shape,
            dtype=np.float32 if out_info.numpy_data_type() == "float32" else np.float16,
        )
    return outputs


@log_exception
def run_benchmark(
    model: onnx.ModelProto,
    inputs: Dict[str, Any],
    *,
    raw_model: AnyStr = None,
    raw_inputs: Dict[str, Any] = None,
    output_names: list = None,
    num_loops: int = 1000,
    sequence_lens: list = None,
    compiler_options: Union[Dict, CompilerOptions] = CompilerOptions(),
) -> BenchmarkOutput:
    if raw_model is None:
        raw_model = model.SerializeToString()
    if raw_inputs is None:
        raw_inputs = inputs

    if isinstance(compiler_options, dict):
        compiler_options = CompilerOptions.from_dict(compiler_options)
    compiler_options.show_compilation_progress_bar = False

    time_costs = {}
    time_start = time.time()
    if output_names is None:
        output_names = [o.name for o in model.graph.output]
    popef = Compiler.compile(model, outputs=output_names, options=compiler_options)
    config = RuntimeConfig()
    config.ring_buffer_size_multiplier = 5
    model_runner = runtime.Runner(popef, config)
    outputs_info = model_runner.get_execute_outputs()

    time_delta = time.time() - time_start
    time_costs[-1] = time_delta  # compile time

    # warmup
    outputs = init_outputs(outputs_info)
    for i in range(5):
        model_runner.execute(inputs, outputs)

    allocated_outputs = {}
    for i in range(num_loops):
        outputs = init_outputs(outputs_info)
        allocated_outputs[i] = outputs

    pop_outputs = {}
    process_num = 5
    pool = ThreadPool(processes=process_num)

    def execute(process_id):
        len = num_loops // process_num
        latency = {}
        pop_res = {}
        for i in range(len):
            # outputs = init_outputs(outputs_info)
            start = time.time()
            model_runner.execute(inputs, allocated_outputs[process_id * len + i])
            latency[process_id * len + i] = time.time() - start

            if process_id == 0 and i == 0:
                output_list = []
                for name in output_names:
                    output_list.append(outputs[name])
                pop_res[process_id * len + i] = output_list
        return (latency, pop_res)

    async_results = []
    total_time_start = time.time()
    for i in range(process_num):
        async_result = pool.apply_async(execute, (i,))
        async_results.append(async_result)
    for res in async_results:
        latency, output = res.get()
        time_costs.update(latency)
        pop_outputs.update(output)

    total_delta_time = time.time() - total_time_start

    time_costs[-2] = total_delta_time
    ort_session = onnxruntime.InferenceSession(raw_model)
    ort_output = ort_session.run(output_names, raw_inputs)
    ort_outputs = {0: ort_output}

    return BenchmarkOutput(
        time_costs, pop_outputs, ort_outputs, num_loops, sequence_lens
    )
