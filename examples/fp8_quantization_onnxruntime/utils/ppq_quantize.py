# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import onnx
import ppq.lib as PFL

from ppq import TargetPlatform, TorchExecutor
from ppq.api import load_onnx_graph
from ppq.core import QuantizationStates
from ppq.IR.quantize import QuantableOperation
from ppq.quantization.optim import (
    ParameterBakingPass,
    ParameterQuantizePass,
    RuntimeCalibrationPass,
)


def get_first_last_conv_name(model_path: str, skip_layer: str) -> str:
    # For the cv model, we need to set the first and last conv/matmul/gemm to fp16,
    # For the nlp model, we need to set the last matmul/gemm to fp16.
    first_conv_name = ''
    last_conv_name = ''
    onnx_model = onnx.load(model_path)
    for node in onnx_model.graph.node:
        if node.op_type in ['Conv', 'MatMul', 'Gemm']:
            # Temporarily take the last conv/matmul/gemm in nodes as the last conv/matmul/gemm of the model
            if not first_conv_name:
                first_conv_name = node.name
            last_conv_name = node.name
    if skip_layer == "first":
        return first_conv_name
    elif skip_layer == "last":
        return last_conv_name
    elif skip_layer == "both":
        return first_conv_name + ',' + last_conv_name
    else:
        return []


def reset_ppq_scale(graph, scale):
    for op in [op for op in graph.operations.values()]:
        if not isinstance(op, QuantableOperation):
            continue
        for config, var in zip(op.input_quant_config, op.inputs):
            if config.scale is None:
                continue
            temp_state = config.state
            config.state = QuantizationStates.ACTIVATED
            config.scale = scale
            config.state = temp_state


def quantize(onnx_model_path, dataloader, dummy_input, scale, dispatching_ops):
    """ppq fp8 quantization."""
    graph = load_onnx_graph(onnx_model_path)
    # get quantizer
    quantizer = PFL.Quantizer(platform=TargetPlatform.GRAPHCORE_FP8, graph=graph)
    # generate scheduling table
    dispatching = PFL.Dispatcher(graph=graph).dispatch(
        quant_types=quantizer.quant_operation_types
    )
    # send dispatching_ops to the non-quantized area
    for name_ in dispatching_ops:
        dispatching[name_] = TargetPlatform.FP32
    for op in graph.operations.values():
        quantizer.quantize_operation(op_name=op.name, platform=dispatching[op.name])

    executor = TorchExecutor(graph=graph, device='cpu')
    executor.tracing_operation_meta(inputs=dummy_input)
    executor.load_graph(graph=graph)
    pipeline = PFL.Pipeline(
        [ParameterQuantizePass(), RuntimeCalibrationPass(), ParameterBakingPass()]
    )
    # call the pipeline to complete quantization
    pipeline.optimize(
        graph=graph,
        dataloader=dataloader,
        verbose=True,
        calib_steps=8,
        executor=executor,
    )
    # rest scale
    reset_ppq_scale(graph, scale)
    return graph
